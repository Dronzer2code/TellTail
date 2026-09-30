/**
 * TELLTAIL attestation bridge.
 *
 *   npm run bridge           # poll forever
 *   npm run bridge:once      # drain once and exit
 *   npm run bridge:dry       # claim nothing, sign nothing, print the payloads
 *
 * THE SECURITY PROPERTY, which is the whole reason this is a separate process:
 * THE KEYPAIR NEVER TOUCHES SNOWFLAKE. The warehouse stages claims in
 * ORACLE.PUBLISH_QUEUE. This process holds the key, signs, submits to devnet,
 * and writes the transaction signature back. A full dump of the warehouse
 * yields no key material, and Snowflake has no code path that can sign
 * anything. That is checkable rather than asserted: grep the SQL for a private
 * key and there is nothing to find.
 *
 * WHY OUTBOUND POLLING rather than Snowflake calling out: external access
 * integrations are not supported on trial accounts (error 509009). An outbound
 * queue-poll needs no integration, no network rule and no secret object.
 *
 * WHY MEMO TRANSACTIONS: a memo is a real, confirmed, explorer-visible
 * transaction carrying the claim in its instruction data. If the Anchor program
 * in oracle-program/ deploys with time to spare, set TELLTAIL_PROGRAM_ID and
 * this switches to a PDA per subject. If not, memos are honest and the README
 * says exactly that.
 *
 * The state machine lives in SQL (ORACLE.SP_CLAIM_BATCH / SP_MARK_CONFIRMED /
 * SP_MARK_FAILED), so this file never writes ad hoc SQL and two bridges polling
 * the same queue cannot double-publish: CLAIM flips rows to SENT before
 * anything is signed, so the second claim returns nothing.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  Connection,
  Keypair,
  LAMPORTS_PER_SOL,
  PublicKey,
  SystemProgram,
  Transaction,
  TransactionInstruction,
} from "@solana/web3.js";
import dotenv from "dotenv";
import snowflake from "snowflake-sdk";

dotenv.config();

const MEMO_PROGRAM_ID = new PublicKey("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr");

const ONCE = process.argv.includes("--once");
const DRY = process.argv.includes("--dry-run");
const POLL_MS = Number(process.env.BRIDGE_POLL_SECONDS || 15) * 1000;
const BATCH = Number(process.env.BRIDGE_BATCH_SIZE || 5);
const RPC_URL = process.env.SOLANA_RPC_URL || "https://api.devnet.solana.com";
const PROGRAM_ID = (process.env.TELLTAIL_PROGRAM_ID || "").trim();
const AUTHORITY = (process.env.SOLANA_AUTHORITY || "").trim();

snowflake.configure({ logLevel: "ERROR" });

// ---------------------------------------------------------------------------
// Snowflake
// ---------------------------------------------------------------------------

function connectSnowflake() {
  const required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"];
  const missing = required.filter((k) => !process.env[k]);
  if (missing.length) {
    console.error(`missing env: ${missing.join(", ")}`);
    console.error("cp .env.example .env and fill it in");
    process.exit(1);
  }

  const conn = snowflake.createConnection({
    account: process.env.SNOWFLAKE_ACCOUNT,
    username: process.env.SNOWFLAKE_USER,
    password: process.env.SNOWFLAKE_PASSWORD,
    role: process.env.SNOWFLAKE_ROLE || "SYSADMIN",
    warehouse: process.env.SNOWFLAKE_WAREHOUSE || "TELLTAIL_WH",
    database: process.env.SNOWFLAKE_DATABASE || "TELLTAIL",
    schema: "ORACLE",
    timezone: "Etc/UTC",
  });

  return new Promise((res, rej) =>
    conn.connect((err) => (err ? rej(err) : res(conn)))
  );
}

function sql(conn, sqlText, binds = []) {
  return new Promise((res, rej) =>
    conn.execute({
      sqlText,
      binds,
      complete: (err, _stmt, rowsOut) => (err ? rej(err) : res(rowsOut || [])),
    })
  );
}

// ---------------------------------------------------------------------------
// Solana
// ---------------------------------------------------------------------------

function loadKeypair() {
  const path = resolve(process.env.SOLANA_KEYPAIR_PATH || "./secrets/devnet.json");
  let raw;
  try {
    raw = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    console.error(`cannot read keypair at ${path}`);
    console.error("generate one:  npm run keygen");
    process.exit(1);
  }
  return Keypair.fromSecretKey(Uint8Array.from(raw));
}

/**
 * Build the instruction that carries the claim.
 *
 * Memo mode: the payload is the instruction data of an SPL Memo instruction, so
 * it is visible on any explorer without a custom decoder.
 *
 * Program mode: a PDA is derived per subject from the dog hash, so all findings
 * for one animal accumulate at a deterministic address that a shelter could look
 * up without knowing anything but the hash.
 */
function buildInstruction(payer, payloadJson, subject) {
  const data = Buffer.from(payloadJson, "utf8");

  if (!PROGRAM_ID) {
    return {
      mode: "memo",
      ix: new TransactionInstruction({
        keys: [{ pubkey: payer, isSigner: true, isWritable: false }],
        programId: MEMO_PROGRAM_ID,
        data,
      }),
      pda: null,
    };
  }

  const programId = new PublicKey(PROGRAM_ID);
  const [pda] = PublicKey.findProgramAddressSync(
    [Buffer.from("telltail"), Buffer.from(subject.slice(0, 16), "utf8")],
    programId
  );
  return {
    mode: "program",
    ix: new TransactionInstruction({
      keys: [
        { pubkey: pda, isSigner: false, isWritable: true },
        { pubkey: payer, isSigner: true, isWritable: true },
        { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
      ],
      programId,
      data,
    }),
    pda: pda.toBase58(),
  };
}

async function publish(conn, connection, keypair, row) {
  const publishId = row.PUBLISH_ID;
  const parsed = typeof row.PAYLOAD === "string" ? JSON.parse(row.PAYLOAD) : row.PAYLOAD;

  // The publishing authority travels with the claim. Snowflake builds the
  // clinical half of the payload and knows nothing about wallets; the bridge
  // adds who published it, because the bridge is what publishes. A reader can
  // then check the claim came from the wallet TELLTAIL says it operates,
  // independently of which key happened to pay the fee.
  if (AUTHORITY) parsed.authority = AUTHORITY;
  const payload = JSON.stringify(parsed);

  const { mode, ix, pda } = buildInstruction(keypair.publicKey, payload, parsed.subject || "");

  if (DRY) {
    console.log(`  [dry] #${publishId}  ${mode}${pda ? ` pda=${pda}` : ""}`);
    console.log(`        ${payload}`);
    return { ok: true, dry: true };
  }

  try {
    const tx = new Transaction().add(ix);
    const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
    tx.recentBlockhash = blockhash;
    tx.feePayer = keypair.publicKey;
    tx.sign(keypair);

    const signature = await connection.sendRawTransaction(tx.serialize(), {
      skipPreflight: false,
      maxRetries: 3,
    });
    const conf = await connection.confirmTransaction(
      { signature, blockhash, lastValidBlockHeight },
      "confirmed"
    );
    if (conf.value?.err) {
      throw new Error(`transaction returned an error: ${JSON.stringify(conf.value.err)}`);
    }

    const detail = await connection.getTransaction(signature, {
      maxSupportedTransactionVersion: 0,
    });
    const slot = detail?.slot ?? 0;
    const explorer = `https://explorer.solana.com/tx/${signature}?cluster=devnet`;

    await sql(conn, "CALL ORACLE.SP_MARK_CONFIRMED(?, ?, ?, ?)", [
      publishId,
      signature,
      slot,
      explorer,
    ]);

    console.log(`  ✓ #${publishId}  ${parsed.finding} sev${parsed.severity}  slot ${slot}`);
    console.log(`      ${explorer}`);
    return { ok: true, signature };
  } catch (err) {
    const message = String(err?.message || err).slice(0, 480);
    await sql(conn, "CALL ORACLE.SP_MARK_FAILED(?, ?)", [publishId, message]);
    console.error(`  ✗ #${publishId}  ${message}`);
    return { ok: false, error: message };
  }
}

// ---------------------------------------------------------------------------

async function drain(conn, connection, keypair) {
  // In dry-run, read PENDING without claiming, so nothing changes state.
  const claimed = DRY
    ? await sql(
        conn,
        `SELECT publish_id, payload FROM ORACLE.PUBLISH_QUEUE
          WHERE status = 'PENDING' AND attempts < 5
          ORDER BY severity DESC, queued_at ASC LIMIT ?`,
        [BATCH]
      )
    : await sql(conn, "CALL ORACLE.SP_CLAIM_BATCH(?)", [BATCH]);

  if (!claimed.length) return 0;

  console.log(`[${new Date().toISOString()}] ${claimed.length} claim(s)`);
  let ok = 0;
  for (const row of claimed) {
    const r = await publish(conn, connection, keypair, row);
    if (r.ok) ok += 1;
  }
  return ok;
}

async function main() {
  console.log("TELLTAIL attestation bridge");
  console.log(`  rpc        ${RPC_URL}`);
  console.log(`  mode       ${PROGRAM_ID ? `program ${PROGRAM_ID} (PDA per subject)` : "SPL Memo"}`);
  console.log(`  batch      ${BATCH}`);
  console.log(`  poll       ${POLL_MS / 1000}s`);
  if (DRY) console.log("  DRY RUN — nothing is claimed, signed or submitted");

  const keypair = loadKeypair();
  const connection = new Connection(RPC_URL, "confirmed");
  const signer = keypair.publicKey.toBase58();
  console.log(`  signer     ${signer}`);

  if (AUTHORITY) {
    if (AUTHORITY === signer) {
      console.log(`  authority  ${AUTHORITY}  (same wallet signs and publishes)`);
    } else {
      console.log(`  authority  ${AUTHORITY}  (recorded in the payload)`);
      console.log("             the authority wallet is NOT the signer. To make it");
      console.log("             the on-chain fee payer, export its keypair to");
      console.log(`             ${process.env.SOLANA_KEYPAIR_PATH || "./secrets/devnet.json"}`);
      console.log("             (a file — never paste a secret key into a terminal).");
    }
  }

  const balance = await connection.getBalance(keypair.publicKey);
  console.log(`  balance    ${balance / LAMPORTS_PER_SOL} SOL`);
  if (balance === 0 && !DRY) {
    const kpPath = process.env.SOLANA_KEYPAIR_PATH || "./secrets/devnet.json";
    console.error("");
    console.error("  The signer holds 0 SOL, so it cannot pay a transaction fee.");
    console.error("");
    console.error("  Devnet faucets are exhausted quickly and are NOT a reliable");
    console.error("  path — both the public faucet and the Helius per-project");
    console.error("  allowance rate-limit hard. Fund from a wallet you control:");
    console.error("");
    if (AUTHORITY && AUTHORITY !== signer) {
      console.error(`    A. Make ${AUTHORITY}`);
      console.error(`       the actual signer — export ITS keypair to`);
      console.error(`         ${kpPath}`);
      console.error("       as a JSON byte array. Then the wallet named in every");
      console.error("       payload is also the wallet that pays, which is the");
      console.error("       strongest form of the claim. Move the file; never");
      console.error("       paste a secret key into a terminal or a chat.");
      console.error("");
      console.error(`    B. Or keep this signer and send it devnet SOL from`);
      console.error(`       ${AUTHORITY} — 0.5 SOL covers thousands of memos:`);
      console.error(`         recipient  ${signer}`);
      console.error("       The authority stays recorded in every payload.");
    } else {
      console.error(`    Send devnet SOL to  ${signer}`);
    }
    console.error("");
    console.error("  `npm run bridge:dry` shows exactly what would be signed.");
    process.exit(1);
  }

  const conn = await connectSnowflake();
  console.log("  snowflake  connected");

  const summary = await sql(conn, "SELECT status, COUNT(*) AS N FROM ORACLE.PUBLISH_QUEUE GROUP BY status");
  console.log(`  queue      ${summary.map((r) => `${r.STATUS}=${r.N}`).join("  ") || "empty"}`);
  console.log("");

  let stopping = false;
  process.on("SIGINT", () => {
    if (stopping) process.exit(130);
    stopping = true;
    console.log("\nstopping after this batch (ctrl-c again to force)…");
  });

  let totalPublished = 0;
  for (;;) {
    try {
      totalPublished += await drain(conn, connection, keypair);
    } catch (err) {
      console.error(`poll failed: ${err.message}`);
    }
    if (ONCE || stopping) break;
    await new Promise((r) => setTimeout(r, POLL_MS));
  }

  console.log("");
  console.log(`published ${totalPublished} attestation(s)`);
  const final = await sql(
    conn,
    `SELECT status, COUNT(*) AS N FROM ORACLE.PUBLISH_QUEUE GROUP BY status`
  );
  console.log(`queue      ${final.map((r) => `${r.STATUS}=${r.N}`).join("  ") || "empty"}`);
  conn.destroy(() => process.exit(0));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
