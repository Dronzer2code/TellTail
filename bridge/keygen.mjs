/**
 * Generate a devnet keypair for the attestation bridge.
 *
 *   npm run keygen
 *
 * Writes SOLANA_KEYPAIR_PATH (default ./secrets/devnet.json) in the same
 * 64-byte array format the Solana CLI uses, so `solana` can read it too.
 *
 * This key exists ONLY on the host running the bridge. It is never inserted
 * into Snowflake, never referenced from SQL, and ./secrets/ is gitignored. The
 * warehouse stages claims; it cannot sign one.
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Keypair, Connection, LAMPORTS_PER_SOL, PublicKey } from "@solana/web3.js";
import dotenv from "dotenv";

dotenv.config();

const KEYPAIR_PATH = resolve(process.env.SOLANA_KEYPAIR_PATH || "./secrets/devnet.json");
const RPC_URL = process.env.SOLANA_RPC_URL || "https://api.devnet.solana.com";

if (existsSync(KEYPAIR_PATH) && !process.argv.includes("--force")) {
  console.error(`refusing to overwrite ${KEYPAIR_PATH}`);
  console.error("pass --force if you really mean to replace it");
  process.exit(1);
}

mkdirSync(dirname(KEYPAIR_PATH), { recursive: true });
const kp = Keypair.generate();
writeFileSync(KEYPAIR_PATH, JSON.stringify(Array.from(kp.secretKey)), { mode: 0o600 });

console.log(`keypair written to ${KEYPAIR_PATH}  (mode 0600)`);
console.log(`public key          ${kp.publicKey.toBase58()}`);
console.log("");

const conn = new Connection(RPC_URL, "confirmed");
try {
  const before = await conn.getBalance(kp.publicKey);
  console.log(`balance             ${before / LAMPORTS_PER_SOL} SOL`);
  if (before === 0) {
    console.log("requesting a devnet airdrop of 1 SOL…");
    const sig = await conn.requestAirdrop(kp.publicKey, LAMPORTS_PER_SOL);
    const bh = await conn.getLatestBlockhash();
    await conn.confirmTransaction({ signature: sig, ...bh }, "confirmed");
    const after = await conn.getBalance(kp.publicKey);
    console.log(`airdrop confirmed   ${after / LAMPORTS_PER_SOL} SOL`);
    console.log(`  https://explorer.solana.com/tx/${sig}?cluster=devnet`);
  }
} catch (err) {
  console.warn(`airdrop failed: ${err.message}`);
  console.warn("the devnet faucet rate-limits aggressively. Alternatives:");
  console.warn(`  solana airdrop 1 ${kp.publicKey.toBase58()} --url devnet`);
  console.warn("  https://faucet.solana.com");
}

console.log("");
console.log("next:  npm run bridge:dry     # claim nothing, sign nothing, show the payloads");
console.log("       npm run bridge         # drain the queue for real");
