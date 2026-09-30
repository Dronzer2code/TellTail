/**
 * Import the authority wallet's keypair into secrets/devnet.json.
 *
 *   node bridge/import-key.mjs <file-containing-the-key>
 *
 * WHY THIS EXISTS RATHER THAN "just paste your key".
 *
 * Phantom and Solflare export a private key as a base58 STRING. @solana/web3.js
 * wants a 64-byte JSON array. Converting by hand means the key passes through a
 * shell command, which puts it in your shell history, your terminal scrollback,
 * and the process table where any other process can read it. This reads the key
 * from a FILE you delete afterwards, so it never becomes an argument.
 *
 * It accepts either format:
 *   - base58 string      (Phantom / Solflare "Show private key")
 *   - JSON byte array    (solana-keygen / `solana config get` id.json)
 *
 * And it REFUSES to write a key whose public key is not SOLANA_AUTHORITY —
 * silently signing with the wrong wallet is exactly the failure this whole
 * separation is meant to make impossible.
 */
import { readFileSync, writeFileSync, mkdirSync, chmodSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Keypair } from "@solana/web3.js";
import dotenv from "dotenv";

dotenv.config();

const B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/** Minimal base58 decode — avoids adding a dependency for forty lines. */
function b58decode(str) {
  const bytes = [0];
  for (const ch of str.trim()) {
    const v = B58.indexOf(ch);
    if (v < 0) throw new Error(`not base58: ${JSON.stringify(ch)}`);
    let carry = v;
    for (let i = 0; i < bytes.length; i++) {
      carry += bytes[i] * 58;
      bytes[i] = carry & 0xff;
      carry >>= 8;
    }
    while (carry) {
      bytes.push(carry & 0xff);
      carry >>= 8;
    }
  }
  for (const ch of str.trim()) {
    if (ch !== B58[0]) break;
    bytes.push(0);
  }
  return Uint8Array.from(bytes.reverse());
}

const src = process.argv[2];
if (!src) {
  console.error("usage: node bridge/import-key.mjs <file-containing-the-key>");
  console.error("");
  console.error("  1. In Phantom/Solflare: Show Private Key, copy it.");
  console.error("  2. Paste it into a NEW empty text file, e.g. key.txt");
  console.error("  3. node bridge/import-key.mjs key.txt");
  console.error("  4. DELETE key.txt");
  process.exit(1);
}

const raw = readFileSync(resolve(src), "utf8").trim();

let secret;
if (raw.startsWith("[")) {
  secret = Uint8Array.from(JSON.parse(raw));
} else {
  secret = b58decode(raw);
}

if (secret.length !== 64) {
  console.error(`expected a 64-byte secret key, got ${secret.length} bytes.`);
  console.error("A 32-byte value is a SEED, not a keypair — export the private");
  console.error("key rather than the recovery phrase.");
  process.exit(1);
}

const kp = Keypair.fromSecretKey(secret);
const pub = kp.publicKey.toBase58();
const want = (process.env.SOLANA_AUTHORITY || "").trim();

console.log(`derived public key : ${pub}`);
if (want) {
  console.log(`SOLANA_AUTHORITY   : ${want}`);
  if (pub !== want) {
    console.error("");
    console.error("REFUSING TO WRITE — this key is not the authority wallet.");
    console.error("Every payload already claims to be published by the authority");
    console.error("above; signing with a different wallet would make the claim on");
    console.error("chain disagree with the claim in the data.");
    process.exit(1);
  }
  console.log("match              : yes");
}

const dest = resolve(process.env.SOLANA_KEYPAIR_PATH || "./secrets/devnet.json");
mkdirSync(dirname(dest), { recursive: true });
writeFileSync(dest, JSON.stringify(Array.from(secret)), { mode: 0o600 });
try {
  chmodSync(dest, 0o600);
} catch {
  /* Windows ignores POSIX modes; secrets/ is gitignored either way */
}

console.log(`written            : ${dest}`);
console.log("");
console.log(`NOW DELETE ${src} — it still contains the key in plain text.`);
console.log("Then run:  npm run bridge:once");
