# TELLTAIL

**A dog cannot tell you where it hurts. Ten million rows of collar data can.**

Every consumer dog tracker computes activity minutes and step counts, because
that is what you get from `AVG()` and `SUM()`. But a vet does not diagnose from
an average. A vet diagnoses from a **sequence**: this, then that, then this
again, in that order, that many times.

TELLTAIL streams 10.6 million rows of real dog collar telemetry into Snowflake,
classifies every second into a behavioural state, and then hunts for **clinical
syndromes expressed as regular expressions over rows** using `MATCH_RECOGNIZE`.

Not *"scratching is high today."* Instead:

```text
head shake, scratch cluster, head shake, scratch cluster, emerging from rest
```

which is otitis and nothing else.

> **The thesis.** A threshold cannot detect a syndrome. Aggregation tells you a
> number changed. Only row pattern recognition tells you that events happened in
> a diagnostic order. Snowflake is the only warehouse where a differential
> diagnosis is a SQL clause.

---

## Five numbers

| Number | What it is |
|---|---|
| **10,611,068** | Raw telemetry rows, from a published research dataset |
| **45 / 27** | Dogs and distinct breeds, each dog its own control |
| **100 Hz** | Sample rate, dual sensor: neck collar **and** back harness |
| **6** | Clinical syndromes expressed as row patterns |
| **1 minute** | Declared target lag on the transform DAG. No cron anywhere |

---

## Quickstart

```bash
git clone https://github.com/SoumyaEXE/telltail && cd telltail
cp .env.example .env                     # fill in Snowflake creds
pip install -r requirements.txt && npm install

python scripts/smoke_test.py             # 90s: does MATCH_RECOGNIZE + Cortex work here?
python scripts/fetch_dataset.py          # 454 MB corpus, resumable
python scripts/profile_dataset.py        # GATE A. writes ref/column_map.json
python scripts/load_raw.py               # 10.6M rows -> Snowflake
python scripts/run_sql.py --all          # schemas, DAG, models, patterns, tasks
python ingest/replay.py --speed 60       # the feed goes live
python scripts/deploy_streamlit.py       # dashboard inside the warehouse
python scripts/austin_sync.py            # shelter ground truth
npm run keygen && npm run bridge         # queue -> signed devnet writes
```

**Run `smoke_test.py` before anything else.** It checks `MATCH_RECOGNIZE`,
Cortex, all three ML function classes, `ASOF JOIN`, dynamic-table creation and
your remaining credit in about ninety seconds. If the row-pattern engine or
Cortex is unavailable on your account or region, you need to know at hour zero,
not hour thirty.

Two things it will tell you that are easy to get wrong:

- **`SNOWFLAKE_ACCOUNT` is the account identifier, not the URL.** For
  `https://abcdefg-xy12345.snowflakecomputing.com` it is `abcdefg-xy12345`.
- **`CORTEX_MODEL` must be a model your region actually serves.** Model ids
  differ by region and change over time; the smoke test names the one it used,
  and it is a single `.env` line rather than a grep across ten SQL files.

`fetch_dataset.py` needs no Kaggle CLI and no `kaggle.json` — the mirror is
publicly downloadable, and the download resumes if it drops.

---

## Verify it without an account

The riskiest part of this build is the syndrome layer, and it is normally the
last thing you can test. Here it is the first:

```bash
python tests/run_all.py
```

No Snowflake account, no credits, no network. It:

1. **compiles the real `PATTERN` / `DEFINE` clauses out of `warehouse/*.sql`**
   into regular expressions (which is what they are), and exercises them against
   fixtures — each syndrome must fire on its own signature, stay silent on four
   near-misses, and produce nothing at all on 20,000 epochs of random ordinary
   behaviour;
2. **synthesises 100 Hz IMU signal** for every behaviour, runs the real feature
   math and the real state ladder over it with thresholds parsed out of the seed
   SQL, and confirms each syndrome signature still produces its own syndrome and
   only its own;
3. checks the Streamlit app parses under **Python 3.11**, which is what
   Streamlit in Snowflake pins;
4. checks every SQL file splits into statements and resolves its template
   variables.

That suite has already caught two real bugs: a smoothing filter that deleted the
single-epoch head shake S1 is defined by, and a `PAUSE` rule that swallowed the
alert stand S5 is defined by. Both would have presented as *"the syndrome just
never fires"* with no error anywhere.

---

## Architecture

```text
Kaggle CSV --replay--> Snowflake --ML--> Snowflake --pattern--> Snowflake --attest--> Solana
10.6M rows @ 100Hz     VARIANT land      CLASSIFICATION         MATCH_RECOGNIZE       devnet
45 dogs, 27 breeds     Dynamic Tables    dog-disjoint folds     syndrome catalogue    claim not data
wall-clock replay      1 min lag DAG     epoch -> state         clinical sequences    portable record
```

**Snowflake holds the compute**: ten million sensor rows, a declarative transform
DAG, supervised classification, row pattern recognition, four ML functions and an
LLM, all without leaving SQL.

**Solana holds the thing Snowflake cannot provide: portability.** A dog arriving
at a shelter arrives with no history, because shelters, vets and adopters share
no database and never will. The health attestation is published as a signed
claim — hashed identifier only — so the record survives rehoming. *Publish the
claim, never the data.*

### Ten stages

| # | Stage | What it does |
|---|---|---|
| 1 | **REPLAY** | `ingest/replay.py` streams the corpus in timestamp order at wall-clock speed, 8-second micro-batches, so a static dataset behaves as a live feed. `--speed` compresses time for the demo |
| 2 | **LAND** | `RAW.COLLAR_TELEMETRY`, typed columns plus a `VARIANT` payload, `_batch_id`, `is_replay`, `is_synthetic` |
| 3 | **FEATURES** | Dynamic Table, `TARGET_LAG '1 minute'`. 100 Hz → 1-second epochs. Per epoch: vector magnitude stats, SMA, energy, zero-crossing rate, jerk, pitch/roll, yaw geometry — **and `CORR(vm_neck, vm_back)`** |
| 4 | **STATE** | `SNOWFLAKE.ML.CLASSIFICATION` over the labelled epochs, **split by dog, never by row**, plus a geometry/context ladder for states the label vocabulary cannot express |
| 5 | **SYNDROME** | `MATCH_RECOGNIZE` over the epoch state sequence. Six clinical patterns, `ONE ROW PER MATCH`, plus `ALL ROWS PER MATCH` + `CLASSIFIER()` for per-epoch symbols |
| 6 | **BASELINE** | `ASOF JOIN` each epoch against the dog's own trailing baseline and its breed cohort. Every dog is its own control |
| 7 | **ML** | `FORECAST`, `ANOMALY_DETECTION` on a strict boundary split, `TOP_INSIGHTS` on the deviation metric |
| 8 | **LANGUAGE** | `AI_COMPLETE` writes a SOAP handoff note, `AI_CLASSIFY` assigns triage, `AI_AGG` writes a pack brief. All batched into tables by a task |
| 9 | **TRUTH** | Austin Animal Center intakes and outcomes via the Socrata API, so detected categories sit beside real shelter outcomes |
| 10 | **ATTEST** | `ORACLE.PUBLISH_QUEUE` staged by a task; a Node bridge signs and submits to devnet. **The keypair never touches Snowflake** |

### Lineage

```text
RAW.COLLAR_TELEMETRY  (100 Hz, live micro-batches)
  └── STAGING.EPOCH_FEATURES          [Dynamic Table, TARGET_LAG '1 minute']
        └── MARTS.EPOCH_STATES        [Dynamic Table]  classifier + state ladder
              ├── MARTS.SYNDROME_MATCHES     [TASK, 2 min]  MATCH_RECOGNIZE
              ├── MARTS.STATE_TRANSITIONS    [Dynamic Table] behavioural Markov
              └── MARTS.DOG_BASELINE         [Dynamic Table] ASOF JOIN self-compare
                    ├── ML.ACTIVITY_FORECAST / ANOMALIES  [TASK]
                    ├── ML.DRIVER_INSIGHTS                [TASK]  TOP_INSIGHTS
                    └── AI.VET_NOTES / TRIAGE / PACK_BRIEF [TASK]  Cortex
                          └── ORACLE.PUBLISH_QUEUE  [TASK] ── Node bridge ──> Solana
```

Seven schemas, data flowing strictly left to right: `REF → RAW → STAGING →
MARTS → ML → AI → ORACLE`. Nothing reaches backwards.

---

## The feature that earned its place

Almost every published approach to this dataset feeds a large feature vector
into a black-box classifier. This build adds one column:

```sql
CORR(vm_neck, vm_back) AS neck_back_corr
```

Two IMUs — one on the collar, one on the back harness. When they move **in
phase**, the whole body is translating: walk, trot, gallop. When they
**decouple**, the neck is moving and the body is not: head shake, scratch.

One SQL function, physically interpretable, and it exists only because this
dataset has two sensors and someone read the description. It is exactly the
discriminator the aural syndrome depends on.

Measured separation on synthesised signal (`tests/test_demo_signal.py`):

| family | mean `neck_back_corr` |
|---|---|
| locomotion (walk / trot / gallop) | **+0.99** |
| neck-dominant (scratch / shake / sniff) | **−0.03** |

Validate it on the real data at Gate C:

```sql
SELECT * FROM STAGING.V_CORR_BY_LABEL ORDER BY avg_corr;
```

If locomotion labels do not sit meaningfully above posture and neck-dominant
labels, the feature is not doing what you think — and you want to know then, not
in the write-up.

---

## The six syndromes

Each is a `PATTERN` over the epoch state sequence. Each ships with the reason a
sequence beats a threshold, because that reason *is* the argument.

| | Syndrome | Pattern | Why a sequence, not a threshold |
|---|---|---|---|
| **S1** | Otitis / ear irritation | `onset shake itch{3,} shake2 itch2{2,}` | Daily scratch count is normal in a flea-free dog. The signal is the **alternation** of shake and scratch, emerging from rest. A total has no concept of order |
| **S2** | Intermittent lameness | `stride{3,} halt stride2{1,3} halt2 stride3{1,3} halt3` | Step count unchanged, distance unchanged. Stride **interruption frequency** is rising. *"He seems fine, he just stops a lot now"* |
| **S3** | Exercise intolerance | `burst+ recover{5,} burst2{1,2} recover2{8,}` | Total activity minutes are **identical**, redistributed into shorter bursts with longer recoveries. A daily total is definitionally blind to redistribution |
| **S4** | Reluctance to rise | `settled{10,} lever rise settled2{10,}` | Rest totals unchanged; a "hours resting" threshold fires on a healthy sleeping dog. The **transition** is the finding, and a transition only exists between two states in order |
| **S5** | Separation distress | `alert tread{4,} alert2 tread2{4,}` | Pacing exists in happy dogs. The **cycle** — pace, check the door, pace, check again — is what distinguishes distress |
| **S6** | GI discomfort | `probe{5,} turn{2,} probe2{5,}` | Sniffing is the commonest outdoor behaviour and circling precedes every normal elimination. Only **repetition without resolution** is abnormal |

The patterns live as text in `REF.SYNDROME_CATALOGUE`, so one definition drives
three consumers: the hand-written views, the sensitivity sweep built by
`EXECUTE IMMEDIATE`, and the code block printed beside the chart. A test asserts
they never drift apart — the SQL on screen is the SQL that ran.

### Sensitivity, not a magic number

A single hand-tuned quantifier is a magic number and judges can smell it. Every
pattern runs at **loose / tuned / strict**, all eighteen variants generated from
one metadata table:

```sql
SELECT * FROM MARTS.V_SENSITIVITY_CURVE ORDER BY syndrome_code, variant;
```

---

## The split nobody else did

The literature on this dataset reports single-subject classifiers falling from
~91% to **~70–74%** when generalising to dogs they were not trained on.

At 100 Hz, samples 40 ms apart are near-identical. A random **row** split puts
adjacent samples from the same dog on both sides of the fold, the model
memorises the individual, and the reported accuracy is a fiction.

TELLTAIL holds out **whole dogs** — breed-stratified, deterministic — and prints
the lower number large on the Drivers tab:

```sql
SELECT holdout_accuracy, holdout_dogs, macro_f1, protocol FROM ML.MODEL_SUMMARY;
```

---

## The dashboard

Nine tabs, in narrative order. A judge who clicks left to right gets the whole
argument without reading a word of the post.

**Pack** (who) → **Live Collar** (is it real) → **Ethogram** (what normal looks
like) → **Syndromes** (the finding) → **Baselines** (why it is abnormal for
*this* dog) → **Vet Note** (what to do) → **Drivers** (what explains it) →
**Shelter Reality** (why it matters) → **Pipeline** (how it was built).

The design budget goes to one place: on the Syndromes tab, selecting a match
lights up **the exact epochs that matched, coloured by the pattern variable each
one played**, with the pattern string above it and the SQL beside it. That comes
straight from `ALL ROWS PER MATCH` + `CLASSIFIER()`.

---

## Repository

```text
telltail/
  warehouse/          00 → 11, numbered, idempotent, runnable start to finish
    streamlit_app.py  nine tabs
    environment.yml   MUST ship to the stage beside the app, or plotly fails in SiS only
  scripts/
    profile_dataset.py  GATE A — reads the real CSV, writes ref/column_map.json
    smoke_test.py       90 seconds, before anything else
    load_raw.py         CSV -> Parquet shards -> stage -> COPY INTO
    run_sql.py          --all, with role switching and cold-start bootstrap hooks
    austin_sync.py      Socrata -> REF.AAC_*
    demo_spike.py       labelled synthetic deterioration, --clean to undo
    deploy_streamlit.py app + environment.yml -> stage
  ingest/replay.py    bulk -> live micro-batches at wall-clock speed
  bridge/             Node: queue poll -> sign -> devnet -> audit back
  tests/              the whole offline suite
  ref/                column_map contract
```

### Everything numeric is a row, not a literal

`REF.PARAMS` holds every threshold in the build, with a unit and a description,
and the Pipeline tab renders the whole table. There are no magic numbers below
`02_ref_seed.sql`.

---

## Known gotchas, pre-loaded

Researched, not guessed. Handled in the code; listed here so you do not
rediscover them.

| Symptom | Root cause | Handling |
|---|---|---|
| Dynamic table with `MATCH_RECOGNIZE` fails on `REFRESH_MODE = INCREMENTAL` | Unsupported construct; `AUTO` silently resolves to `FULL` | The syndrome layer is driven by a **task**, not a dynamic table. Where `FULL` is accepted it is declared explicitly |
| `MATCH_RECOGNIZE` never returns | Pointed at the 100 Hz raw table, ~10.6M rows per partition | Epoch layer only. 10.6M → ~106K. A hard performance gate |
| Compilation error using `MATCH_RECOGNIZE` in a recursive CTE | Explicitly unsupported | Materialise first, then pattern match |
| Cortex anomaly detection rejects the model | Train and detect windows overlap | One `ML.SPLIT_BOUNDARY` table; train is `ts < T`, detect is `ts >= T`. They cannot disagree |
| Classification accuracy suspiciously high | Random row split leaked adjacent samples | Whole dogs held out. Expect a materially lower, honest number |
| Every line chart is an identical straight diagonal | `to_pandas()` returns object-dtype `Decimal`; Plotly treats them as categories so y becomes the row index | One `rows()` helper converts element-wise with `float()`; charts get plain Python lists |
| `ModuleNotFoundError: plotly` in SiS only | Packages must be declared per app | `environment.yml` ships to the stage beside the app; the deploy script refuses without it |
| `module 'streamlit' has no attribute 'column_config'` | SiS pins an older Streamlit | `hasattr` guard plus a styled HTML table fallback |
| App fails to parse in SiS but runs locally | SiS pins Python **3.11**; PEP 701 f-strings are 3.12+ | `tests/sis_compat.py` enforces 3.11 grammar in CI |
| Timestamps drift, epoch boundaries land wrong | Account timezone is not UTC | `ALTER ACCOUNT SET TIMEZONE = 'Etc/UTC'` in `00_account_setup.sql`, and the smoke test warns |
| Cortex stops working mid-Sunday | Trial daily AI Function credit cap exhausted | Batched into tables by a task, deduped on `(dog_id, syndrome_code, onset_ts)`, hard-capped per batch, never called from a render path |
| External access integration rejected | Not supported on trial accounts | Outbound queue-poll bridge design |
| SiS cannot fetch shelter data | No public internet egress | Host-side sync writes into `REF`; the app only reads a table |
| `COPY INTO` slow or erroring on the big CSV | Single ~2 GB uncompressed file | Sharded to Parquet, `PUT` with `AUTO_COMPRESS`, verified against the profile |

---

## Honesty

Every simplification is listed in **[HONESTY.md](HONESTY.md)** with the query
that lets you check it. Short version: the telemetry is replayed (`is_replay`),
some states are heuristic (`state_source`), demo deterioration is injected
(`is_synthetic`), attestations are devnet, shelter data is Austin only, and the
accuracy figure is the dog-disjoint one.

**TELLTAIL is not a diagnostic device and nothing it produces substitutes for a
veterinarian.**

Data: Vehkaoja et al., *Description of Movement Sensor Dataset for Dog Behavior
Classification*, **Data in Brief**, 2022, University of Helsinki. Shelter data:
City of Austin open data portal.

---

## The bigger thing

Strip away the dogs and the pattern is this: **row pattern recognition turns a
warehouse into a diagnostic engine for any sensored system where the sequence
carries the meaning.** Fleet telemetry, industrial equipment, patient
monitoring. The plumbing is exactly what is in this repo.

> The dog was showing the pattern for eleven days before anyone noticed.
> The warehouse noticed on day two.
