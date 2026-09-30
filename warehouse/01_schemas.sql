-- ===========================================================================
-- 01_schemas.sql
--
-- One database, seven schemas, data flowing strictly left to right.
-- Every schema has a single job and nothing reaches backwards.
--
--   REF     hand-loaded seed + external truth      (no upstream)
--   RAW     untouched landing                      (<- replayer, bulk loader)
--   STAGING epoch features                         (<- RAW)
--   MARTS   states, syndromes, baselines           (<- STAGING, ML)
--   ML      model artefacts and outputs            (<- STAGING, MARTS)
--   AI      cached Cortex output                   (<- MARTS)
--   ORACLE  attestation queue and audit            (<- MARTS, AI)
--
-- If you ever find yourself writing a MARTS -> STAGING reference, the layering
-- is wrong and the DAG will deadlock on refresh order.
-- ===========================================================================

USE DATABASE ${SNOWFLAKE_DATABASE};

CREATE SCHEMA IF NOT EXISTS REF     COMMENT = 'Seed data and external truth. Hand-loaded or host-synced.';
CREATE SCHEMA IF NOT EXISTS RAW     COMMENT = 'Untouched landing. 100 Hz telemetry, typed + VARIANT payload.';
CREATE SCHEMA IF NOT EXISTS STAGING COMMENT = 'Epoch feature layer. 10.6M rows -> ~106K one-second epochs.';
CREATE SCHEMA IF NOT EXISTS MARTS   COMMENT = 'States, syndrome matches, baselines, deviations.';
CREATE SCHEMA IF NOT EXISTS ML      COMMENT = 'Model artefacts, evaluation, forecasts, anomalies, drivers.';
CREATE SCHEMA IF NOT EXISTS AI      COMMENT = 'Cached Cortex output. Never called from a render path.';
CREATE SCHEMA IF NOT EXISTS ORACLE  COMMENT = 'Attestation queue and audit trail. The keypair lives elsewhere.';

-- Internal stage for the Parquet shards the bulk loader produces, and for the
-- Streamlit app + environment.yml.
CREATE STAGE IF NOT EXISTS RAW.DOG_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Parquet shards of DogMoveData.csv, PUT by scripts/load_raw.py';

CREATE STAGE IF NOT EXISTS MARTS.APP_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'streamlit_app.py + environment.yml. environment.yml MUST ship next to the app.';

-- --------------------------------------------------------------------------
-- Build audit. Every run_sql.py invocation writes here, so the Pipeline tab
-- can show what was built when, and so a half-finished run is visible.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS REF.BUILD_LOG (
    build_id      STRING,
    script_name   STRING,
    started_at    TIMESTAMP_NTZ,
    finished_at   TIMESTAMP_NTZ,
    statements_ok NUMBER,
    statements_ko NUMBER,
    status        STRING,
    detail        STRING
);
