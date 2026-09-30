-- ===========================================================================
-- 03_raw.sql   ·   untouched landing
--
-- Two tables, deliberately:
--
--   COLLAR_TELEMETRY_BULK  the full historical corpus. Loaded once from Parquet
--                          shards, never written again. Trains the classifier
--                          and backs the deep-history tabs.
--
--   COLLAR_TELEMETRY       the LIVE landing table. The replayer feeds it in
--                          8-second micro-batches at wall-clock speed and the
--                          Dynamic Table DAG watches it. This is what makes the
--                          demo feel alive instead of static.
--
-- The DAG never watches BULK. Ten million rows re-scanned every minute is how
-- you burn a trial account before Sunday.
-- ===========================================================================

USE DATABASE ${SNOWFLAKE_DATABASE};
USE SCHEMA RAW;

-- ---------------------------------------------------------------------------
-- Historical corpus. COPY INTO target for scripts/load_raw.py.
-- Column names here are canonical TELLTAIL names; the loader renames from the
-- real CSV headers using ref/column_map.json, so the profile stays the single
-- source of truth for what the file actually calls things.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS RAW.COLLAR_TELEMETRY_BULK (
    dog_id          NUMBER,
    test_num        NUMBER,
    t_sec           FLOAT,                 -- seconds from session start

    neck_ax FLOAT, neck_ay FLOAT, neck_az FLOAT,
    neck_gx FLOAT, neck_gy FLOAT, neck_gz FLOAT,
    back_ax FLOAT, back_ay FLOAT, back_az FLOAT,
    back_gx FLOAT, back_gy FLOAT, back_gz FLOAT,

    label_primary   STRING,
    label_secondary STRING,
    label_tertiary  STRING,
    point_event     STRING,
    task            STRING,

    _shard          STRING,
    loaded_at       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (dog_id, test_num)
COMMENT = 'DogMoveData.csv, 10.6M rows @ 100 Hz, dual IMU. Loaded once from Parquet.';

-- ---------------------------------------------------------------------------
-- Live landing. sample_ts is t_sec projected onto a wall-clock replay epoch, so
-- TIME_SLICE() has something real to bucket and the dashboard has a "now".
-- raw_payload keeps the full source row untouched, including any column the
-- canonical schema has no slot for.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS RAW.COLLAR_TELEMETRY (
    dog_id          NUMBER,
    test_num        NUMBER,
    t_sec           FLOAT,
    sample_ts       TIMESTAMP_NTZ,

    neck_ax FLOAT, neck_ay FLOAT, neck_az FLOAT,
    neck_gx FLOAT, neck_gy FLOAT, neck_gz FLOAT,
    back_ax FLOAT, back_ay FLOAT, back_az FLOAT,
    back_gx FLOAT, back_gy FLOAT, back_gz FLOAT,

    label_primary   STRING,
    label_secondary STRING,
    label_tertiary  STRING,
    point_event     STRING,
    task            STRING,

    raw_payload     VARIANT,
    _batch_id       STRING,
    is_replay       BOOLEAN       DEFAULT TRUE,
    is_synthetic    BOOLEAN       DEFAULT FALSE,   -- demo_spike.py writes TRUE
    landed_at       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (dog_id, sample_ts)
COMMENT = 'Live micro-batch landing. The Dynamic Table DAG watches this table.';

-- ---------------------------------------------------------------------------
-- Ingest audit. The Live Collar tab reads this to show "rows landed in the last
-- minute", so liveness is a number and not a vibe.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS RAW.INGEST_LOG (
    batch_id     STRING,
    run_id       STRING,
    n_rows       NUMBER,
    window_start TIMESTAMP_NTZ,
    window_end   TIMESTAMP_NTZ,
    speed        NUMBER,
    dogs         ARRAY,
    is_synthetic BOOLEAN DEFAULT FALSE,
    landed_at    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE VIEW RAW.V_INGEST_RATE AS
SELECT
    DATE_TRUNC('minute', landed_at)                       AS minute,
    SUM(n_rows)                                           AS rows_landed,
    COUNT(*)                                              AS batches,
    MAX(window_end)                                       AS latest_sample_ts,
    ARRAY_UNIQUE_AGG(run_id)                              AS runs
FROM RAW.INGEST_LOG
GROUP BY 1;

-- ---------------------------------------------------------------------------
-- Load verification. scripts/load_raw.py asserts against this after COPY INTO;
-- a mismatch against ref/column_map.json aborts rather than proceeding into a
-- pipeline built on a partial load.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW RAW.V_BULK_PROFILE AS
SELECT
    COUNT(*)                                              AS row_count,
    COUNT(DISTINCT dog_id)                                AS distinct_dogs,
    COUNT(DISTINCT test_num)                              AS distinct_tests,
    MIN(t_sec)                                            AS t_min,
    MAX(t_sec)                                            AS t_max,
    COUNT(DISTINCT label_primary)                         AS distinct_labels,
    SUM(IFF(label_primary IS NULL, 1, 0))                 AS null_labels,
    SUM(IFF(neck_ax IS NULL OR back_ax IS NULL, 1, 0))    AS null_sensor_rows
FROM RAW.COLLAR_TELEMETRY_BULK;

CREATE OR REPLACE VIEW RAW.V_LABEL_COVERAGE AS
SELECT
    b.label_primary                                       AS raw_label,
    COUNT(*)                                              AS n_rows,
    ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 3) AS pct,
    m.state                                               AS mapped_state,
    IFF(m.raw_label IS NULL, TRUE, FALSE)                 AS is_unmapped
FROM RAW.COLLAR_TELEMETRY_BULK b
LEFT JOIN REF.LABEL_MAP m
       ON m.raw_label = b.label_primary
      AND m.source_column = 'label_primary'
GROUP BY b.label_primary, m.state, m.raw_label;
