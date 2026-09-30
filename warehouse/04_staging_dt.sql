-- ===========================================================================
-- 04_staging_dt.sql   ·   the epoch feature layer
--
-- 10.6 million 100 Hz samples become roughly 106 thousand one-second epochs.
-- This is a hard performance gate, not an optimisation: MATCH_RECOGNIZE over
-- the raw table scans ~10.6M rows per partition and never returns.
--
-- THE feature is neck_back_corr. Two IMUs — collar and back harness — moving in
-- phase means the whole body is translating (walk, trot, gallop). Two IMUs
-- decoupled means the neck is moving and the body is not, which is exactly the
-- shake/scratch family the aural syndrome depends on. One CORR() across two
-- sensor streams gives a physically interpretable discriminator, and it exists
-- only because this dataset has two sensors.
--
-- REFRESH_MODE is FULL and declared. The sample-level window functions (epoch
-- mean for zero-crossing rate, LAG for jerk) are not incrementally refreshable.
-- Setting INCREMENTAL explicitly fails compilation with an unsupported
-- construct; AUTO silently resolves to FULL. Declaring it makes the cost
-- visible instead of surprising.
-- ===========================================================================

USE DATABASE ${SNOWFLAKE_DATABASE};
USE SCHEMA STAGING;

-- ---------------------------------------------------------------------------
-- Sample-level derivation, shared by the live and bulk paths.
--
-- Every quantity here is per-sample. Anything that needs the whole epoch is a
-- window function over the epoch partition, computed once at 100 Hz so the
-- aggregate below is a plain GROUP BY.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW STAGING.V_SAMPLE_DERIVED AS
SELECT
    dog_id,
    test_num,
    sample_ts,
    TIME_SLICE(sample_ts, 1, 'SECOND')                             AS epoch_ts,
    label_primary,
    label_secondary,
    is_synthetic,

    -- vector magnitudes: the rotation-invariant summary of a 3-axis reading
    SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az)      AS vm_neck,
    SQRT(back_ax*back_ax + back_ay*back_ay + back_az*back_az)      AS vm_back,
    SQRT(neck_gx*neck_gx + neck_gy*neck_gy + neck_gz*neck_gz)      AS gm_neck,
    SQRT(back_gx*back_gx + back_gy*back_gy + back_gz*back_gz)      AS gm_back,

    -- signal magnitude area, the standard actigraphy intensity proxy
    ABS(neck_ax) + ABS(neck_ay) + ABS(neck_az)                     AS sma_neck,
    ABS(back_ax) + ABS(back_ay) + ABS(back_az)                     AS sma_back,

    -- orientation from the gravity component
    ATAN2(neck_ax, SQRT(neck_ay*neck_ay + neck_az*neck_az))        AS pitch_neck,
    ATAN2(neck_ay, SQRT(neck_ax*neck_ax + neck_az*neck_az))        AS roll_neck,
    ATAN2(back_ax, SQRT(back_ay*back_ay + back_az*back_az))        AS pitch_back,

    -- yaw rate. Gyro z on the back harness is rotation about the dorsoventral
    -- axis: turning. CIRCLE and PACE are both defined from its behaviour.
    back_gz                                                        AS yaw_back,

    -- jerk proxy: first difference of neck magnitude within the session.
    -- Ordered by sample_ts inside (dog, test) so a session boundary cannot
    -- manufacture a spike.
    SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az)
      - LAG(SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az))
          OVER (PARTITION BY dog_id, test_num ORDER BY sample_ts)  AS jerk_neck,

    -- epoch-mean-removed magnitude, so the zero-crossing count below is a real
    -- ZCR on the AC component rather than a count of crossings of zero (which a
    -- +1g gravity offset would make identically zero).
    SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az)
      - AVG(SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az))
          OVER (PARTITION BY dog_id, test_num, TIME_SLICE(sample_ts, 1, 'SECOND'))
                                                                   AS vm_neck_ac
FROM RAW.COLLAR_TELEMETRY;

-- ---------------------------------------------------------------------------
-- Epoch aggregation, live path. TARGET_LAG '1 minute'. No cron anywhere.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE DYNAMIC TABLE STAGING.EPOCH_FEATURES
    TARGET_LAG   = '1 minute'
    WAREHOUSE    = ${SNOWFLAKE_WAREHOUSE}
    REFRESH_MODE = FULL
    INITIALIZE   = ON_CREATE
    COMMENT      = 'One row per dog-second. ~100 samples in, one feature vector out.'
AS
WITH s AS (
    SELECT
        *,
        -- sign of the AC component, and the previous sign, for the ZCR count
        SIGN(vm_neck_ac)                                            AS sgn,
        LAG(SIGN(vm_neck_ac)) OVER (
            PARTITION BY dog_id, test_num, epoch_ts ORDER BY sample_ts) AS sgn_prev
    FROM STAGING.V_SAMPLE_DERIVED
)
SELECT
    dog_id,
    test_num,
    epoch_ts,
    COUNT(*)                                                        AS n_samples,

    -- ---- neck channel -----------------------------------------------------
    AVG(vm_neck)                                                    AS vm_neck_mean,
    STDDEV(vm_neck)                                                 AS vm_neck_std,
    MAX(vm_neck) - MIN(vm_neck)                                     AS vm_neck_range,
    SUM(vm_neck * vm_neck)                                          AS energy_neck,
    AVG(sma_neck)                                                   AS sma_neck,
    AVG(ABS(jerk_neck))                                             AS jerk_neck_mean,
    STDDEV(jerk_neck)                                               AS jerk_neck_std,

    -- zero-crossing rate of the mean-removed neck magnitude: crossings per
    -- second, i.e. roughly twice the dominant oscillation frequency. Scratching
    -- is a fast repetitive motion and this is what separates it from a slow
    -- postural shift with the same variance.
    SUM(IFF(sgn_prev IS NOT NULL AND sgn <> sgn_prev AND sgn <> 0, 1, 0))
        / NULLIF(COUNT(*) - 1, 0)                                   AS zcr_neck,

    -- ---- back channel -----------------------------------------------------
    AVG(vm_back)                                                    AS vm_back_mean,
    STDDEV(vm_back)                                                 AS vm_back_std,
    AVG(sma_back)                                                   AS sma_back,
    SUM(vm_back * vm_back)                                          AS energy_back,

    -- ---- gyro -------------------------------------------------------------
    AVG(gm_neck)                                                    AS gyro_neck_mean,
    AVG(gm_back)                                                    AS gyro_back_mean,

    -- Yaw geometry. |mean(yaw)| / mean(|yaw|) is a directional consistency
    -- ratio in [0,1]:
    --   ~1  every sample turns the same way          -> circling
    --   ~0  turning cancels out over the second      -> reversing, i.e. pacing
    -- The denominator distinguishes both from simply standing still.
    AVG(yaw_back)                                                   AS yaw_mean,
    AVG(ABS(yaw_back))                                              AS yaw_abs_mean,
    ABS(AVG(yaw_back)) / NULLIF(AVG(ABS(yaw_back)), 0)              AS yaw_consistency,

    -- ---- orientation ------------------------------------------------------
    AVG(pitch_neck)                                                 AS pitch_neck_mean,
    STDDEV(pitch_neck)                                              AS pitch_var,
    AVG(roll_neck)                                                  AS roll_neck_mean,
    STDDEV(roll_neck)                                               AS roll_var,
    AVG(pitch_back)                                                 AS pitch_back_mean,

    -- ---- THE FEATURE ------------------------------------------------------
    -- Pearson correlation between the two sensors' magnitudes across the ~100
    -- samples of this epoch.
    --   high  both sensors move together   -> whole-body locomotion
    --   ~0    neck moves, body does not    -> head shake, scratch
    -- Physically interpretable, one SQL function, and only possible because
    -- this dataset has two sensors.
    CORR(vm_neck, vm_back)                                          AS neck_back_corr,

    -- neck dominance: how much more the neck is moving than the back. Backs up
    -- the correlation with a magnitude term, so a quiet epoch with incidental
    -- low correlation is not mistaken for scratching.
    STDDEV(vm_neck) / NULLIF(STDDEV(vm_back), 0)                    AS neck_dominance,

    -- ---- composite --------------------------------------------------------
    -- Activity index: the single scalar the baseline, forecast and anomaly
    -- layers all run on. Deliberately simple and documented rather than tuned.
    (AVG(sma_neck) + AVG(sma_back)) / 2.0                           AS activity_index,

    -- ---- labels and provenance -------------------------------------------
    MODE(label_primary)                                             AS label_primary,
    MODE(label_secondary)                                           AS label_secondary,
    BOOLOR_AGG(is_synthetic)                                        AS is_synthetic,
    MAX(sample_ts)                                                  AS last_sample_ts
FROM s
GROUP BY dog_id, test_num, epoch_ts;

-- ---------------------------------------------------------------------------
-- Historical epoch features. Same computation, bulk source, materialised ONCE.
--
-- This is a table and not a dynamic table on purpose: the historical corpus is
-- immutable, so re-deriving 10.6M rows on a one-minute lag would burn the trial
-- account for no new information. Built by scripts/run_sql.py, or by hand:
--     CALL STAGING.SP_BUILD_BULK_FEATURES();
--
-- sample_ts does not exist in BULK (t_sec is seconds from session start), so it
-- is projected onto a deterministic wall clock: a fixed anchor, one notional
-- day per dog, sessions laid end to end. Deterministic means the deep-history
-- tabs look the same on every rebuild.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE STAGING.SP_BUILD_BULK_FEATURES()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    CREATE OR REPLACE TABLE STAGING.EPOCH_FEATURES_BULK AS
    WITH proj AS (
        SELECT
            dog_id, test_num, t_sec,
            neck_ax, neck_ay, neck_az, neck_gx, neck_gy, neck_gz,
            back_ax, back_ay, back_az, back_gx, back_gy, back_gz,
            label_primary, label_secondary,
            FALSE AS is_synthetic,
            -- anchor + one day per dog + one hour per test session
            DATEADD('millisecond', CAST(t_sec * 1000 AS NUMBER),
              DATEADD('hour', COALESCE(test_num, 1),
                DATEADD('day', dog_id, '2026-01-01 00:00:00'::TIMESTAMP_NTZ)))  AS sample_ts
        FROM RAW.COLLAR_TELEMETRY_BULK
    ),
    d AS (
        SELECT
            dog_id, test_num, sample_ts,
            TIME_SLICE(sample_ts, 1, 'SECOND')                          AS epoch_ts,
            label_primary, label_secondary, is_synthetic,
            SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az)   AS vm_neck,
            SQRT(back_ax*back_ax + back_ay*back_ay + back_az*back_az)   AS vm_back,
            SQRT(neck_gx*neck_gx + neck_gy*neck_gy + neck_gz*neck_gz)   AS gm_neck,
            SQRT(back_gx*back_gx + back_gy*back_gy + back_gz*back_gz)   AS gm_back,
            ABS(neck_ax) + ABS(neck_ay) + ABS(neck_az)                  AS sma_neck,
            ABS(back_ax) + ABS(back_ay) + ABS(back_az)                  AS sma_back,
            ATAN2(neck_ax, SQRT(neck_ay*neck_ay + neck_az*neck_az))     AS pitch_neck,
            ATAN2(neck_ay, SQRT(neck_ax*neck_ax + neck_az*neck_az))     AS roll_neck,
            ATAN2(back_ax, SQRT(back_ay*back_ay + back_az*back_az))     AS pitch_back,
            back_gz                                                     AS yaw_back,
            SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az)
              - LAG(SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az))
                  OVER (PARTITION BY dog_id, test_num ORDER BY sample_ts) AS jerk_neck,
            SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az)
              - AVG(SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az))
                  OVER (PARTITION BY dog_id, test_num, TIME_SLICE(sample_ts,1,'SECOND'))
                                                                        AS vm_neck_ac
        FROM proj
    ),
    s AS (
        SELECT *,
            SIGN(vm_neck_ac) AS sgn,
            LAG(SIGN(vm_neck_ac)) OVER (
                PARTITION BY dog_id, test_num, epoch_ts ORDER BY sample_ts) AS sgn_prev
        FROM d
    )
    SELECT
        dog_id, test_num, epoch_ts,
        COUNT(*)                                                        AS n_samples,
        AVG(vm_neck)                                                    AS vm_neck_mean,
        STDDEV(vm_neck)                                                 AS vm_neck_std,
        MAX(vm_neck) - MIN(vm_neck)                                     AS vm_neck_range,
        SUM(vm_neck * vm_neck)                                          AS energy_neck,
        AVG(sma_neck)                                                   AS sma_neck,
        AVG(ABS(jerk_neck))                                             AS jerk_neck_mean,
        STDDEV(jerk_neck)                                               AS jerk_neck_std,
        SUM(IFF(sgn_prev IS NOT NULL AND sgn <> sgn_prev AND sgn <> 0, 1, 0))
            / NULLIF(COUNT(*) - 1, 0)                                   AS zcr_neck,
        AVG(vm_back)                                                    AS vm_back_mean,
        STDDEV(vm_back)                                                 AS vm_back_std,
        AVG(sma_back)                                                   AS sma_back,
        SUM(vm_back * vm_back)                                          AS energy_back,
        AVG(gm_neck)                                                    AS gyro_neck_mean,
        AVG(gm_back)                                                    AS gyro_back_mean,
        AVG(yaw_back)                                                   AS yaw_mean,
        AVG(ABS(yaw_back))                                              AS yaw_abs_mean,
        ABS(AVG(yaw_back)) / NULLIF(AVG(ABS(yaw_back)), 0)              AS yaw_consistency,
        AVG(pitch_neck)                                                 AS pitch_neck_mean,
        STDDEV(pitch_neck)                                              AS pitch_var,
        AVG(roll_neck)                                                  AS roll_neck_mean,
        STDDEV(roll_neck)                                               AS roll_var,
        AVG(pitch_back)                                                 AS pitch_back_mean,
        CORR(vm_neck, vm_back)                                          AS neck_back_corr,
        STDDEV(vm_neck) / NULLIF(STDDEV(vm_back), 0)                    AS neck_dominance,
        (AVG(sma_neck) + AVG(sma_back)) / 2.0                           AS activity_index,
        MODE(label_primary)                                             AS label_primary,
        MODE(label_secondary)                                           AS label_secondary,
        FALSE                                                           AS is_synthetic,
        MAX(sample_ts)                                                  AS last_sample_ts
    FROM s
    GROUP BY dog_id, test_num, epoch_ts;

    RETURN 'EPOCH_FEATURES_BULK: '
        || (SELECT COUNT(*) FROM STAGING.EPOCH_FEATURES_BULK) || ' epochs from '
        || (SELECT COUNT(*) FROM RAW.COLLAR_TELEMETRY_BULK)   || ' samples';
END;
$$;

-- Placeholder so STAGING.V_EPOCH_ALL below can compile before the procedure has
-- ever run. LIKE copies the dynamic table's column list, which keeps the two
-- shapes in lockstep automatically — add a feature above and the bulk table
-- follows without a second edit. SP_BUILD_BULK_FEATURES then CREATE OR REPLACEs
-- it with the real contents, selecting the same columns in the same order.
CREATE TABLE IF NOT EXISTS STAGING.EPOCH_FEATURES_BULK LIKE STAGING.EPOCH_FEATURES;

-- The union both the classifier and the pattern layer read. Live rows win on
-- collision because they are the ones the demo is watching.
--
-- WHY THIS IS A DYNAMIC TABLE AND NOT A VIEW. It was a view, and MARTS.EPOCH_STATES
-- failed to compile against it:
--
--   Dynamic Tables do not support reading from views containing objects of
--   type DYNAMIC_TABLE, found in V_EPOCH_ALL
--
-- A dynamic table may read another dynamic table directly — that is the whole
-- point of the DAG — but it may not reach one through a view. Probed all three
-- shapes against this account before rewriting: DT→DT, DT→(DT ∪ base table),
-- and DT→model PREDICT all compile; only the view wrapper is refused. So the
-- union becomes a node in the DAG rather than a lens over it.
--
-- TARGET_LAG = DOWNSTREAM: this table has no freshness requirement of its own.
-- It is fresh exactly when something that reads it needs it to be, and
-- Snowflake derives the schedule from the leaf lags. Pinning '1 minute' here
-- would refresh the whole chain on a timer whether or not anyone was looking.
CREATE OR REPLACE DYNAMIC TABLE STAGING.EPOCH_ALL
    TARGET_LAG   = DOWNSTREAM
    WAREHOUSE    = ${SNOWFLAKE_WAREHOUSE}
    REFRESH_MODE = FULL
    INITIALIZE   = ON_CREATE
    COMMENT      = 'Live epochs unioned over the bulk corpus. Live wins on collision.'
AS
SELECT *, 'LIVE'::STRING AS source FROM STAGING.EPOCH_FEATURES
UNION ALL
SELECT b.*, 'BULK'::STRING AS source
FROM STAGING.EPOCH_FEATURES_BULK b
WHERE NOT EXISTS (
    SELECT 1 FROM STAGING.EPOCH_FEATURES l
    WHERE l.dog_id = b.dog_id AND l.test_num = b.test_num AND l.epoch_ts = b.epoch_ts
);

-- Compatibility lens for everything that is NOT a dynamic table: the Gate C
-- views below, the ML views in 05, the dashboard. Views reading dynamic tables
-- are fine — the restriction runs the other way.
CREATE OR REPLACE VIEW STAGING.V_EPOCH_ALL AS
SELECT * FROM STAGING.EPOCH_ALL;

-- ---------------------------------------------------------------------------
-- GATE C VALIDATION. Run these before trusting the feature.
--
-- If locomotion labels do not sit meaningfully above posture and neck-dominant
-- labels on avg_corr, the feature is not doing what you think and you need to
-- know now, not in the write-up.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW STAGING.V_CORR_BY_LABEL AS
SELECT
    label_primary,
    COUNT(*)                                    AS epochs,
    ROUND(AVG(neck_back_corr), 3)               AS avg_corr,
    ROUND(STDDEV(neck_back_corr), 3)            AS sd_corr,
    ROUND(MEDIAN(neck_back_corr), 3)            AS median_corr,
    ROUND(APPROX_PERCENTILE(neck_back_corr, 0.25), 3) AS p25_corr,
    ROUND(APPROX_PERCENTILE(neck_back_corr, 0.75), 3) AS p75_corr,
    ROUND(AVG(neck_dominance), 3)               AS avg_neck_dominance,
    ROUND(AVG(zcr_neck), 3)                     AS avg_zcr,
    ROUND(AVG(vm_neck_std), 3)                  AS avg_vm_neck_std
FROM STAGING.V_EPOCH_ALL
WHERE label_primary IS NOT NULL
GROUP BY label_primary;

-- Ten epochs at each end of the correlation range, with their true labels.
-- If the correlation does not separate them, the feature is wrong.
CREATE OR REPLACE VIEW STAGING.V_CORR_EXTREMES AS
SELECT 'low  (< 0.2)' AS band, dog_id, epoch_ts, label_primary,
       ROUND(neck_back_corr,3) AS corr, ROUND(vm_neck_std,3) AS vm_neck_std,
       ROUND(neck_dominance,3) AS neck_dominance, ROUND(zcr_neck,3) AS zcr
FROM STAGING.V_EPOCH_ALL
WHERE neck_back_corr < 0.2 AND label_primary IS NOT NULL
QUALIFY ROW_NUMBER() OVER (ORDER BY RANDOM(42)) <= 10
UNION ALL
SELECT 'high (> 0.8)', dog_id, epoch_ts, label_primary,
       ROUND(neck_back_corr,3), ROUND(vm_neck_std,3),
       ROUND(neck_dominance,3), ROUND(zcr_neck,3)
FROM STAGING.V_EPOCH_ALL
WHERE neck_back_corr > 0.8 AND label_primary IS NOT NULL
QUALIFY ROW_NUMBER() OVER (ORDER BY RANDOM(43)) <= 10;

-- Epoch quality. n_samples should sit at ~100; anything far below is a gap in
-- the feed and is gated to UNKNOWN downstream rather than classified.
CREATE OR REPLACE VIEW STAGING.V_EPOCH_QUALITY AS
SELECT
    source,
    COUNT(*)                                              AS epochs,
    ROUND(AVG(n_samples), 1)                              AS avg_samples,
    MIN(n_samples)                                        AS min_samples,
    MAX(n_samples)                                        AS max_samples,
    SUM(IFF(n_samples < 60, 1, 0))                        AS below_gate,
    ROUND(100.0 * SUM(IFF(n_samples < 60, 1, 0)) / NULLIF(COUNT(*),0), 2) AS pct_below_gate,
    SUM(IFF(neck_back_corr IS NULL, 1, 0))                AS null_corr
FROM STAGING.V_EPOCH_ALL
GROUP BY source;
