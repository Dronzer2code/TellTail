-- ===========================================================================
-- 08_ml_timeseries.sql   ·   forecast, anomaly, drivers
--
-- Three ML functions over the per-dog activity index.
--
-- THE ANOMALY GOTCHA, PAID FOR ONCE ALREADY: the training and detection windows
-- must not overlap. Split both at the SAME boundary T, strict `<` on the train
-- side and `>=` on the detect side, or Cortex rejects the call with a message
-- about evaluation timestamps needing to follow the fitting data. T is computed
-- once, into a table, so both views cannot disagree about where it is.
-- ===========================================================================

USE DATABASE ${SNOWFLAKE_DATABASE};
USE SCHEMA ML;

-- ---------------------------------------------------------------------------
-- The series everything here runs on: one activity value per dog per minute.
--
-- Minute buckets rather than seconds: ML.FORECAST on 45 series x 106K points is
-- a lot of trial credits for no extra insight, and a per-minute activity index
-- is the right resolution for "is this dog declining" anyway.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ML.ACTIVITY_HISTORY (
    dog_id          NUMBER,
    ts              TIMESTAMP_NTZ,
    activity_index  FLOAT,
    n_epochs        NUMBER,
    is_synthetic    BOOLEAN,
    snapshot_at     TIMESTAMP_NTZ
);

CREATE OR REPLACE PROCEDURE ML.SP_SNAPSHOT_ACTIVITY()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Idempotent by (dog_id, ts): a re-run replaces a bucket rather than
    -- duplicating it, so the task can run on a schedule without drift.
    MERGE INTO ML.ACTIVITY_HISTORY t
    USING (
        SELECT
            dog_id,
            TIME_SLICE(epoch_ts, 60, 'SECOND')  AS ts,
            AVG(activity_index)                 AS activity_index,
            COUNT(*)                            AS n_epochs,
            BOOLOR_AGG(is_synthetic)            AS is_synthetic
        FROM MARTS.ACTIVITY_EPOCH
        GROUP BY dog_id, TIME_SLICE(epoch_ts, 60, 'SECOND')
    ) s
    ON t.dog_id = s.dog_id AND t.ts = s.ts
    WHEN MATCHED THEN UPDATE SET
        t.activity_index = s.activity_index,
        t.n_epochs       = s.n_epochs,
        t.is_synthetic   = s.is_synthetic,
        t.snapshot_at    = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT
        (dog_id, ts, activity_index, n_epochs, is_synthetic, snapshot_at)
        VALUES (s.dog_id, s.ts, s.activity_index, s.n_epochs, s.is_synthetic,
                CURRENT_TIMESTAMP());

    RETURN 'ACTIVITY_HISTORY: ' || (SELECT COUNT(*) FROM ML.ACTIVITY_HISTORY)
        || ' points across ' || (SELECT COUNT(DISTINCT dog_id) FROM ML.ACTIVITY_HISTORY)
        || ' dogs';
END;
$$;

-- ---------------------------------------------------------------------------
-- The split boundary. ONE table, so train and detect cannot drift apart.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ML.SPLIT_BOUNDARY (
    boundary_ts  TIMESTAMP_NTZ,
    computed_at  TIMESTAMP_NTZ,
    detect_window_s NUMBER,
    note         STRING
);

CREATE OR REPLACE PROCEDURE ML.SP_SET_BOUNDARY()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    win NUMBER;
BEGIN
    SELECT value_num INTO :win FROM REF.PARAMS WHERE key = 'anomaly_detect_window';

    DELETE FROM ML.SPLIT_BOUNDARY;
    INSERT INTO ML.SPLIT_BOUNDARY (boundary_ts, computed_at, detect_window_s, note)
    SELECT
        DATEADD('second', -:win, MAX(ts)),
        CURRENT_TIMESTAMP(),
        :win,
        'train: ts < boundary (strict). detect: ts >= boundary. Same boundary, no overlap.'
    FROM ML.ACTIVITY_HISTORY;

    RETURN 'boundary = ' || (SELECT TO_VARCHAR(boundary_ts) FROM ML.SPLIT_BOUNDARY);
END;
$$;

-- STRICT `<` on the training side. Synthetic rows are excluded: the demo spike
-- must be visible to the detector and invisible to the fit, or the detector
-- learns the spike is normal and the demo silently does nothing.
-- DENSE SERIES ONLY.
--
-- ML.ACTIVITY_HISTORY mixes two clocks: dogs replayed into the live path carry
-- current timestamps, while the rest carry the corpus's own. Fed the union,
-- SNOWFLAKE.ML.FORECAST sees 45 series spread over seven months with enormous
-- gaps, spends twenty-plus minutes trying to infer a frequency that does not
-- exist, and produces nothing worth reading even when it returns.
--
-- A forecast needs a regular grid, so this trains only on series that HAVE
-- one: at least `forecast_min_points` observations inside the recent window.
-- Excluded dogs are not hidden — ML.V_FORECAST_COVERAGE below counts them, and
-- the Baselines tab says how many dogs the forecast actually covers.
CREATE OR REPLACE VIEW ML.V_ACTIVITY_TRAIN AS
WITH windowed AS (
    SELECT dog_id, ts, activity_index
    FROM ML.ACTIVITY_HISTORY
    WHERE ts < (SELECT boundary_ts FROM ML.SPLIT_BOUNDARY)
      AND ts >= DATEADD('hour',
                        -(SELECT value_num FROM REF.PARAMS
                           WHERE key = 'forecast_window_hours'),
                        (SELECT boundary_ts FROM ML.SPLIT_BOUNDARY))
      AND NOT COALESCE(is_synthetic, FALSE)
      AND activity_index IS NOT NULL
),
dense AS (
    SELECT dog_id FROM windowed GROUP BY dog_id
    HAVING COUNT(*) >= (SELECT value_num FROM REF.PARAMS
                         WHERE key = 'forecast_min_points')
)
SELECT w.dog_id::VARCHAR AS series, w.ts, w.activity_index
FROM windowed w JOIN dense d ON d.dog_id = w.dog_id;

-- How much of the pack the forecast actually speaks for. A column, not a
-- footnote: a forecast over 12 of 45 dogs is a different claim from one over
-- all of them, and the dashboard has to be able to say which.
CREATE OR REPLACE VIEW ML.V_FORECAST_COVERAGE AS
SELECT
    (SELECT COUNT(DISTINCT dog_id) FROM ML.ACTIVITY_HISTORY)      AS dogs_total,
    (SELECT COUNT(DISTINCT series) FROM ML.V_ACTIVITY_TRAIN)      AS dogs_forecast,
    (SELECT COUNT(*) FROM ML.V_ACTIVITY_TRAIN)                    AS train_points,
    (SELECT value_num FROM REF.PARAMS WHERE key = 'forecast_window_hours')
                                                                  AS window_hours,
    (SELECT value_num FROM REF.PARAMS WHERE key = 'forecast_min_points')
                                                                  AS min_points;

-- `>=` on the detection side. Same boundary. Synthetic rows INCLUDED.
CREATE OR REPLACE VIEW ML.V_ACTIVITY_DETECT AS
SELECT dog_id::VARCHAR AS series, ts, activity_index
FROM ML.ACTIVITY_HISTORY
WHERE ts >= (SELECT boundary_ts FROM ML.SPLIT_BOUNDARY)
  AND activity_index IS NOT NULL;

-- ---------------------------------------------------------------------------
-- FORECAST and ANOMALY_DETECTION.
--
-- Both are wrapped in a procedure with an exception handler that records the
-- failure in ML.FUNCTION_STATUS rather than aborting the build. On a trial
-- account these are the two things most likely to be unavailable, and losing
-- them must not stop the syndrome layer from shipping.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ML.FUNCTION_STATUS (
    fn          STRING,
    status      STRING,        -- OK | FAILED | SKIPPED
    detail      STRING,
    rows_out    NUMBER,
    ran_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- FULLY QUALIFIED VIEW NAMES BELOW. SYSTEM$REFERENCE resolves its argument
-- without the procedure's USE DATABASE / USE SCHEMA context, so a bare
-- 'ML.V_ACTIVITY_TRAIN' fails with "View 'ML.V_ACTIVITY_TRAIN' does not exist
-- or not authorized" even while the view sits there with 2,027 rows in it.
-- Same trap as SYSTEM$QUERY_REFERENCE in 05_ml_classification.sql.
CREATE OR REPLACE PROCEDURE ML.SP_RUN_FORECAST()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    horizon NUMBER;
    n_train NUMBER;
    -- SQLERRM is not a valid identifier inside a DML statement, only inside an
    -- expression assignment. Bare in the INSERT below, the handler itself
    -- raised "invalid identifier 'SQLERRM'" — so the task died on the very line
    -- meant to record why it died, and ML.FUNCTION_STATUS never learned. Park
    -- it in a variable and bind that.
    err      STRING DEFAULT '';
BEGIN
    SELECT value_num INTO :horizon FROM REF.PARAMS WHERE key = 'forecast_horizon';
    SELECT COUNT(*) INTO :n_train FROM ML.V_ACTIVITY_TRAIN;

    IF (:n_train < 100) THEN
        DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'FORECAST';
        INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
        VALUES ('FORECAST', 'SKIPPED',
                'only ' || :n_train || ' training points; needs the replayer to run first', 0);
        RETURN 'FORECAST skipped: insufficient history (' || :n_train || ' points)';
    END IF;

    BEGIN
        CREATE OR REPLACE SNOWFLAKE.ML.FORECAST ML.ACTIVITY_FORECASTER(
            INPUT_DATA          => SYSTEM$REFERENCE('VIEW', '${SNOWFLAKE_DATABASE}.ML.V_ACTIVITY_TRAIN'),
            SERIES_COLNAME      => 'SERIES',
            TIMESTAMP_COLNAME   => 'TS',
            TARGET_COLNAME      => 'ACTIVITY_INDEX'
        );

        CALL ML.ACTIVITY_FORECASTER!FORECAST(FORECASTING_PERIODS => :horizon);

        CREATE OR REPLACE TABLE ML.ACTIVITY_FORECAST AS
        SELECT
            series::NUMBER                        AS dog_id,
            ts                                    AS forecast_ts,
            forecast,
            lower_bound,
            upper_bound,
            CURRENT_TIMESTAMP()                   AS generated_at
        FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

        DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'FORECAST';
        INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
        SELECT 'FORECAST', 'OK', 'horizon=' || :horizon, COUNT(*) FROM ML.ACTIVITY_FORECAST;

        RETURN 'FORECAST: ' || (SELECT COUNT(*) FROM ML.ACTIVITY_FORECAST) || ' points';
    EXCEPTION
        WHEN OTHER THEN
            err := SQLERRM;
            CREATE TABLE IF NOT EXISTS ML.ACTIVITY_FORECAST (
                dog_id NUMBER, forecast_ts TIMESTAMP_NTZ, forecast FLOAT,
                lower_bound FLOAT, upper_bound FLOAT, generated_at TIMESTAMP_NTZ);
            DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'FORECAST';
            INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
            VALUES ('FORECAST', 'FAILED', :err, 0);
            RETURN 'FORECAST failed (recorded, build continues): ' || SQLERRM;
    END;
END;
$$;

CREATE OR REPLACE PROCEDURE ML.SP_RUN_ANOMALY()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    n_train  NUMBER;
    n_detect NUMBER;
    err      STRING DEFAULT '';
BEGIN
    SELECT COUNT(*) INTO :n_train  FROM ML.V_ACTIVITY_TRAIN;
    SELECT COUNT(*) INTO :n_detect FROM ML.V_ACTIVITY_DETECT;

    IF (:n_train < 100 OR :n_detect < 5) THEN
        DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'ANOMALY_DETECTION';
        INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
        VALUES ('ANOMALY_DETECTION', 'SKIPPED',
                'train=' || :n_train || ' detect=' || :n_detect, 0);
        RETURN 'ANOMALY skipped: train=' || :n_train || ' detect=' || :n_detect;
    END IF;

    BEGIN
        CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION ML.ACTIVITY_DETECTOR(
            INPUT_DATA        => SYSTEM$REFERENCE('VIEW', '${SNOWFLAKE_DATABASE}.ML.V_ACTIVITY_TRAIN'),
            SERIES_COLNAME    => 'SERIES',
            TIMESTAMP_COLNAME => 'TS',
            TARGET_COLNAME    => 'ACTIVITY_INDEX',
            LABEL_COLNAME     => ''
        );

        CALL ML.ACTIVITY_DETECTOR!DETECT_ANOMALIES(
            INPUT_DATA        => SYSTEM$REFERENCE('VIEW', '${SNOWFLAKE_DATABASE}.ML.V_ACTIVITY_DETECT'),
            SERIES_COLNAME    => 'SERIES',
            TIMESTAMP_COLNAME => 'TS',
            TARGET_COLNAME    => 'ACTIVITY_INDEX'
        );

        CREATE OR REPLACE TABLE ML.ACTIVITY_ANOMALIES AS
        SELECT
            r.series::NUMBER      AS dog_id,
            r.ts                  AS anomaly_ts,
            r.y                   AS observed,
            r.forecast,
            r.lower_bound,
            r.upper_bound,
            r.is_anomaly,
            r.percentile,
            r.distance,
            -- carried through so the UI can say "this point is the injected
            -- spike" on camera instead of the viewer having to trust us
            COALESCE(h.is_synthetic, FALSE) AS is_synthetic,
            CURRENT_TIMESTAMP()   AS generated_at
        FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) r
        LEFT JOIN ML.ACTIVITY_HISTORY h
               ON h.dog_id = r.series::NUMBER AND h.ts = r.ts;

        DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'ANOMALY_DETECTION';
        INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
        SELECT 'ANOMALY_DETECTION', 'OK',
               'anomalies=' || SUM(IFF(is_anomaly, 1, 0)), COUNT(*)
        FROM ML.ACTIVITY_ANOMALIES;

        RETURN 'ANOMALY: ' || (SELECT SUM(IFF(is_anomaly,1,0)) FROM ML.ACTIVITY_ANOMALIES)
            || ' flagged of ' || (SELECT COUNT(*) FROM ML.ACTIVITY_ANOMALIES);
    EXCEPTION
        WHEN OTHER THEN
            err := SQLERRM;
            CREATE TABLE IF NOT EXISTS ML.ACTIVITY_ANOMALIES (
                dog_id NUMBER, anomaly_ts TIMESTAMP_NTZ, observed FLOAT, forecast FLOAT,
                lower_bound FLOAT, upper_bound FLOAT, is_anomaly BOOLEAN,
                percentile FLOAT, distance FLOAT, is_synthetic BOOLEAN,
                generated_at TIMESTAMP_NTZ);
            DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'ANOMALY_DETECTION';
            INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
            VALUES ('ANOMALY_DETECTION', 'FAILED', :err, 0);
            RETURN 'ANOMALY failed (recorded, build continues): ' || SQLERRM;
    END;
END;
$$;

-- ---------------------------------------------------------------------------
-- TOP_INSIGHTS — the Drivers tab.
--
-- Question: which dimension values move the deviation metric in surprising
-- ways? Not "which breed has the highest deviation" (that is a GROUP BY) but
-- "which slice contributes disproportionately to the change".
--
-- The input contrasts two groups: epochs inside a syndrome window (label TRUE)
-- against everything else (label FALSE), with breed, age band, weight band, sex
-- and state provenance as dimensions.
--
-- NOTE ON THE CALL SIGNATURE: Snowflake's Top Insights signature has moved
-- between preview and GA. Rather than let a signature mismatch take the tab
-- down, the procedure tries the function and, on any failure, computes a
-- transparent contribution decomposition in plain SQL instead. Both paths write
-- the same table, and ML.DRIVER_INSIGHTS.method records which one produced the
-- rows. The fallback is a real technique, not a placeholder: per dimension
-- value it reports the metric lift, the share of population, and the product of
-- the two, which is the contribution to the overall difference.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ML.V_INSIGHT_INPUT AS
SELECT
    COALESCE(d.breed, 'unknown')                       AS breed,
    COALESCE(d.age_band, 'unknown')                    AS age_band,
    COALESCE(d.weight_band, 'unknown')                 AS weight_band,
    COALESCE(d.sex, 'unknown')                         AS sex,
    s.state_source                                     AS state_source,
    ABS(COALESCE(dev.z_self, 0))                       AS deviation,
    IFF(EXISTS (
            SELECT 1 FROM MARTS.SYNDROME_MATCHES m
            WHERE m.dog_id = s.dog_id
              AND s.epoch_ts BETWEEN m.onset_ts AND m.resolve_ts
        ), TRUE, FALSE)                                AS in_syndrome
FROM MARTS.EPOCH_STATES s
LEFT JOIN REF.V_DOG_COHORT d   ON d.dog_id = s.dog_id
LEFT JOIN MARTS.DOG_DEVIATION dev
       ON dev.dog_id = s.dog_id AND dev.epoch_ts = s.epoch_ts
WHERE s.state <> 'UNKNOWN';

CREATE OR REPLACE PROCEDURE ML.SP_RUN_TOP_INSIGHTS()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    err STRING DEFAULT '';
BEGIN
    BEGIN
        CREATE OR REPLACE TABLE ML.DRIVER_INSIGHTS AS
        SELECT
            'TOP_INSIGHTS'::STRING       AS method,
            *,
            CURRENT_TIMESTAMP()          AS generated_at
        FROM TABLE(SNOWFLAKE.ML.TOP_INSIGHTS(
            SELECT
                {'breed':        breed,
                 'age_band':     age_band,
                 'weight_band':  weight_band,
                 'sex':          sex,
                 'state_source': state_source}   AS dimensions,
                deviation                        AS metric,
                in_syndrome                      AS label
            FROM ML.V_INSIGHT_INPUT
        ));

        DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'TOP_INSIGHTS';
        INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
        SELECT 'TOP_INSIGHTS', 'OK', 'native function', COUNT(*) FROM ML.DRIVER_INSIGHTS;
        RETURN 'TOP_INSIGHTS (native): '
            || (SELECT COUNT(*) FROM ML.DRIVER_INSIGHTS) || ' insights';
    EXCEPTION
        WHEN OTHER THEN
            err := SQLERRM;
            -- Transparent contribution decomposition. For every dimension value:
            --   lift         = mean(metric | in syndrome) - mean(metric | not)
            --   share        = fraction of all epochs in this slice
            --   contribution = lift * share, i.e. how much of the overall gap
            --                  this slice is responsible for
            CREATE OR REPLACE TABLE ML.DRIVER_INSIGHTS AS
            WITH unpivoted AS (
                SELECT 'breed'        AS dimension, breed        AS value, deviation, in_syndrome FROM ML.V_INSIGHT_INPUT
                UNION ALL SELECT 'age_band',     age_band,     deviation, in_syndrome FROM ML.V_INSIGHT_INPUT
                UNION ALL SELECT 'weight_band',  weight_band,  deviation, in_syndrome FROM ML.V_INSIGHT_INPUT
                UNION ALL SELECT 'sex',          sex,          deviation, in_syndrome FROM ML.V_INSIGHT_INPUT
                UNION ALL SELECT 'state_source', state_source, deviation, in_syndrome FROM ML.V_INSIGHT_INPUT
            ),
            agg AS (
                SELECT
                    dimension, value,
                    COUNT(*)                                                    AS n,
                    AVG(IFF(in_syndrome, deviation, NULL))                      AS mean_in,
                    AVG(IFF(in_syndrome, NULL, deviation))                      AS mean_out,
                    SUM(IFF(in_syndrome, 1, 0))                                 AS n_in
                FROM unpivoted
                GROUP BY dimension, value
            ),
            totals AS (SELECT dimension, SUM(n) AS n_total FROM agg GROUP BY dimension)
            SELECT
                'SQL_CONTRIBUTION'::STRING                                      AS method,
                a.dimension,
                a.value,
                a.n,
                a.n_in,
                ROUND(a.mean_in, 4)                                             AS mean_in_syndrome,
                ROUND(a.mean_out, 4)                                            AS mean_out_syndrome,
                ROUND(COALESCE(a.mean_in, 0) - COALESCE(a.mean_out, 0), 4)      AS lift,
                ROUND(a.n / NULLIF(t.n_total, 0), 4)                            AS share,
                ROUND((COALESCE(a.mean_in, 0) - COALESCE(a.mean_out, 0))
                      * (a.n / NULLIF(t.n_total, 0)), 5)                        AS contribution,
                CURRENT_TIMESTAMP()                                             AS generated_at
            FROM agg a
            JOIN totals t ON t.dimension = a.dimension
            WHERE a.n >= 50            -- a slice of nine epochs is not a driver
            ORDER BY ABS(contribution) DESC;

            -- :err, not SQLERRM. Concatenation does not make it legal — this
            -- is still SQLERRM inside a DML statement, and unhandled it took
            -- T_DRIVERS down on the line meant to record why the native call
            -- had failed, losing both the insights and the reason.
            DELETE FROM ML.FUNCTION_STATUS WHERE fn = 'TOP_INSIGHTS';
            INSERT INTO ML.FUNCTION_STATUS (fn, status, detail, rows_out)
            SELECT 'TOP_INSIGHTS', 'OK',
                   'SQL contribution fallback: ' || :err, COUNT(*) FROM ML.DRIVER_INSIGHTS;
            RETURN 'TOP_INSIGHTS (SQL fallback): '
                || (SELECT COUNT(*) FROM ML.DRIVER_INSIGHTS) || ' rows. Native call said: '
                || :err;
    END;
END;
$$;

-- ---------------------------------------------------------------------------
-- The dumbbell chart on the Baselines tab: current index -> projected index,
-- so who is declining reads at a glance.
-- ---------------------------------------------------------------------------
-- ---------------------------------------------------------------------------
-- Placeholders so the reporting views below compile on a cold warehouse.
--
-- The procedures above create these, and the bootstrap calls procedures only
-- after every statement in this file has run — so on a fresh account the very
-- first view that reads a forecast fails before any of them exists. Identical
-- shapes to the ones the procedures' own exception handlers declare, and
-- IF NOT EXISTS so a rebuild never discards real forecasts.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ML.ACTIVITY_FORECAST (
    dog_id NUMBER, forecast_ts TIMESTAMP_NTZ, forecast FLOAT,
    lower_bound FLOAT, upper_bound FLOAT, generated_at TIMESTAMP_NTZ);

CREATE TABLE IF NOT EXISTS ML.ACTIVITY_ANOMALIES (
    dog_id NUMBER, anomaly_ts TIMESTAMP_NTZ, observed FLOAT, forecast FLOAT,
    lower_bound FLOAT, upper_bound FLOAT, is_anomaly BOOLEAN,
    percentile FLOAT, distance FLOAT, is_synthetic BOOLEAN,
    generated_at TIMESTAMP_NTZ);

CREATE OR REPLACE VIEW ML.V_TRAJECTORY AS
WITH cur AS (
    SELECT dog_id, AVG(activity_index) AS current_index
    FROM ML.ACTIVITY_HISTORY
    WHERE ts >= (SELECT DATEADD('minute', -15, MAX(ts)) FROM ML.ACTIVITY_HISTORY)
    GROUP BY dog_id
),
fut AS (
    SELECT dog_id, AVG(forecast) AS projected_index,
           AVG(lower_bound) AS projected_lower, AVG(upper_bound) AS projected_upper
    FROM ML.ACTIVITY_FORECAST
    GROUP BY dog_id
)
SELECT
    c.dog_id,
    d.breed, d.age_band, d.weight_band,
    ROUND(c.current_index, 4)                                    AS current_index,
    ROUND(f.projected_index, 4)                                  AS projected_index,
    ROUND(f.projected_lower, 4)                                  AS projected_lower,
    ROUND(f.projected_upper, 4)                                  AS projected_upper,
    ROUND(f.projected_index - c.current_index, 4)                AS delta,
    ROUND(100.0 * (f.projected_index - c.current_index)
          / NULLIF(c.current_index, 0), 2)                       AS pct_change
FROM cur c
LEFT JOIN fut f            ON f.dog_id = c.dog_id
LEFT JOIN REF.V_DOG_COHORT d ON d.dog_id = c.dog_id;
