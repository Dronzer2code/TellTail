-- ===========================================================================
-- 05_ml_classification.sql   ·   the ethogram classifier
--
-- Every epoch gets exactly one state, and the split that produces the accuracy
-- number is BY DOG, never by row.
--
-- Why that matters more than anything else in this file: at 100 Hz, samples
-- 40 ms apart are near-identical. A random row split puts adjacent samples from
-- the same dog on both sides, the model memorises the individual, and the
-- reported accuracy is a fiction. The literature on this dataset reports
-- single-subject classifiers falling from ~91% to ~70-74% when generalising to
-- dogs they were not trained on. Hold out whole dogs, report the lower number,
-- and print it large rather than burying it.
-- ===========================================================================

USE DATABASE ${SNOWFLAKE_DATABASE};
USE SCHEMA ML;

-- ---------------------------------------------------------------------------
-- The labelled epoch set, with raw labels mapped to ethogram states.
--
-- IDENTIFIERS ARE NOT FEATURES. dog_id, test_num and epoch_ts are carried for
-- the split and for joining back, and are dropped before the model sees a row.
-- Leaving dog_id in the feature vector would let the model learn "dog 23 mostly
-- trots" and would invalidate the entire holdout protocol.
-- ---------------------------------------------------------------------------
-- STATE RESOLUTION FALLS BACK TO THE SECONDARY ANNOTATION, and it has to.
--
-- The dataset carries up to three simultaneous behaviour annotations, and the
-- POSTURE is not always in the first one. 'Panting' is the primary annotation
-- for 836K rows, and it is a respiratory behaviour rather than a posture — the
-- actual posture sits in Behavior_2 (48% Sitting, 48% Standing, 4% Lying).
-- Mapping Panting to any single posture would be wrong about half the time on
-- 7.9% of the corpus; dropping those rows would throw away the same 7.9%.
-- COALESCE over the two columns recovers the true posture instead.
CREATE OR REPLACE VIEW ML.V_LABELLED_EPOCHS AS
SELECT
    e.dog_id,
    e.test_num,
    e.epoch_ts,
    COALESCE(m1.state, m2.state)                       AS state,
    IFF(m1.state IS NULL, 'label_secondary', 'label_primary') AS label_source,

    -- ---- feature vector (25 columns) ----
    e.vm_neck_mean, e.vm_neck_std, e.vm_neck_range, e.energy_neck, e.sma_neck,
    e.jerk_neck_mean, e.jerk_neck_std, e.zcr_neck,
    e.vm_back_mean, e.vm_back_std, e.sma_back, e.energy_back,
    e.gyro_neck_mean, e.gyro_back_mean,
    e.yaw_mean, e.yaw_abs_mean, e.yaw_consistency,
    e.pitch_neck_mean, e.pitch_var, e.roll_neck_mean, e.roll_var, e.pitch_back_mean,
    e.neck_back_corr, e.neck_dominance, e.activity_index
FROM STAGING.V_EPOCH_ALL e
LEFT JOIN REF.LABEL_MAP m1
       ON m1.raw_label = e.label_primary
      AND m1.source_column = 'label_primary'
LEFT JOIN REF.LABEL_MAP m2
       ON m2.raw_label = e.label_secondary
      AND m2.source_column = 'label_secondary'
CROSS JOIN REF.V_PARAM p
WHERE COALESCE(m1.state, m2.state) IS NOT NULL
  AND e.n_samples >= p.o:epoch_min_samples::NUMBER   -- quality gate
  AND e.neck_back_corr IS NOT NULL                   -- a constant epoch has no correlation
  AND NOT e.is_synthetic;                            -- training NEVER fits injected rows

-- Train: every dog not held out. Feature columns only + the target.
CREATE OR REPLACE VIEW ML.V_TRAIN AS
SELECT
    state,
    vm_neck_mean, vm_neck_std, vm_neck_range, energy_neck, sma_neck,
    jerk_neck_mean, jerk_neck_std, zcr_neck,
    vm_back_mean, vm_back_std, sma_back, energy_back,
    gyro_neck_mean, gyro_back_mean,
    yaw_mean, yaw_abs_mean, yaw_consistency,
    pitch_neck_mean, pitch_var, roll_neck_mean, roll_var, pitch_back_mean,
    neck_back_corr, neck_dominance, activity_index
FROM ML.V_LABELLED_EPOCHS
WHERE dog_id NOT IN (SELECT dog_id FROM REF.HOLDOUT_DOGS);

-- Test: entire dogs the model has never seen. Keeps dog_id so per-dog accuracy
-- is reportable — generalisation is not uniform across individuals and that is
-- itself a finding.
CREATE OR REPLACE VIEW ML.V_TEST AS
SELECT *
FROM ML.V_LABELLED_EPOCHS
WHERE dog_id IN (SELECT dog_id FROM REF.HOLDOUT_DOGS);

CREATE OR REPLACE VIEW ML.V_SPLIT_SUMMARY AS
SELECT
    IFF(dog_id IN (SELECT dog_id FROM REF.HOLDOUT_DOGS), 'holdout', 'train') AS split,
    COUNT(DISTINCT dog_id)   AS dogs,
    COUNT(*)                 AS epochs,
    COUNT(DISTINCT state)    AS states
FROM ML.V_LABELLED_EPOCHS
GROUP BY 1;

CREATE OR REPLACE VIEW ML.V_CLASS_BALANCE AS
SELECT
    state,
    COUNT(*)                                                        AS epochs,
    ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2)   AS pct,
    COUNT(DISTINCT dog_id)                                          AS dogs
FROM ML.V_LABELLED_EPOCHS
GROUP BY state;

-- ---------------------------------------------------------------------------
-- The transparent SQL rules ethogram.
--
-- This is not a stub. If SNOWFLAKE.ML.CLASSIFICATION is unavailable on the
-- account, or if REF.PARAMS.use_rules_classifier is set to 1, this is the
-- classifier — every threshold is a row in REF.PARAMS, on screen, tunable, and
-- labelled state_source='RULES' wherever it appears. The syndromes are the
-- submission, not the classifier; losing the model must not lose the day.
--
-- A dynamic table rather than a view for the same reason STAGING.EPOCH_ALL is
-- one: ML.STATE_PREDICTION is a dynamic table, and a dynamic table cannot
-- reach another one through a view. TARGET_LAG = DOWNSTREAM — it is fresh when
-- its reader needs it.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE DYNAMIC TABLE ML.RULES_STATE
    TARGET_LAG   = DOWNSTREAM
    WAREHOUSE    = ${SNOWFLAKE_WAREHOUSE}
    REFRESH_MODE = FULL
    INITIALIZE   = ON_CREATE
    COMMENT      = 'Transparent threshold ethogram. Every threshold is a row in REF.PARAMS.'
AS
SELECT
    e.dog_id, e.test_num, e.epoch_ts,
    CASE
        WHEN e.vm_neck_std < p.o:rules_rest_vm_std_max::FLOAT
         AND e.pitch_var   < p.o:rules_rest_pitch_var_max::FLOAT       THEN 'REST'
        WHEN e.vm_neck_std > p.o:shake_vm_std_min::FLOAT
         AND e.neck_back_corr < p.o:shake_corr_max::FLOAT
         AND e.neck_dominance > p.o:neck_dominance_min::FLOAT          THEN 'SHAKE'
        WHEN e.vm_neck_std BETWEEN p.o:scratch_vm_std_min::FLOAT
                               AND p.o:scratch_vm_std_max::FLOAT
         AND e.neck_back_corr < p.o:scratch_corr_max::FLOAT
         AND e.neck_dominance > p.o:neck_dominance_min::FLOAT          THEN 'SCRATCH'
        -- SNIFF before the locomotion rules. S6 (GI discomfort) is built on
        -- SNIFF and CIRCLE, so omitting it here would leave the rules-only path
        -- unable to express one of the six syndromes.
        WHEN e.pitch_neck_mean < p.o:rules_sniff_pitch_max::FLOAT
         AND e.vm_neck_std BETWEEN p.o:rules_sniff_vm_std_min::FLOAT
                               AND p.o:rules_sniff_vm_std_max::FLOAT   THEN 'SNIFF'
        WHEN e.neck_back_corr > p.o:rules_gallop_corr_min::FLOAT
         AND e.vm_neck_mean   > p.o:rules_gallop_vm_min::FLOAT         THEN 'GALLOP'
        WHEN e.neck_back_corr > p.o:rules_trot_corr_min::FLOAT
         AND e.vm_neck_mean   > p.o:rules_trot_vm_min::FLOAT           THEN 'TROT'
        WHEN e.neck_back_corr > p.o:rules_walk_corr_min::FLOAT
         AND e.vm_neck_mean   > p.o:rules_walk_vm_min::FLOAT           THEN 'WALK'
        ELSE 'STAND'
    END                                                               AS state,
    NULL::FLOAT                                                       AS confidence,
    'RULES'::STRING                                                   AS state_source
FROM STAGING.EPOCH_ALL e
CROSS JOIN REF.V_PARAM p;

-- Compatibility lens for readers that are not dynamic tables.
CREATE OR REPLACE VIEW ML.V_RULES_STATE AS
SELECT * FROM ML.RULES_STATE;

-- ---------------------------------------------------------------------------
-- FEATURE SEPARATION — our own answer to "what does the signal actually carry?"
--
-- This exists because SHOW_FEATURE_IMPORTANCE does not run on this account
-- (see the accessor block in SP_TRAIN_STATE_MODEL). Rather than ship a blank
-- panel, compute the ranking directly: a one-way ANOVA F-ratio per feature
-- over the labelled epochs — between-class variance of the class means divided
-- by the mean within-class variance.
--
-- It answers a slightly different question than a tree model's split-gain
-- importance: this is how well a feature separates the ethogram states ON ITS
-- OWN, ignoring what the model chose to lean on given correlated alternatives.
-- Stated plainly on the dashboard rather than passed off as the model's.
--
-- Unpivoting by name keeps this honest: add a feature to EPOCH_FEATURES and it
-- must be added here too, so nothing gets silently ranked or silently omitted.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ML.V_FEATURE_SEPARATION AS
WITH long AS (
    SELECT state, f.key::STRING AS feature, f.value::FLOAT AS val
    FROM ML.V_LABELLED_EPOCHS e,
    LATERAL FLATTEN(input => OBJECT_CONSTRUCT(
        'NECK_BACK_CORR',  e.neck_back_corr,
        'NECK_DOMINANCE',  e.neck_dominance,
        'VM_NECK_MEAN',    e.vm_neck_mean,
        'VM_NECK_STD',     e.vm_neck_std,
        'VM_NECK_RANGE',   e.vm_neck_range,
        'ENERGY_NECK',     e.energy_neck,
        'SMA_NECK',        e.sma_neck,
        'JERK_NECK_MEAN',  e.jerk_neck_mean,
        'JERK_NECK_STD',   e.jerk_neck_std,
        'ZCR_NECK',        e.zcr_neck,
        'VM_BACK_MEAN',    e.vm_back_mean,
        'VM_BACK_STD',     e.vm_back_std,
        'SMA_BACK',        e.sma_back,
        'ENERGY_BACK',     e.energy_back,
        'GYRO_NECK_MEAN',  e.gyro_neck_mean,
        'GYRO_BACK_MEAN',  e.gyro_back_mean,
        'YAW_MEAN',        e.yaw_mean,
        'YAW_ABS_MEAN',    e.yaw_abs_mean,
        'YAW_CONSISTENCY', e.yaw_consistency,
        'PITCH_NECK_MEAN', e.pitch_neck_mean,
        'PITCH_VAR',       e.pitch_var,
        'ROLL_NECK_MEAN',  e.roll_neck_mean,
        'ROLL_VAR',        e.roll_var,
        'PITCH_BACK_MEAN', e.pitch_back_mean,
        'ACTIVITY_INDEX',  e.activity_index
    )) f
    WHERE f.value IS NOT NULL
),
per_class AS (
    SELECT feature, state, COUNT(*) AS n, AVG(val) AS class_mean,
           VARIANCE_POP(val) AS class_var
    FROM long GROUP BY feature, state
),
grand AS (
    SELECT feature, AVG(val) AS grand_mean, COUNT(*) AS n_total
    FROM long GROUP BY feature
)
SELECT
    c.feature,
    -- between-class variance, weighted by class size
    SUM(c.n * POWER(c.class_mean - g.grand_mean, 2)) / NULLIF(COUNT(*) - 1, 0)  AS between_var,
    -- pooled within-class variance
    SUM(c.n * c.class_var) / NULLIF(SUM(c.n) - COUNT(*), 0)                     AS within_var,
    (SUM(c.n * POWER(c.class_mean - g.grand_mean, 2)) / NULLIF(COUNT(*) - 1, 0))
      / NULLIF(SUM(c.n * c.class_var) / NULLIF(SUM(c.n) - COUNT(*), 0), 0)      AS f_ratio,
    COUNT(*)                                                                    AS n_classes,
    SUM(c.n)                                                                    AS n_epochs
FROM per_class c
JOIN grand g ON g.feature = c.feature
GROUP BY c.feature;

-- ---------------------------------------------------------------------------
-- Did the model train, and did its introspection accessors work? A column,
-- not a comment: the dashboard reads this instead of guessing why a panel is
-- empty.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ML.MODEL_STATUS (
    classifier      STRING,
    trained_at      TIMESTAMP_NTZ,
    n_train         NUMBER,
    accessors_ok    BOOLEAN,
    accessor_error  STRING
);

-- ---------------------------------------------------------------------------
-- Train, evaluate, and pick a prediction source.
--
-- The procedure decides which of the two classifiers backs
-- ML.V_STATE_PREDICTION, so 06_marts_dt.sql has one stable interface to read
-- and does not care which path ran. Whichever it is, state_source says so in
-- the data, not in a comment.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE ML.SP_TRAIN_STATE_MODEL()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    use_rules  NUMBER  DEFAULT 0;
    n_train    NUMBER  DEFAULT 0;
    msg        STRING  DEFAULT '';
    -- SQLERRM reads fine inside an expression assignment but is not a valid
    -- identifier inside a DML statement. Park it in a variable and bind that.
    err        STRING  DEFAULT '';
BEGIN
    SELECT COALESCE(value_num, 0) INTO :use_rules
      FROM REF.PARAMS WHERE key = 'use_rules_classifier';
    SELECT COUNT(*) INTO :n_train FROM ML.V_TRAIN;

    IF (:n_train = 0) THEN
        RETURN 'ABORT: ML.V_TRAIN is empty. Either the bulk load has not run, or '
            || 'REF.LABEL_MAP is unpopulated (Gate A output not pushed). '
            || 'Check RAW.V_LABEL_COVERAGE.';
    END IF;

    IF (:use_rules = 1) THEN
        CREATE OR REPLACE DYNAMIC TABLE ML.STATE_PREDICTION
            TARGET_LAG = DOWNSTREAM WAREHOUSE = ${SNOWFLAKE_WAREHOUSE}
            REFRESH_MODE = FULL INITIALIZE = ON_CREATE
            COMMENT = 'One predicted state per epoch. state_source says which classifier.'
        AS SELECT dog_id, test_num, epoch_ts, state, confidence, state_source
           FROM ML.RULES_STATE;
        CREATE OR REPLACE VIEW ML.V_STATE_PREDICTION AS
            SELECT * FROM ML.STATE_PREDICTION;
        DELETE FROM ML.MODEL_STATUS;
        INSERT INTO ML.MODEL_STATUS
            SELECT 'RULES', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ, :n_train, FALSE,
                   'rules classifier pinned by REF.PARAMS.use_rules_classifier=1';
        RETURN 'RULES classifier selected by REF.PARAMS.use_rules_classifier=1 ('
            || :n_train || ' labelled epochs available)';
    END IF;

    BEGIN
        CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION ML.STATE_MODEL(
            -- FULLY QUALIFIED. SYSTEM$QUERY_REFERENCE resolves its text without the
            -- procedure's USE DATABASE/SCHEMA context, so a bare ML.V_TRAIN
            -- fails with "does not exist or not authorized" and the handler
            -- below silently drops to the rules classifier.
            INPUT_DATA     => SYSTEM$QUERY_REFERENCE(
                'SELECT * FROM ${SNOWFLAKE_DATABASE}.ML.V_TRAIN'),
            TARGET_COLNAME => 'STATE',
            CONFIG_OBJECT  => {'on_error': 'skip'}
        );

        -- Record the win before touching anything that can fail. If an
        -- accessor blows up below, this row still says a real model trained.
        DELETE FROM ML.MODEL_STATUS;
        INSERT INTO ML.MODEL_STATUS
            SELECT 'ML.CLASSIFICATION', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ,
                   :n_train, TRUE, NULL;

        -- The model's introspection accessors, in their OWN handler.
        --
        -- On this account every accessor — SHOW_EVALUATION_METRICS,
        -- SHOW_FEATURE_IMPORTANCE, SHOW_TRAINING_LOGS — raises
        -- "Computation Error in function __SHOW_*". PREDICT works perfectly.
        -- Verified it is not our data: the same failure reproduces after
        -- dropping the rare classes, and after training from a materialised
        -- table instead of a view. It is the introspection API on this
        -- Snowflake version, not the model.
        --
        -- Without this nested block that failure propagates to the outer
        -- handler, which throws away a model that trained fine and silently
        -- drops the whole build to the rules classifier. Losing the optional
        -- reporting tables is survivable; losing the classifier is not.
        --
        -- Neither table is load-bearing. SHOW_EVALUATION_METRICS reports the
        -- model's internal split, which shares dogs across the fold, so it was
        -- never the headline number — ML.SP_EVALUATE_HOLDOUT computes that on
        -- entirely unseen dogs. Feature importance is replaced by
        -- ML.FEATURE_SEPARATION below, which we compute ourselves.
        BEGIN
            CREATE OR REPLACE TABLE ML.MODEL_EVAL AS
                SELECT * FROM TABLE(ML.STATE_MODEL!SHOW_EVALUATION_METRICS());
            UPDATE ML.MODEL_STATUS SET accessors_ok = TRUE, accessor_error = NULL;
        EXCEPTION
            WHEN OTHER THEN
                err := SQLERRM;
                CREATE OR REPLACE TABLE ML.MODEL_EVAL (
                    metric STRING, value FLOAT, note STRING);
                UPDATE ML.MODEL_STATUS
                   SET accessors_ok   = FALSE,
                       accessor_error = :err;
        END;

        -- Inference as a node in the DAG, not a lens over one. A dynamic table
        -- may call the model directly (probed against this account before
        -- relying on it), which is what lets MARTS.EPOCH_STATES read predicted
        -- states without a view in between.
        CREATE OR REPLACE DYNAMIC TABLE ML.STATE_PREDICTION
            TARGET_LAG = DOWNSTREAM WAREHOUSE = ${SNOWFLAKE_WAREHOUSE}
            REFRESH_MODE = FULL INITIALIZE = ON_CREATE
            COMMENT = 'One predicted state per epoch. state_source says which classifier.'
        AS
            SELECT
                e.dog_id, e.test_num, e.epoch_ts,
                pred:class::STRING                                      AS state,
                GREATEST(
                  COALESCE(pred:probability[pred:class::STRING]::FLOAT, 0)
                )                                                       AS confidence,
                'MODEL'::STRING                                         AS state_source
            FROM (
                SELECT
                    f.dog_id, f.test_num, f.epoch_ts,
                    ML.STATE_MODEL!PREDICT(OBJECT_CONSTRUCT(
                        'VM_NECK_MEAN',   f.vm_neck_mean,
                        'VM_NECK_STD',    f.vm_neck_std,
                        'VM_NECK_RANGE',  f.vm_neck_range,
                        'ENERGY_NECK',    f.energy_neck,
                        'SMA_NECK',       f.sma_neck,
                        'JERK_NECK_MEAN', f.jerk_neck_mean,
                        'JERK_NECK_STD',  f.jerk_neck_std,
                        'ZCR_NECK',       f.zcr_neck,
                        'VM_BACK_MEAN',   f.vm_back_mean,
                        'VM_BACK_STD',    f.vm_back_std,
                        'SMA_BACK',       f.sma_back,
                        'ENERGY_BACK',    f.energy_back,
                        'GYRO_NECK_MEAN', f.gyro_neck_mean,
                        'GYRO_BACK_MEAN', f.gyro_back_mean,
                        'YAW_MEAN',       f.yaw_mean,
                        'YAW_ABS_MEAN',   f.yaw_abs_mean,
                        'YAW_CONSISTENCY',f.yaw_consistency,
                        'PITCH_NECK_MEAN',f.pitch_neck_mean,
                        'PITCH_VAR',      f.pitch_var,
                        'ROLL_NECK_MEAN', f.roll_neck_mean,
                        'ROLL_VAR',       f.roll_var,
                        'PITCH_BACK_MEAN',f.pitch_back_mean,
                        'NECK_BACK_CORR', f.neck_back_corr,
                        'NECK_DOMINANCE', f.neck_dominance,
                        'ACTIVITY_INDEX', f.activity_index
                    ))                                                  AS pred
                FROM STAGING.EPOCH_ALL f
                WHERE f.neck_back_corr IS NOT NULL
            ) e;

        CREATE OR REPLACE VIEW ML.V_STATE_PREDICTION AS
            SELECT * FROM ML.STATE_PREDICTION;

        msg := 'MODEL trained on ' || :n_train || ' epochs from '
            || (SELECT COUNT(*) FROM (SELECT DISTINCT dog_id FROM ML.V_LABELLED_EPOCHS
                                      WHERE dog_id NOT IN (SELECT dog_id FROM REF.HOLDOUT_DOGS)))
            || ' dogs';
    EXCEPTION
        WHEN OTHER THEN
            -- CLASSIFICATION unavailable on this account/region. Do not lose the
            -- day: fall back, and record why in the data.
            CREATE OR REPLACE DYNAMIC TABLE ML.STATE_PREDICTION
                TARGET_LAG = DOWNSTREAM WAREHOUSE = ${SNOWFLAKE_WAREHOUSE}
                REFRESH_MODE = FULL INITIALIZE = ON_CREATE
                COMMENT = 'One predicted state per epoch. state_source says which classifier.'
            AS SELECT dog_id, test_num, epoch_ts, state, confidence, state_source
               FROM ML.RULES_STATE;
            CREATE OR REPLACE VIEW ML.V_STATE_PREDICTION AS
                SELECT * FROM ML.STATE_PREDICTION;
            UPDATE REF.PARAMS SET value_num = 1 WHERE key = 'use_rules_classifier';
            err := SQLERRM;
            DELETE FROM ML.MODEL_STATUS;
            INSERT INTO ML.MODEL_STATUS
                SELECT 'RULES (fallback)', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ,
                       :n_train, FALSE, :err;
            msg := 'FALLBACK to RULES classifier — ML.CLASSIFICATION failed: '
                || SQLERRM || ' (recorded in REF.PARAMS.use_rules_classifier=1)';
    END;

    RETURN :msg;
END;
$$;

-- ---------------------------------------------------------------------------
-- THE HONEST NUMBER: evaluation on entirely held-out dogs.
--
-- Computed here rather than taken from SHOW_EVALUATION_METRICS, because that
-- reports the model's internal split, which shares dogs across the fold. This
-- is the number that goes on the Drivers tab in large type.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE ML.SP_EVALUATE_HOLDOUT()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    CREATE OR REPLACE TABLE ML.HOLDOUT_PREDICTIONS AS
    SELECT
        t.dog_id, t.test_num, t.epoch_ts,
        t.state                       AS actual_state,
        pr.state                      AS predicted_state,
        pr.confidence,
        pr.state_source,
        IFF(t.state = pr.state, 1, 0) AS is_correct
    FROM ML.V_TEST t
    JOIN ML.V_STATE_PREDICTION pr
      ON pr.dog_id  = t.dog_id
     AND pr.test_num = t.test_num
     AND pr.epoch_ts = t.epoch_ts;

    -- Confusion matrix, long form. The dashboard pivots it into a heatmap.
    CREATE OR REPLACE TABLE ML.CONFUSION_MATRIX AS
    SELECT
        actual_state,
        predicted_state,
        COUNT(*)                                                              AS n,
        ROUND(100.0 * COUNT(*) /
              NULLIF(SUM(COUNT(*)) OVER (PARTITION BY actual_state), 0), 2)   AS pct_of_actual
    FROM ML.HOLDOUT_PREDICTIONS
    GROUP BY actual_state, predicted_state;

    -- Per-class precision / recall / F1 on unseen dogs.
    CREATE OR REPLACE TABLE ML.CLASS_METRICS AS
    WITH per_class AS (
        SELECT
            s.state,
            SUM(IFF(h.actual_state = s.state AND h.predicted_state = s.state, 1, 0)) AS tp,
            SUM(IFF(h.actual_state <> s.state AND h.predicted_state = s.state, 1, 0)) AS fp,
            SUM(IFF(h.actual_state = s.state AND h.predicted_state <> s.state, 1, 0)) AS fn,
            SUM(IFF(h.actual_state = s.state, 1, 0))                                  AS support
        FROM (SELECT DISTINCT actual_state AS state FROM ML.HOLDOUT_PREDICTIONS) s
        CROSS JOIN ML.HOLDOUT_PREDICTIONS h
        GROUP BY s.state
    )
    SELECT
        state, tp, fp, fn, support,
        ROUND(tp / NULLIF(tp + fp, 0), 4)                                    AS precision,
        ROUND(tp / NULLIF(tp + fn, 0), 4)                                    AS recall,
        ROUND(2.0 * tp / NULLIF(2.0 * tp + fp + fn, 0), 4)                   AS f1
    FROM per_class;

    -- Per-dog accuracy: generalisation is not uniform across individuals, and
    -- the spread is more informative than the mean.
    CREATE OR REPLACE TABLE ML.HOLDOUT_BY_DOG AS
    SELECT
        h.dog_id,
        d.breed,
        COUNT(*)                                     AS epochs,
        ROUND(AVG(h.is_correct), 4)                  AS accuracy
    FROM ML.HOLDOUT_PREDICTIONS h
    LEFT JOIN REF.DOG_INFO d ON d.dog_id = h.dog_id
    GROUP BY h.dog_id, d.breed;

    CREATE OR REPLACE TABLE ML.MODEL_SUMMARY AS
    SELECT
        (SELECT value_str FROM REF.PARAMS WHERE key = 'model_version')       AS model_version,
        (SELECT MAX(state_source) FROM ML.HOLDOUT_PREDICTIONS)               AS classifier,
        (SELECT COUNT(*) FROM ML.V_TRAIN)                                    AS train_epochs,
        (SELECT COUNT(DISTINCT dog_id) FROM ML.V_LABELLED_EPOCHS
          WHERE dog_id NOT IN (SELECT dog_id FROM REF.HOLDOUT_DOGS))         AS train_dogs,
        (SELECT COUNT(*) FROM ML.HOLDOUT_PREDICTIONS)                        AS holdout_epochs,
        (SELECT COUNT(*) FROM REF.HOLDOUT_DOGS)                              AS holdout_dogs,
        (SELECT ROUND(AVG(is_correct), 4) FROM ML.HOLDOUT_PREDICTIONS)       AS holdout_accuracy,
        (SELECT ROUND(AVG(f1), 4) FROM ML.CLASS_METRICS)                     AS macro_f1,
        (SELECT ROUND(SUM(f1 * support) / NULLIF(SUM(support),0), 4)
           FROM ML.CLASS_METRICS)                                            AS weighted_f1,
        'Dog-disjoint. Whole dogs held out, never rows. Row splits leak adjacent '
        || '100 Hz samples across the fold and inflate accuracy by ~20 points.'  AS protocol,
        CURRENT_TIMESTAMP()                                                  AS evaluated_at;

    RETURN 'holdout accuracy: '
        || (SELECT TO_VARCHAR(ROUND(100 * holdout_accuracy, 2)) FROM ML.MODEL_SUMMARY)
        || '% on ' || (SELECT holdout_dogs FROM ML.MODEL_SUMMARY) || ' unseen dogs'
        || ' | macro F1 ' || (SELECT macro_f1 FROM ML.MODEL_SUMMARY);
END;
$$;
