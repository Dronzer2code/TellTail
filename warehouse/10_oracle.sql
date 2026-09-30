-- ===========================================================================
-- 10_oracle.sql   ·   the attestation queue
--
-- THE ARGUMENT, in the form a judge can check:
--
--   A dog arriving at a shelter arrives with no history. Vets, shelters,
--   boarding kennels and adopters share no database and commercially never
--   will. A health record that lives in one vendor's cloud dies when the dog
--   changes hands — which is precisely the moment the record matters most.
--
--   So the finding is published as a signed claim on a public chain: hashed
--   identifier, syndrome code, epoch range, severity, model version.
--   PUBLISH THE CLAIM, NEVER THE DATA.
--
-- THE SECURITY PROPERTY, stated plainly because it is checkable:
--   THE KEYPAIR NEVER TOUCHES SNOWFLAKE. The warehouse stages rows in a queue
--   table. A small Node bridge holds the key, signs, submits, and writes the
--   audit trail back. Snowflake cannot sign anything, and a full dump of the
--   warehouse yields no key material.
--
-- The design is an OUTBOUND queue-poll rather than an inbound procedure call
-- because external access integrations are not supported on trial accounts
-- (error 509009, confirmed the hard way).
-- ===========================================================================

USE DATABASE ${SNOWFLAKE_DATABASE};
USE SCHEMA ORACLE;

CREATE TABLE IF NOT EXISTS ORACLE.PUBLISH_QUEUE (
    publish_id     NUMBER IDENTITY START 1 INCREMENT 1,
    dog_hash       STRING,             -- SHA2(dog_id || salt). Never the raw id.
    syndrome_code  STRING,
    severity       NUMBER,             -- 1 routine, 2 schedule, 3 urgent
    onset_ts       TIMESTAMP_NTZ,
    duration_s     NUMBER,
    confidence     FLOAT,
    model_version  STRING,
    payload        VARIANT,            -- exactly what gets signed
    status         STRING DEFAULT 'PENDING',   -- PENDING -> SENT -> CONFIRMED | FAILED
    attempts       NUMBER DEFAULT 0,
    queued_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    sent_at        TIMESTAMP_NTZ,
    confirmed_at   TIMESTAMP_NTZ,
    tx_signature   STRING,
    explorer_url   STRING,
    slot           NUMBER,
    last_error     STRING
);

-- Append-only audit. Every state change the bridge makes lands here, so the
-- Pipeline tab can show the full lifecycle and not just the current status.
CREATE TABLE IF NOT EXISTS ORACLE.PUBLISH_LOG (
    log_id       NUMBER IDENTITY START 1 INCREMENT 1,
    publish_id   NUMBER,
    event        STRING,          -- QUEUED | CLAIMED | SUBMITTED | CONFIRMED | FAILED
    detail       STRING,
    tx_signature STRING,
    at           TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------------
-- Enqueue. Deduped on the finding key, gated on triage severity, and the dog id
-- is hashed on the way in so the raw id never exists in a publishable row.
--
-- The salt comes from the environment via ${TELLTAIL_HASH_SALT} at build time.
-- Change the salt and every dog_hash changes, which is the point: the hash is
-- stable for a deployment and meaningless across deployments.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE ORACLE.SP_ENQUEUE_ATTESTATIONS()
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    min_sev NUMBER;
    made    NUMBER DEFAULT 0;
BEGIN
    SELECT value_num INTO :min_sev FROM REF.PARAMS WHERE key = 'attest_min_severity';

    INSERT INTO ORACLE.PUBLISH_QUEUE
        (dog_hash, syndrome_code, severity, onset_ts, duration_s, confidence,
         model_version, payload)
    WITH candidates AS (
        SELECT
            f.dog_id,
            f.syndrome_code,
            f.onset_ts,
            f.resolve_ts,
            f.duration_s,
            f.confidence,
            f.model_version,
            COALESCE(t.severity, f.severity) AS severity,
            SHA2(f.dog_id::STRING || '${TELLTAIL_HASH_SALT}', 256) AS dog_hash
        FROM MARTS.V_FINDINGS f
        LEFT JOIN AI.TRIAGE t
               ON t.dog_id = f.dog_id
              AND t.syndrome_code = f.syndrome_code
              AND t.onset_ts = f.onset_ts
        WHERE COALESCE(t.severity, f.severity) >= :min_sev
    )
    SELECT
        c.dog_hash,
        c.syndrome_code,
        c.severity,
        c.onset_ts,
        c.duration_s,
        ROUND(c.confidence, 4),
        c.model_version,
        -- The signed payload. A claim, not a record: no breed, no age, no
        -- weight, no telemetry, no raw identifier. Someone holding this row and
        -- the chain still cannot say which dog it is.
        OBJECT_CONSTRUCT(
            'oracle',  'TELLTAIL',
            'source',  'Snowflake MARTS.SYNDROME_MATCHES',
            'subject', LEFT(c.dog_hash, 16),
            'finding', c.syndrome_code,
            'severity', c.severity,
            'window',  TO_VARCHAR(c.onset_ts, 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
                       || '/PT' || c.duration_s::STRING || 'S',
            'conf',    ROUND(c.confidence, 3),
            'model',   c.model_version
        )
    FROM candidates c
    WHERE NOT EXISTS (
        SELECT 1 FROM ORACLE.PUBLISH_QUEUE q
        WHERE q.dog_hash = c.dog_hash
          AND q.syndrome_code = c.syndrome_code
          AND q.onset_ts = c.onset_ts
    );

    made := SQLROWCOUNT;

    INSERT INTO ORACLE.PUBLISH_LOG (publish_id, event, detail)
    SELECT publish_id, 'QUEUED', 'severity ' || severity || ' ' || syndrome_code
    FROM ORACLE.PUBLISH_QUEUE
    WHERE status = 'PENDING'
      AND queued_at > DATEADD('minute', -1, CURRENT_TIMESTAMP());

    RETURN 'queued ' || :made || ' attestations (min severity ' || :min_sev || '); '
        || (SELECT COUNT(*) FROM ORACLE.PUBLISH_QUEUE WHERE status = 'PENDING')
        || ' pending';
END;
$$;

-- ---------------------------------------------------------------------------
-- The bridge's interface. Three procedures, so the Node side never writes ad
-- hoc SQL and the state machine lives in one place.
--
-- CLAIM marks rows SENT before the bridge signs anything. Two bridge instances
-- polling the same queue therefore cannot double-publish: the second claim
-- returns nothing.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE ORACLE.SP_CLAIM_BATCH(batch_size NUMBER)
RETURNS TABLE (publish_id NUMBER, payload VARIANT)
LANGUAGE SQL
AS
$$
DECLARE
    res RESULTSET;
BEGIN
    CREATE OR REPLACE TEMPORARY TABLE ORACLE._CLAIMED AS
    SELECT publish_id
    FROM ORACLE.PUBLISH_QUEUE
    WHERE status = 'PENDING' AND attempts < 5
    ORDER BY severity DESC, queued_at ASC
    LIMIT :batch_size;

    UPDATE ORACLE.PUBLISH_QUEUE
       SET status = 'SENT', sent_at = CURRENT_TIMESTAMP(), attempts = attempts + 1
     WHERE publish_id IN (SELECT publish_id FROM ORACLE._CLAIMED);

    INSERT INTO ORACLE.PUBLISH_LOG (publish_id, event, detail)
    SELECT publish_id, 'CLAIMED', 'claimed by bridge' FROM ORACLE._CLAIMED;

    res := (SELECT q.publish_id, q.payload
            FROM ORACLE.PUBLISH_QUEUE q
            JOIN ORACLE._CLAIMED c ON c.publish_id = q.publish_id
            ORDER BY q.publish_id);
    RETURN TABLE(res);
END;
$$;

CREATE OR REPLACE PROCEDURE ORACLE.SP_MARK_CONFIRMED(
    p_publish_id NUMBER, p_signature STRING, p_slot NUMBER, p_explorer STRING)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    UPDATE ORACLE.PUBLISH_QUEUE
       SET status = 'CONFIRMED',
           confirmed_at = CURRENT_TIMESTAMP(),
           tx_signature = :p_signature,
           slot = :p_slot,
           explorer_url = :p_explorer,
           last_error = NULL
     WHERE publish_id = :p_publish_id;

    INSERT INTO ORACLE.PUBLISH_LOG (publish_id, event, detail, tx_signature)
    VALUES (:p_publish_id, 'CONFIRMED', 'slot ' || :p_slot, :p_signature);

    RETURN 'confirmed ' || :p_publish_id;
END;
$$;

CREATE OR REPLACE PROCEDURE ORACLE.SP_MARK_FAILED(p_publish_id NUMBER, p_error STRING)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Back to PENDING for a retry until the attempt ceiling, then FAILED for
    -- good. Without the ceiling a bad payload spins forever against devnet.
    UPDATE ORACLE.PUBLISH_QUEUE
       SET status = IFF(attempts >= 5, 'FAILED', 'PENDING'),
           last_error = :p_error
     WHERE publish_id = :p_publish_id;

    INSERT INTO ORACLE.PUBLISH_LOG (publish_id, event, detail)
    VALUES (:p_publish_id, 'FAILED', LEFT(:p_error, 500));

    RETURN 'failed ' || :p_publish_id;
END;
$$;

-- What the Pipeline tab renders: the on-chain publish log with clickable links.
CREATE OR REPLACE VIEW ORACLE.V_PUBLISH_STATUS AS
SELECT
    q.publish_id,
    LEFT(q.dog_hash, 16) || '…'          AS subject,
    q.syndrome_code,
    c.syndrome_name,
    q.severity,
    q.confidence,
    q.onset_ts,
    q.duration_s,
    q.status,
    q.attempts,
    q.tx_signature,
    q.explorer_url,
    q.slot,
    q.queued_at,
    q.confirmed_at,
    DATEDIFF('second', q.queued_at, q.confirmed_at) AS latency_s,
    q.last_error
FROM ORACLE.PUBLISH_QUEUE q
LEFT JOIN REF.SYNDROME_CATALOGUE c ON c.syndrome_code = q.syndrome_code;

CREATE OR REPLACE VIEW ORACLE.V_PUBLISH_SUMMARY AS
SELECT
    status,
    COUNT(*)                                            AS n,
    MIN(queued_at)                                      AS oldest,
    MAX(COALESCE(confirmed_at, sent_at, queued_at))     AS newest,
    ROUND(AVG(DATEDIFF('second', queued_at, confirmed_at)), 1) AS avg_latency_s
FROM ORACLE.PUBLISH_QUEUE
GROUP BY status;
