-- ===========================================================================
-- 00_account_setup.sql   ·   run as ACCOUNTADMIN
--
-- run_sql.py switches to SNOWFLAKE_ADMIN_ROLE for this file only, then drops
-- back to SNOWFLAKE_ROLE for everything after it.
--
-- The timezone line is not optional. If the account is not UTC, TIME_SLICE()
-- buckets epochs against local time, epoch boundaries land in the wrong second,
-- and MATCH_RECOGNIZE matches sequences that never happened.
-- ===========================================================================

ALTER ACCOUNT SET TIMEZONE = 'Etc/UTC';

-- Cortex AI Functions are not natively available in every region. This makes a
-- wrong-region trial survivable instead of fatal.
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

CREATE WAREHOUSE IF NOT EXISTS ${SNOWFLAKE_WAREHOUSE}
    WAREHOUSE_SIZE       = 'XSMALL'
    AUTO_SUSPEND         = 60
    AUTO_RESUME          = TRUE
    INITIALLY_SUSPENDED  = TRUE
    COMMENT              = 'TELLTAIL: ingest, DAG refresh, pattern scan, Streamlit';

CREATE DATABASE IF NOT EXISTS ${SNOWFLAKE_DATABASE}
    COMMENT = 'TELLTAIL — canine syndrome detection via row pattern recognition';

-- Everything below goes to SNOWFLAKE_ROLE, the role every later file and the
-- Streamlit app run as. Granting to a different role leaves 01+ with no
-- warehouse and no database.

-- Cortex AI Functions (AI_COMPLETE / AI_CLASSIFY / AI_AGG) need this role.
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE ${SNOWFLAKE_ROLE};

-- Tasks run as the task owner; EXECUTE TASK is account-level.
GRANT EXECUTE TASK ON ACCOUNT TO ROLE ${SNOWFLAKE_ROLE};
GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE ${SNOWFLAKE_ROLE};

-- The dashboard reads task history and credit burn from ACCOUNT_USAGE.
GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE TO ROLE ${SNOWFLAKE_ROLE};

GRANT USAGE ON WAREHOUSE ${SNOWFLAKE_WAREHOUSE} TO ROLE ${SNOWFLAKE_ROLE};
GRANT ALL ON DATABASE ${SNOWFLAKE_DATABASE} TO ROLE ${SNOWFLAKE_ROLE};

-- --------------------------------------------------------------------------
-- Sanity checks. If either of these fails you have a region problem, and you
-- want to find that out now rather than at hour thirty.
-- --------------------------------------------------------------------------
SELECT CURRENT_REGION() AS region,
       CURRENT_ACCOUNT() AS account,
       CURRENT_VERSION() AS sf_version,
       CURRENT_TIMESTAMP() AS ts_utc;

SELECT SNOWFLAKE.CORTEX.COMPLETE('${CORTEX_MODEL}', 'Reply with exactly: OK') AS cortex_probe;
