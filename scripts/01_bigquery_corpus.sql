-- =============================================================
-- #MeToo keyword-matched GDELT corpus
-- Project: fads-metoo
-- Prereq (one-time): upload metoo_keywords_bq.csv as table
--   fads-metoo.metoocorpus.keywords
--   (BQ Console > your dataset > Create table > Upload > CSV,
--    "Auto detect" schema, header row = 1)
-- =============================================================

-- STEP 0: sanity-check the lookup table column names (tiny scan)
-- SELECT * FROM `gdelt-bq.extra.sourcesbycountry` LIMIT 5;
-- Confirmed columns: Domain, FIPS, CountryName

-- =============================================================
-- STEP 1: build the hits table (ONE expensive scan, run once)
-- Match-level output: one row per (article, keyword) pair
-- =============================================================
CREATE OR REPLACE TABLE `fads-metoo.metoocorpus.keyword_hits` AS
WITH kw AS (
  SELECT
    keyword_country,
    keyword_raw,
    keyword,
    kw_match,
    match_mode,
    regex_pattern,
    stance_original,
    stance_primary
  FROM `fads-metoo.metoocorpus.keywords`
),
gkg AS (
  SELECT
    DocumentIdentifier AS url,
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS pub_ts,
    LOWER(V2SourceCommonName) AS domain,
    REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.+?)</PAGE_TITLE>') AS title
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2017-10-01')
    AND _PARTITIONTIME <  TIMESTAMP('2020-01-01')
  -- dedupe GKG re-crawls of the same URL, keep earliest
  QUALIFY ROW_NUMBER() OVER (PARTITION BY DocumentIdentifier ORDER BY DATE) = 1
),
matched AS (
  SELECT
    g.url,
    g.pub_ts,
    g.domain,
    g.title,
    k.keyword,
    k.keyword_raw,
    k.keyword_country,
    k.stance_original,
    k.stance_primary
  FROM gkg g
  JOIN kw k
    ON (
      -- Latin-script terms: word-boundary regex, avoids
      -- e.g. 'viol' matching 'violence'
      (k.match_mode = 'word'
        AND REGEXP_CONTAINS(
              LOWER(CONCAT(IFNULL(g.title, ''), ' ', g.url)),
              k.regex_pattern))
      OR
      -- Non-Latin scripts (CJK, Arabic, Devanagari, etc.):
      -- plain substring, no word boundaries in these scripts
      (k.match_mode = 'substr'
        AND STRPOS(
              CONCAT(IFNULL(g.title, ''), ' ', g.url),
              k.kw_match) > 0)
    )
)
SELECT
  m.*,
  -- article source country via GDELT's domain lookup;
  -- COALESCE keeps .com outlets instead of dropping them
  COALESCE(s.CountryName, 'UNKNOWN') AS source_country
FROM matched m
LEFT JOIN `gdelt-bq.extra.sourcesbycountry` s
  ON m.domain = LOWER(s.Domain);

-- =============================================================
-- STEP 2: the table you asked for
-- One row per article: keywords it contains, stance per keyword,
-- article source country, outlet
-- (cheap, re-run freely against the hits table)
-- =============================================================
SELECT
  url,
  ANY_VALUE(title)          AS title,
  ANY_VALUE(pub_ts)         AS published,
  ANY_VALUE(domain)         AS outlet,
  ANY_VALUE(source_country) AS source_country,
  STRING_AGG(DISTINCT keyword, ' | ' ORDER BY keyword)  AS keywords_matched,
  STRING_AGG(DISTINCT CONCAT(keyword, ' [', stance_primary, ']'),
             ' | ' ORDER BY CONCAT(keyword, ' [', stance_primary, ']'))
                            AS keyword_stances,
  COUNT(DISTINCT keyword)   AS n_keywords,
  COUNTIF(stance_primary = 'Support')  AS n_support,
  COUNTIF(stance_primary = 'Backlash') AS n_backlash
FROM `fads-metoo.metoocorpus.keyword_hits`
GROUP BY url
ORDER BY n_keywords DESC;

-- =============================================================
-- Optional: match-level view (one row per article-keyword pair,
-- includes the keyword's own country from the xlsx)
-- =============================================================
-- SELECT url, title, pub_ts, domain AS outlet, source_country,
--        keyword, keyword_country, stance_original, stance_primary
-- FROM `fads-metoo.metoocorpus.keyword_hits`
-- ORDER BY pub_ts;
