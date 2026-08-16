#!/usr/bin/env python3
"""
add_country.py
--------------
One-time patch. Adds the source_country column that the matcher forgot to write
into articles_v3.csv, so the dashboard map can fill in. Uses the GDELT lookup
you exported (sourcesbycountry.csv). Does NOT re-run the matcher.

USAGE (put all three files in the same folder as this script):
    articles_v3_enriched.csv   (from 03_enrich_v3.py)
    sourcesbycountry.csv       (Domain, CountryName — the lookup you exported)

    python add_country.py

Overwrites articles_v3_enriched.csv in place, with source_country added.
Then just re-run:  python 04_build_dashboard.py
"""

import pandas as pd

ARTICLES = "articles_v3_enriched.csv"
LOOKUP = "sourcesbycountry.csv"


def main():
    df = pd.read_csv(ARTICLES, low_memory=False)
    look = pd.read_csv(LOOKUP)

    # normalize both sides: lowercase domain, strip whitespace
    look.columns = [c.strip() for c in look.columns]
    look["Domain"] = look["Domain"].astype(str).str.lower().str.strip()
    df["_outlet_key"] = df["outlet"].astype(str).str.lower().str.strip()

    # collapse the lookup to one country per domain (first non-null)
    look = look.dropna(subset=["Domain"]).drop_duplicates("Domain", keep="first")
    m = dict(zip(look["Domain"], look["CountryName"]))

    df["source_country"] = df["_outlet_key"].map(m).fillna("UNKNOWN")
    df = df.drop(columns=["_outlet_key"])
    df.to_csv(ARTICLES, index=False)

    n_known = (df["source_country"] != "UNKNOWN").sum()
    print(f"{len(df):,} articles")
    print(f"  country resolved : {n_known:,} ({n_known/len(df)*100:.1f}%)")
    print(f"  UNKNOWN          : {len(df)-n_known:,}")
    print(f"  distinct countries: {df[df['source_country']!='UNKNOWN']['source_country'].nunique()}")
    print("\ntop countries:")
    print(df[df["source_country"] != "UNKNOWN"]["source_country"]
          .value_counts().head(12).to_string())
    print(f"\nwrote {ARTICLES}. Now re-run: python 04_build_dashboard.py")


if __name__ == "__main__":
    main()
