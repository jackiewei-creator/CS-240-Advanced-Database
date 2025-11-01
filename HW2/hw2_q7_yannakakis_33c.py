#---------------------------check the data----------------------------------
from __future__ import annotations
from pathlib import Path
import csv
from dataclasses import dataclass
from pathlib import Path
import argparse
import duckdb as ddb
import pandas as pd
TABLES = [
    "company_name",
    "info_type",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info_idx",
    "movie_link",
    "title",
]

def normalize(col: str) -> str:
    c = (col or "").strip().replace("\ufeff", "").lower()
    if "." in c:
        c = c.split(".")[-1].strip()
    return c

def read_header(p: Path):
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    print(f"[INFO] data_dir = {data_dir}\n")
    for t in TABLES:
        p = data_dir / f"{t}.csv"
        if not p.exists():
            print(f"[MISSING] {p}")
            continue
        raw = read_header(p)
        norm = [normalize(c) for c in raw]
        print(f"=== {t}.csv ===")
        print(f"Raw columns : {raw}")
        print(f"Normalized  : {norm}\n")

if __name__ == "__main__":
    main()








#------------------Query 33c with Yannakakis + SQL cross-check-----------------

"""
HW2 Q7 — Implement query 33c (Join Order Benchmark) with Yannakakis.
- Robust CSV loading (headerless & ragged rows) via DuckDB read_csv_auto
- Build SQL baseline (original query) using DuckDB views
- Build a join tree and run Yannakakis (semi-join reduce + join)
- Include all predicates; compare with SQL result

Usage:
  python3 src/hw2_q7_yannakakis_33c.py --data-dir "/path/to/data"
"""

# -------------------------------
# 0）Utils
# -------------------------------

def must_exist(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

def info(msg: str) -> None:
    print(f"[INFO] {msg}")


# -------------------------------
# 1) Create headerless CSV views for SQL baseline
# -------------------------------

def create_views_for_sql(con: ddb.DuckDBPyConnection, data_dir: Path) -> None:
    """
    Create DuckDB SQL views for the 8 base tables using robust read_csv settings.
    We explicitly set delimiter/quote/escape and turn on lenient options so that
    JOB's messy CSVs can be parsed without dialect sniffing.
    """
    files = {
        # name -> (filename, columns in the file *in order*)
        "company_name": ("company_name.csv",
                         ["id","name","country_code","imdb_id",
                          "name_pcode_nf","name_pcode_sf","md5sum"]),
        "info_type": ("info_type.csv", ["id","info"]),
        "kind_type": ("kind_type.csv", ["id","kind"]),
        "link_type": ("link_type.csv", ["id","link"]),
        "movie_companies": ("movie_companies.csv",
                            # the CSV order in JOB is: movie_id, company_id, company_type_id, note, nr_order
                            ["movie_id","company_id","company_type_id","note","nr_order"]),
        "movie_info_idx": ("movie_info_idx.csv",
                           # CSV order: movie_id, info_type_id, info, note, info_index
                           ["movie_id","info_type_id","info","note","info_index"]),
        "movie_link": ("movie_link.csv", ["id","movie_id","linked_movie_id","link_type_id"]),
        "title": ("title.csv",
                  # CSV order in JOB:
                  ["id","title","imdb_index","kind_id","production_year",
                   "imdb_id","phonetic_code","episode_of_id","season_nr",
                   "episode_nr","series_years","md5sum"]),
    }

    for vname, (fname, cols) in files.items():
        path = (data_dir / fname).as_posix()
        path_literal = path.replace("'", "''")

        # Build DuckDB COLUMNS={} dict (all VARCHAR to be safe)
        cols_map = ", ".join([f"'{c}': 'VARCHAR'" for c in cols])

        # NOTE: use read_csv (NOT read_csv_auto) with explicit, lenient options
        sql = f"""
            CREATE OR REPLACE VIEW {vname} AS
            SELECT {", ".join(cols)}
            FROM read_csv(
                '{path_literal}',
                delim=',',
                quote='"',
                escape='"',
                header=FALSE,
                columns={{ {cols_map} }},
                sample_size=-1,
                auto_detect=FALSE,
                ignore_errors=TRUE,
                null_padding=TRUE,
                max_line_size=10000000
            );
        """
        con.execute(sql)


# -------------------------------
# 2) SQL baseline for 33c
# -------------------------------

JOB_33C_SQL = r"""
SELECT
  MIN(cn1.name) AS first_company,
  MIN(cn2.name) AS second_company,
  MIN(mi_idx1.info) AS first_rating,
  MIN(mi_idx2.info) AS second_rating,
  MIN(t1.title) AS first_movie,
  MIN(t2.title) AS second_movie
FROM company_name AS cn1,
     company_name AS cn2,
     info_type AS it1,
     info_type AS it2,
     kind_type AS kt1,
     kind_type AS kt2,
     link_type AS lt,
     movie_companies AS mc1,
     movie_companies AS mc2,
     movie_info_idx AS mi_idx1,
     movie_info_idx AS mi_idx2,
     movie_link AS ml,
     title AS t1,
     title AS t2
WHERE cn1.country_code <> '[us]'
  AND it1.info = 'rating'
  AND it2.info = 'rating'
  AND kt1.kind IN ('tv series','episode')
  AND kt2.kind IN ('tv series','episode')
  AND lt.link IN ('sequel','follows','followed by')
  AND TRY_CAST(mi_idx2.info AS DOUBLE) < 3.5
  AND TRY_CAST(t2.production_year AS INT) BETWEEN 2000 AND 2010
  AND lt.id = ml.link_type_id
  AND t1.id = ml.movie_id
  AND t2.id = ml.linked_movie_id
  AND it1.id = mi_idx1.info_type_id
  AND t1.id = mi_idx1.movie_id
  AND kt1.id = t1.kind_id
  AND cn1.id = mc1.company_id
  AND t1.id = mc1.movie_id
  AND ml.movie_id = mi_idx1.movie_id
  AND ml.movie_id = mc1.movie_id
  AND mi_idx1.movie_id = mc1.movie_id
  AND it2.id = mi_idx2.info_type_id
  AND t2.id = mi_idx2.movie_id
  AND kt2.id = t2.kind_id
  AND cn2.id = mc2.company_id
  AND t2.id = mc2.movie_id
  AND ml.linked_movie_id = mi_idx2.movie_id
  AND ml.linked_movie_id = mc2.movie_id
  AND mi_idx2.movie_id = mc2.movie_id;
"""

def sql_baseline(con: ddb.DuckDBPyConnection, data_dir: Path) -> pd.DataFrame:
    create_views_for_sql(con, data_dir)
    return con.execute(JOB_33C_SQL).df()


# -------------------------------
# 3) Robust CSV -> pandas (for Yannakakis)
# -------------------------------

def read_noheader_csv_duck(con: ddb.DuckDBPyConnection, path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Load a JOB CSV (no header) robustly using DuckDB read_csv with explicit options.
    - No prepared params (inline the path)
    - columns mapping must be {'col':'VARCHAR', ...}
    - lenient parsing options enabled for messy rows
    """
    # 1) inline path (escape single quotes)
    path_literal = path.as_posix().replace("'", "''")
    # 2) build columns mapping dict literal for DuckDB
    cols_map = ", ".join([f"'{c}': 'VARCHAR'" for c in columns])
    # 3) compose SQL (note header=FALSE; auto_detect=FALSE; ignore_errors=true)
    sql = f"""
        SELECT {", ".join(columns)}
        FROM read_csv(
            '{path_literal}',
            delim=',',
            quote='"',
            escape='"',
            header=FALSE,
            columns={{ {cols_map} }},
            sample_size=-1,
            auto_detect=FALSE,
            ignore_errors=TRUE,
            null_padding=TRUE,
            max_line_size=10000000
        );
    """
    return con.execute(sql).df()

@dataclass
class BaseTables:
    company_name: pd.DataFrame
    info_type: pd.DataFrame
    kind_type: pd.DataFrame
    link_type: pd.DataFrame
    movie_companies: pd.DataFrame
    movie_info_idx: pd.DataFrame
    movie_link: pd.DataFrame
    title: pd.DataFrame

def load_all_base_tables(con: ddb.DuckDBPyConnection, data_dir: Path) -> BaseTables:
    # ensure presence
    for fn in ["company_name.csv","info_type.csv","kind_type.csv","link_type.csv",
               "movie_companies.csv","movie_info_idx.csv",
               "movie_link.csv","title.csv"]:
        must_exist(data_dir / fn)

    company_name = read_noheader_csv_duck(
        con, data_dir / "company_name.csv",
        ["id","name","country_code","imdb_id",
         "name_pcode_nf","name_pcode_sf","md5sum"]
    )[["id","name","country_code"]]

    info_type = read_noheader_csv_duck(
        con, data_dir / "info_type.csv",
        ["id","info"]
    )[["id","info"]]

    kind_type = read_noheader_csv_duck(
        con, data_dir / "kind_type.csv",
        ["id","kind"]
    )[["id","kind"]]

    link_type = read_noheader_csv_duck(
        con, data_dir / "link_type.csv",
        ["id","link"]
    )[["id","link"]]

    movie_companies = read_noheader_csv_duck(
        con, data_dir / "movie_companies.csv",
        ["movie_id","company_id","company_type_id","note","nr_order"]
    )[["movie_id","company_id"]]

    movie_info_idx = read_noheader_csv_duck(
        con, data_dir / "movie_info_idx.csv",
        ["movie_id","info_type_id","info","note","info_index"]
    )[["movie_id","info_type_id","info"]]

    movie_link = read_noheader_csv_duck(
        con, data_dir / "movie_link.csv",
        ["id","movie_id","linked_movie_id","link_type_id"]
    )[["movie_id","linked_movie_id","link_type_id"]]

    title = read_noheader_csv_duck(
        con, data_dir / "title.csv",
        ["id","title","imdb_index","kind_id","production_year",
         "imdb_id","phonetic_code","episode_of_id","season_nr",
         "episode_nr","series_years","md5sum"]
    )[["id","title","kind_id","production_year"]]

    return BaseTables(company_name, info_type, kind_type, link_type,
                      movie_companies, movie_info_idx, movie_link, title)


# -------------------------------
# 4) Yannakakis — build filtered branches, semijoin reduce, final join
# -------------------------------

def to_numeric_safe(s: pd.Series, kind: str):
    if kind == "int":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    else:
        return pd.to_numeric(s, errors="coerce")

def build_filtered_relations(base: BaseTables) -> dict[str, pd.DataFrame]:
    """
    Build the per-branch relations with all predicates applied.
    Column names follow the CQ variables:
      left:  m1, k1, it1, c1, lt
      right: m2, k2, it2, c2, lt
    Keep extra columns needed for aggregation (company names, titles, ratings).
    """
    # Domain filters
    it_ids = base.info_type[base.info_type["info"] == "rating"][["id"]].rename(columns={"id":"it"})
    kt_ids = base.kind_type[base.kind_type["kind"].isin(["tv series","episode"])][["id"]].rename(columns={"id":"k"})
    lt_ids = base.link_type[base.link_type["link"].isin(["sequel","follows","followed by"])][["id"]].rename(columns={"id":"lt"})

    # movie_link (root) with lt constraint
    ml = base.movie_link.rename(columns={"movie_id":"m1","linked_movie_id":"m2","link_type_id":"lt"}) \
                        .merge(lt_ids, on="lt", how="inner")[["m1","m2","lt"]]

    # Left branch
    t1 = base.title.rename(columns={"id":"m1","kind_id":"k1","title":"title1","production_year":"year1"})
    t1["k1"] = t1["k1"]
    t1 = t1.merge(kt_ids.rename(columns={"k":"k1"}), on="k1", how="inner")[["m1","k1","title1"]]

    mi1 = base.movie_info_idx.rename(columns={"movie_id":"m1","info_type_id":"it1","info":"info1"})
    mi1 = mi1.merge(it_ids.rename(columns={"it":"it1"}), on="it1", how="inner")[["m1","it1","info1"]]

    mc1 = base.movie_companies.rename(columns={"movie_id":"m1","company_id":"c1"})[["m1","c1"]]
    cn1 = base.company_name[base.company_name["country_code"] != "[us]"] \
            .rename(columns={"id":"c1","name":"name1"})[["c1","name1"]]

    # Right branch (year and rating thresholds)
    t2 = base.title.rename(columns={"id":"m2","kind_id":"k2","title":"title2","production_year":"year2"})
    year2 = to_numeric_safe(t2["year2"], "int")
    t2 = t2[(year2 >= 2000) & (year2 <= 2010)]
    t2 = t2.merge(kt_ids.rename(columns={"k":"k2"}), on="k2", how="inner")[["m2","k2","title2"]]

    mi2 = base.movie_info_idx.rename(columns={"movie_id":"m2","info_type_id":"it2","info":"info2"})
    # cast rating to float and keep info2 for aggregation
    rating2 = to_numeric_safe(mi2["info2"], "float")
    mi2 = mi2.assign(_r2=rating2).merge(it_ids.rename(columns={"it":"it2"}), on="it2", how="inner")
    mi2 = mi2[mi2["_r2"] < 3.5][["m2","it2","info2"]]

    mc2 = base.movie_companies.rename(columns={"movie_id":"m2","company_id":"c2"})[["m2","c2"]]
    cn2 = base.company_name.rename(columns={"id":"c2","name":"name2"})[["c2","name2"]]

    # Collect
    return {
        # root
        "ml": ml,  # (m1,m2,lt)

        # left
        "t1": t1,          # (m1,k1,title1)
        "mi1": mi1,        # (m1,it1,info1)
        "mc1": mc1,        # (m1,c1)
        "cn1": cn1,        # (c1,name1)

        # right
        "t2": t2,          # (m2,k2,title2)
        "mi2": mi2,        # (m2,it2,info2)
        "mc2": mc2,        # (m2,c2)
        "cn2": cn2,        # (c2,name2)
    }


def semijoin_keep(df: pd.DataFrame, key: str, allowed: pd.Series) -> pd.DataFrame:
    """Keep rows whose df[key] is in allowed (semi-join reduction)."""
    return df[df[key].isin(pd.unique(allowed))]

def yannakakis_33c(rel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Yannakakis on the join tree:

               ml(m1,m2,lt)
               /          \
     (t1,mi1,mc1,cn1)   (t2,mi2,mc2,cn2)

    Two-phase:
      - bottom-up: compute surviving m1, m2 by intersecting leaf constraints and ml connectivity
      - top-down join: join ml with both branches and aggregate
    """
    ml = rel["ml"]

    # ---- bottom-up: left ----
    m1_from_t1  = rel["t1"]["m1"]
    m1_from_mi1 = rel["mi1"]["m1"]
    # mc1 survives only if company exists and passes country filter
    mc1_kept = rel["mc1"].merge(rel["cn1"], on="c1", how="inner")[["m1","c1"]]
    m1_from_mc1 = mc1_kept["m1"]

    # must also be present in ml on the left
    m1_from_ml  = ml["m1"]

    # intersection
    m1_keep = set(pd.unique(m1_from_t1)) & set(pd.unique(m1_from_mi1)) \
              & set(pd.unique(m1_from_mc1)) & set(pd.unique(m1_from_ml))

    # ---- bottom-up: right ----
    m2_from_t2  = rel["t2"]["m2"]
    m2_from_mi2 = rel["mi2"]["m2"]
    mc2_kept = rel["mc2"].merge(rel["cn2"], on="c2", how="inner")[["m2","c2"]]
    m2_from_mc2 = mc2_kept["m2"]
    m2_from_ml  = ml["m2"]

    m2_keep = set(pd.unique(m2_from_t2)) & set(pd.unique(m2_from_mi2)) \
              & set(pd.unique(m2_from_mc2)) & set(pd.unique(m2_from_ml))

    # prune ml accordingly
    ml_red = ml[ml["m1"].isin(m1_keep) & ml["m2"].isin(m2_keep)][["m1","m2","lt"]]

    # ---- top-down: materialize minimal join (only necessary cols kept) ----
    # Left branch
    left = ml_red.merge(rel["t1"][["m1","title1"]], on="m1", how="inner")
    left = left.merge(rel["mi1"][["m1","info1"]], on="m1", how="inner")
    left = left.merge(mc1_kept, on="m1", how="inner")           # (m1,c1)
    left = left.merge(rel["cn1"][["c1","name1"]], on="c1", how="inner")

    # Right branch
    right = ml_red.merge(rel["t2"][["m2","title2"]], on="m2", how="inner")
    right = right.merge(rel["mi2"][["m2","info2"]], on="m2", how="inner")
    right = right.merge(mc2_kept, on="m2", how="inner")         # (m2,c2)
    right = right.merge(rel["cn2"][["c2","name2"]], on="c2", how="inner")

    # Join left and right via (m1,m2,lt)
    # (They already both contain (m1,m2,lt) from ml_red)
    out = left.merge(right, on=["m1","m2","lt"], how="inner")

    # We only need the six output attributes for MIN aggregation:
    out = out[["name1","name2","info1","info2","title1","title2"]]
    return out


def run_yannakakis(con: ddb.DuckDBPyConnection, data_dir: Path) -> pd.DataFrame:
    base = load_all_base_tables(con, data_dir)
    rel = build_filtered_relations(base)
    joined = yannakakis_33c(rel)
    # Aggregate to match SQL output
    if joined.empty:
        # produce an empty single row with NULLs like SQL MIN on empty -> NULL
        return pd.DataFrame([{
            "first_company": None, "second_company": None,
            "first_rating": None,  "second_rating": None,
            "first_movie":  None,  "second_movie":  None,
        }])
    agg = {
        "name1": "min",
        "name2": "min",
        "info1": "min",
        "info2": "min",
        "title1": "min",
        "title2": "min",
    }
    res = joined.agg(agg).to_frame().T
    res.columns = ["first_company","second_company","first_rating",
                   "second_rating","first_movie","second_movie"]
    # Return one-row pandas df to compare with SQL df
    return res.reset_index(drop=True)


# -------------------------------
# 5) Main
# -------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Path to the JOB data directory (contains *.csv). "
                         "If omitted, defaults to ../data relative to script.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    # Default ../data
    if args.data_dir is None:
        data_dir = (Path(__file__).resolve().parent.parent / "data").resolve()
    else:
        data_dir = Path(args.data_dir).resolve()
    info(f"Using data_dir = {data_dir}")

    con = ddb.connect(database=":memory:")

    # SQL baseline
    sql_df = sql_baseline(con, data_dir)
    print("\n[SQL baseline output]")
    print(sql_df)

    # Yannakakis
    yan_df = run_yannakakis(con, data_dir)
    print("\n[Yannakakis output]")
    print(yan_df)

    # Compare (string-wise)
    same = sql_df.astype(str).equals(yan_df.astype(str))
    print(f"\n[Compare] Yannakakis matches SQL baseline? {same}")

if __name__ == "__main__":
    main()