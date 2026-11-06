"""The data warehouse behind one small interface.

- DuckDB when it is installed (the default for the demo and CI),
- SQLite as a fallback (no extra package),
- Snowflake when LW_ENGINE=snowflake (SNOWFLAKE_* environment variables; key-pair login with
  SNOWFLAKE_PRIVATE_KEY, or a password for a trial account). Give the app a role with SELECT only.

All queries are plain SQL with `?` parameters, so the same text runs on all three. Readers open the
ledger read-only: the agent's tools cannot change it even by mistake.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd


def engine() -> str:
    want = os.getenv("LW_ENGINE", "auto").lower()
    if want in ("sqlite", "duckdb", "snowflake"):
        return want
    try:
        import duckdb  # noqa: F401

        return "duckdb"
    except ImportError:
        return "sqlite"


def ledger_exists(data_dir: Path) -> bool:
    if engine() != "snowflake":
        return ledger_path(data_dir).exists()
    try:
        with Warehouse(data_dir) as wh:
            return int(wh.df("SELECT COUNT(*) AS n FROM journal_entries")["n"][0]) > 0
    except Exception:  # no table yet
        return False


def ledger_path(data_dir: Path) -> Path:
    return Path(data_dir) / f"ledger.{'duckdb' if engine() == 'duckdb' else 'sqlite'}"


class Warehouse:
    def __init__(self, data_dir: Path, read_only: bool = True):
        self.kind = engine()
        self.read_only = read_only
        path = ledger_path(data_dir)
        if self.kind == "duckdb":
            import duckdb

            self.con = duckdb.connect(str(path), read_only=read_only)
        elif self.kind == "snowflake":
            import snowflake.connector

            snowflake.connector.paramstyle = "qmark"
            auth: dict = {"password": os.getenv("SNOWFLAKE_PASSWORD")}
            if os.getenv("SNOWFLAKE_PRIVATE_KEY"):  # PEM text from Secrets Manager -> DER bytes for the connector
                from cryptography.hazmat.primitives import serialization

                key = serialization.load_pem_private_key(os.environ["SNOWFLAKE_PRIVATE_KEY"].encode(), password=None)
                auth = {"private_key": key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption())}
            self.con = snowflake.connector.connect(
                account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
                warehouse=os.environ["SNOWFLAKE_WAREHOUSE"], database=os.environ["SNOWFLAKE_DATABASE"],
                schema=os.getenv("SNOWFLAKE_SCHEMA", "LEDGERWATCH"), role=os.getenv("SNOWFLAKE_ROLE"), **auth)
        else:
            if read_only:
                self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10, check_same_thread=False)
            else:
                self.con = sqlite3.connect(path, timeout=10, check_same_thread=False)

    def df(self, sql: str, params: tuple | list = ()) -> pd.DataFrame:
        if self.kind == "duckdb":
            return self.con.execute(sql, list(params)).df()
        if self.kind == "snowflake":
            cur = self.con.cursor()
            cur.execute(sql, list(params))
            out = cur.fetch_pandas_all()
            out.columns = [c.lower() for c in out.columns]
            return out
        return pd.read_sql_query(sql, self.con, params=list(params))

    def write(self, name: str, frame: pd.DataFrame) -> None:
        """Replace a table with a DataFrame (building the simulated company only)."""
        if self.read_only:
            raise PermissionError("read-only connection")
        if self.kind == "duckdb":
            self.con.register("_frame", frame)
            self.con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _frame")
            self.con.unregister("_frame")
        elif self.kind == "snowflake":
            from snowflake.connector.pandas_tools import write_pandas

            write_pandas(self.con, frame, name.upper(), auto_create_table=True, overwrite=True, quote_identifiers=False)
        else:
            frame.to_sql(name, self.con, if_exists="replace", index=False, chunksize=20_000)
            self.con.commit()

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
