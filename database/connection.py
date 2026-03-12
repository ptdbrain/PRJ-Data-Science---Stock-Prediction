"""
Kết nối database dùng chung cho toàn bộ project.
═════════════════════════════════════════════════
Mọi file cần đọc/ghi DB đều import từ đây.

Cách dùng:
    from database.connection import get_connection, read_table, write_table

    # Cách 1: Đọc/ghi trực tiếp
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM raw_prices", conn)
    conn.close()

    # Cách 2: Helper functions
    df = read_table("raw_prices")
    write_table(df, "clean_prices")
"""
import sqlite3
import pandas as pd
from config.settings import DB_PATH


def get_connection() -> sqlite3.Connection:
    """
    Trả về connection tới SQLite database.
    Tự tạo thư mục nếu chưa có.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")     # Cho phép đọc/ghi đồng thời
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def read_table(table_name: str) -> pd.DataFrame:
    """
    Đọc toàn bộ table thành DataFrame.

    Ví dụ:
        df = read_table("raw_prices")
    """
    conn = get_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        return df
    finally:
        conn.close()


def write_table(df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
    """
    Ghi DataFrame vào table.

    Args:
        df: DataFrame cần ghi
        table_name: tên table
        if_exists: 'replace' (xoá cũ ghi mới) hoặc 'append' (thêm vào)

    Ví dụ:
        write_table(df_clean, "clean_prices")
    """
    conn = get_connection()
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.commit()
    finally:
        conn.close()


def table_exists(table_name: str) -> bool:
    """Kiểm tra table có tồn tại không."""
    conn = get_connection()
    try:
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        ).fetchone()
        return result is not None
    finally:
        conn.close()


def table_row_count(table_name: str) -> int:
    """Đếm số dòng trong table."""
    conn = get_connection()
    try:
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0]
    finally:
        conn.close()
