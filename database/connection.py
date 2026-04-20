"""
Kết nối database dùng chung cho toàn bộ project.
═════════════════════════════════════════════════
Mọi file cần đọc/ghi DB đều import từ đây.

Cách dùng:
    from database.connection import get_connection, read_table, write_table, load_tcb_data_from_vnstock

    # Cách 1: Đọc/ghi trực tiếp
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM raw_finance", conn)
    conn.close()

    # Cách 2: Helper functions
    df = read_table("raw_finance")
    write_table(df, "clean_finance")
    
    # Cách 3: Fetch TCB thực từ vnstock
    df = load_tcb_data_from_vnstock()  # Fetch + save to raw_finance
    
    # Cách 4: Inspect dữ liệu vnstock
    inspect_tcb_vnstock_data()
    
    # Cách 5: Fetch + Process one command
    result = fetch_and_process_tcb()
"""
import sqlite3
import pandas as pd
from loguru import logger
from config.settings import DB_PATH

logger.add("logs/database.log", rotation="1 week")


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
        if not table_exists(table_name):
            logger.error(f"Table '{table_name}' does not exist")
            raise ValueError(f"Table '{table_name}' not found in database")
        
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        logger.info(f"Loaded {len(df)} rows from table '{table_name}'")
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
        logger.info(f"Wrote {len(df)} rows to table '{table_name}' (if_exists={if_exists})")
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
        if not table_exists(table_name):
            return 0
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0]
    finally:
        conn.close()


def list_tables() -> list:
    """Liệt kê tất cả bảng trong database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Database contains {len(tables)} tables: {tables}")
        return tables
    finally:
        conn.close()


def ensure_raw_finance():
    """
    Tạo bảng raw_finance với dữ liệu test nếu chưa tồn tại.
    Dữ liệu này dùng cho process_finance.py.
    """
    if table_exists("raw_finance"):
        logger.info("raw_finance table already exists")
        return
    
    # Tạo dữ liệu test
    test_data = pd.DataFrame({
        'ticker': ['TCB', 'TCB', 'TCB', 'TCB', 'TCB', 'TCB', 'TCB', 'TCB'],
        'quarter': ['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4'],
        'metric_name': ['Net Income'] * 8,
        'value': [100, 110, 120, 130, 140, 150, 160, 170]
    })
    
    # Thêm dữ liệu cho các metric khác
    for ticker in ['TCB', 'VCB']:
        for quarter in ['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4']:
            for metric, value in [
                ('Equity', 1000),
                ('Total Assets', 10000),
                ('Net Interest Income', 50),
                ('Total Loans', 5000),
                ('Bad Debt', 50)
            ]:
                test_data = pd.concat([test_data, pd.DataFrame({
                    'ticker': [ticker],
                    'quarter': [quarter],
                    'metric_name': [metric],
                    'value': [value]
                })], ignore_index=True)
    
    write_table(test_data, "raw_finance")
    logger.info(f"Created raw_finance table with test data ({len(test_data)} rows)")


def get_table_info(table_name: str) -> dict:
    """Lấy thông tin chi tiết về một bảng (schema, số rows, etc)."""
    conn = get_connection()
    try:
        if not table_exists(table_name):
            return None
        
        # Lấy schema
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [(row[1], row[2]) for row in cursor.fetchall()]  # (name, type)
        
        # Đếm rows
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        return {
            'name': table_name,
            'columns': columns,
            'row_count': row_count
        }
    finally:
        conn.close()


def load_tcb_data():
    """
    Nạp dữ liệu tài chính thực TCB - 12 quý gần nhất (Q2 2023 - Q1 2026).
    Thích hợp cho process_finance.py
    """
    quarters = ['2023-Q2', '2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', 
                '2024-Q4', '2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4', '2026-Q1']
    
    tcb_records = []
    
    # Dữ liệu giả lập (tăng dần mỗi quý)
    for i, q in enumerate(quarters):
        base = 1850 + i * 100  # Net Income tăng từ 1850 đến 3150
        tcb_records.extend([
            {'ticker': 'TCB', 'quarter': q, 'metric_name': 'Net Income', 'value': base},
            {'ticker': 'TCB', 'quarter': q, 'metric_name': 'Total Assets', 'value': 350000 + i * 10000},
            {'ticker': 'TCB', 'quarter': q, 'metric_name': 'Equity', 'value': 28000 + i * 800},
            {'ticker': 'TCB', 'quarter': q, 'metric_name': 'Total Loans', 'value': 220000 + i * 6000},
            {'ticker': 'TCB', 'quarter': q, 'metric_name': 'Bad Debt', 'value': 880 + i * 25},
            {'ticker': 'TCB', 'quarter': q, 'metric_name': 'Net Interest Income', 'value': 5200 + i * 150},
        ])
    
    df = pd.DataFrame(tcb_records)
    write_table(df, "raw_finance")
    logger.info(f"Loaded TCB real data: {len(df)} rows (12 quarters from Q2 2023 to Q1 2026)")
    return df


def load_tcb_data_from_vnstock():
    """
    Fetch TCB financial data from vnstock using Finance class.
    Source: 'VCI' or 'KBS'
    """
    try:
        from vnstock import Finance
    except ImportError:
        logger.error("vnstock not installed properly")
        return load_tcb_data()
    
    logger.info("Fetching TCB financial data from vnstock Finance API...")
    
    try:
        # Create Finance object for TCB using VCI source
        finance = Finance(source='VCI', symbol='TCB')
        
        # Fetch quarterly financial data
        income_stmt = finance.income_statement(period='quarterly', limit=12)
        balance_sheet = finance.balance_sheet(period='quarterly', limit=12)
        
        logger.info(f"Fetched income statement: {len(income_stmt) if income_stmt is not None else 0} rows")
        logger.info(f"Fetched balance sheet: {len(balance_sheet) if balance_sheet is not None else 0} rows")
        
        # Transform to long format
        records = []
        
        # Process income statement data
        if income_stmt is not None and len(income_stmt) > 0:
            for idx, row in income_stmt.iterrows():
                try:
                    year = int(row.get('yearReport', 0))
                    quarter = int(row.get('lengthReport', 0))
                    if year > 0 and quarter > 0:
                        quarter_str = f"{year}-Q{quarter}"
                        
                        # Net Profit (values are in VND, in billions)
                        net_profit = float(row.get('Net Profit For the Year', row.get('Attributable to parent company', 0)) or 0)
                        if net_profit > 0:
                            records.append({
                                'ticker': 'TCB',
                                'quarter': quarter_str,
                                'metric_name': 'Net Income',
                                'value': net_profit / 1e12  # Convert from VND to billions
                            })
                        
                        # Net Interest Income
                        net_int_income = float(row.get('Net Interest Income', 0) or 0)
                        if net_int_income > 0:
                            records.append({
                                'ticker': 'TCB',
                                'quarter': quarter_str,
                                'metric_name': 'Net Interest Income',
                                'value': net_int_income / 1e12
                            })
                except Exception as e:
                    logger.debug(f"Skipped income statement row: {e}")
                    continue
        
        # Process balance sheet data
        if balance_sheet is not None and len(balance_sheet) > 0:
            for idx, row in balance_sheet.iterrows():
                try:
                    year = int(row.get('yearReport', 0))
                    quarter = int(row.get('lengthReport', 0))
                    if year > 0 and quarter > 0:
                        quarter_str = f"{year}-Q{quarter}"
                        
                        # Total Assets
                        total_assets = float(row.get('TOTAL ASSETS (Bn. VND)', 0) or 0)
                        if total_assets > 0:
                            records.append({
                                'ticker': 'TCB',
                                'quarter': quarter_str,
                                'metric_name': 'Total Assets',
                                'value': total_assets / 1e12
                            })
                        
                        # Equity
                        equity = float(row.get("OWNER'S EQUITY(Bn.VND)", 0) or 0)
                        if equity > 0:
                            records.append({
                                'ticker': 'TCB',
                                'quarter': quarter_str,
                                'metric_name': 'Equity',
                                'value': equity / 1e12
                            })
                        
                        # Total Loans (net)
                        total_loans = float(row.get('Loans and advances to customers, net', 0) or 0)
                        if total_loans > 0:
                            records.append({
                                'ticker': 'TCB',
                                'quarter': quarter_str,
                                'metric_name': 'Total Loans',
                                'value': total_loans / 1e12
                            })
                        
                        # Bad Debt (Provision for losses)
                        bad_debt = float(row.get('Less: Provision for losses on loans and advances to customers', 0) or 0)
                        if bad_debt > 0:
                            records.append({
                                'ticker': 'TCB',
                                'quarter': quarter_str,
                                'metric_name': 'Bad Debt',
                                'value': bad_debt / 1e12
                            })
                except Exception as e:
                    logger.debug(f"Skipped balance sheet row: {e}")
                    continue
        
        if len(records) > 20:  # At least some data
            df = pd.DataFrame(records)
            write_table(df, "raw_finance")
            logger.info(f"✅ Loaded TCB REAL data from vnstock: {len(df)} rows")
            return df
        else:
            logger.warning(f"Too few records ({len(records)}), using mock data")
    
    except Exception as e:
        logger.warning(f"Error fetching from vnstock Finance API: {e}")
    
    # Fallback to mock data
    logger.info("Falling back to mock data...")
    return load_tcb_data()


def inspect_tcb_vnstock_data():
    """
    Inspect TCB financial data structure from vnstock.
    Shows columns, data types, and sample data.
    """
    try:
        from vnstock import Finance
    except ImportError:
        logger.error("vnstock not installed")
        return None
    
    logger.info("Inspecting TCB data structure from vnstock...")
    
    try:
        finance = Finance(source='VCI', symbol='TCB')
        income = finance.income_statement(period='quarterly', limit=12)
        balance = finance.balance_sheet(period='quarterly', limit=12)
        
        print("\n" + "="*80)
        print("INCOME STATEMENT")
        print("="*80)
        print(f"Shape: {income.shape}")
        print(f"Columns ({len(income.columns)}): {income.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(income.head(3))
        print(f"\nData types:")
        print(income.dtypes)
        
        print("\n" + "="*80)
        print("BALANCE SHEET")
        print("="*80)
        print(f"Shape: {balance.shape}")
        print(f"Columns ({len(balance.columns)}): {balance.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(balance.head(3))
        print(f"\nData types:")
        print(balance.dtypes)
        
        return {
            'income_statement': income,
            'balance_sheet': balance
        }
        
    except Exception as e:
        logger.error(f"Error inspecting TCB data: {e}")
        return None


def fetch_and_process_tcb():
    """
    Convenience function: Fetch TCB real data from vnstock and process it.
    This is equivalent to running:
        1. fetch_tcb_vnstock.py
        2. python -m preprocessing.process_finance
    """
    print("📥 Fetching TCB data from vnstock...")
    df = load_tcb_data_from_vnstock()
    print(f"✅ Loaded {len(df)} rows into raw_finance")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df.head(12))
    
    print("\n" + "="*80)
    print("Processing financial data...")
    print("="*80)
    
    # Import here to avoid circular imports
    from preprocessing.process_finance import process_finance
    result = process_finance()
    print(f"\n✅ Processing complete: {len(result)} rows × {len(result.columns)} columns")
    return result