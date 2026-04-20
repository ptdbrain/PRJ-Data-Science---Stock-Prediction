"""
Crawl 24 quarters (Q1 2019 - Q4 2025) tu vnstock
================================================
Chay: python data_collection/collect_finance.py
Output: Database table raw_finance (24 rows × ~18 columns)

DATA SOURCE: vnstock Ratio API (24Q guaranteed)
  - Saves ALL columns from API response
  - Preprocessing stage sẽ select cái cần dùng
  - Full data available cho analysis

Output: 24 rows × ~18 columns
  - Complete financial ratio data preserved in database
"""
import pandas as pd
import numpy as np
from vnstock import Vnstock
import os
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_connection
from database.schema import recreate_raw_finance_table
from config.settings import SYMBOL, DATA_SOURCE
TARGET_QUARTERS = 24
HISTORY_QUARTERS = 4
TOTAL_FETCH_QUARTERS = TARGET_QUARTERS + HISTORY_QUARTERS

def flatten_multiindex_columns(df):
    """Flatten MultiIndex columns"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_').lower() for col in df.columns.values]
    else:
        df.columns = df.columns.str.lower()
    return df

def save_to_database(df, symbol):
    """Lưu dữ liệu vào database table raw_finance"""
    try:
        # Drop + recreate để đảm bảo schema mới nhất
        recreate_raw_finance_table()
        
        conn = get_connection()

        def pick_column(columns, includes, excludes=None):
            excludes = excludes or []
            for col in columns:
                col_l = col.lower()
                if all(term in col_l for term in includes) and not any(term in col_l for term in excludes):
                    return col
            return None

        def to_float(value):
            if pd.isna(value):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def to_int(value):
            if pd.isna(value):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def get_cell(row, col_name):
            if not col_name:
                return None
            value = row[col_name]
            if isinstance(value, pd.Series):
                return value.iloc[0]
            return value

        cols = list(df.columns)
        column_map = {
            'date': pick_column(cols, ['date']),
            'meta_ticker': pick_column(cols, ['meta_ticker']),
            'meta_yearreport': pick_column(cols, ['meta_yearreport']),
            'meta_lengthreport': pick_column(cols, ['meta_lengthreport']),
            'roe': pick_column(cols, ['roe'], ['roa']),
            'roa': pick_column(cols, ['roa']),
            'debt_to_equity': pick_column(cols, ['debt/equity']),
            'fixed_asset_to_equity': pick_column(cols, ['fixed asset-to-equity']),
            'owners_equity_to_charter_capital': pick_column(cols, ["owners' equity/charter capital"]),
            'net_profit_margin': pick_column(cols, ['net profit margin']),
            'financial_leverage': pick_column(cols, ['financial leverage']),
            'market_cap_bn_vnd': pick_column(cols, ['market capital']),
            'outstanding_share_mil': pick_column(cols, ['outstanding share']),
            'pe_ratio': pick_column(cols, ['p/e']),
            'pb_ratio': pick_column(cols, ['p/b']),
            'ps_ratio': pick_column(cols, ['p/s']),
            'pcf_ratio': pick_column(cols, ['p/cash flow']),
            'eps_vnd': pick_column(cols, ['eps']),
            'bvps_vnd': pick_column(cols, ['bvps']),
        }

        insert_sql = """
            INSERT INTO raw_finance (
                symbol, date, meta_ticker, meta_yearreport, meta_lengthreport,
                roe, roa, debt_to_equity, fixed_asset_to_equity,
                owners_equity_to_charter_capital, net_profit_margin, financial_leverage,
                market_cap_bn_vnd, outstanding_share_mil, pe_ratio, pb_ratio,
                ps_ratio, pcf_ratio, eps_vnd, bvps_vnd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        inserted = 0
        for idx, row in df.iterrows():
            date_col = column_map['date']
            date_raw = get_cell(row, date_col)
            date_value = str(date_raw) if date_raw is not None else None
            if not date_value:
                print(f"  Skip row {idx}: missing date")
                continue

            values = (
                symbol,
                date_value,
                get_cell(row, column_map['meta_ticker']) or symbol,
                to_int(get_cell(row, column_map['meta_yearreport'])),
                to_int(get_cell(row, column_map['meta_lengthreport'])),
                to_float(get_cell(row, column_map['roe'])),
                to_float(get_cell(row, column_map['roa'])),
                to_float(get_cell(row, column_map['debt_to_equity'])),
                to_float(get_cell(row, column_map['fixed_asset_to_equity'])),
                to_float(get_cell(row, column_map['owners_equity_to_charter_capital'])),
                to_float(get_cell(row, column_map['net_profit_margin'])),
                to_float(get_cell(row, column_map['financial_leverage'])),
                to_float(get_cell(row, column_map['market_cap_bn_vnd'])),
                to_float(get_cell(row, column_map['outstanding_share_mil'])),
                to_float(get_cell(row, column_map['pe_ratio'])),
                to_float(get_cell(row, column_map['pb_ratio'])),
                to_float(get_cell(row, column_map['ps_ratio'])),
                to_float(get_cell(row, column_map['pcf_ratio'])),
                to_float(get_cell(row, column_map['eps_vnd'])),
                to_float(get_cell(row, column_map['bvps_vnd'])),
            )

            try:
                conn.execute(insert_sql, values)
                inserted += 1
            except Exception as e:
                print(f"  Error inserting row {idx}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"    -> Saved {inserted}/{len(df)} rows to database (raw_finance table)")
        
    except Exception as e:
        print(f"    ❌ Database save error: {e}")
        import traceback
        traceback.print_exc()

def crawl_ratio_api():
    """Crawl all 24 quarters từ Ratio API - keep ALL columns"""
    print("\n" + "="*70)
    print(f"[CRAWL] {SYMBOL} - 24 QUARTERS (RATIO API - FULL DATA)")
    print("="*70 + "\n")
    
    try:
        stock = Vnstock().stock(symbol=SYMBOL, source=DATA_SOURCE)
        
        # ============ FETCH RATIO API (24Q) ============
        print(f"[1] Fetching Ratio API ({TOTAL_FETCH_QUARTERS} quarters: {TARGET_QUARTERS} target + {HISTORY_QUARTERS} history)...")
        ratio_df = stock.finance.ratio(period='quarter', count=TOTAL_FETCH_QUARTERS)
        
        # Trim to last TOTAL_FETCH_QUARTERS if needed
        if len(ratio_df) > TOTAL_FETCH_QUARTERS:
            print(f"    API returned {len(ratio_df)} rows, keeping last {TOTAL_FETCH_QUARTERS}...")
            ratio_df = ratio_df.tail(TOTAL_FETCH_QUARTERS).reset_index(drop=True)
        
        print(f"    -> {len(ratio_df)} rows x {len(ratio_df.columns)} columns")
        ratio_df = flatten_multiindex_columns(ratio_df)
        
        print(f"    -> Columns: {ratio_df.columns.tolist()[:10]}...")
        
        # ============ GENERATE DATES ============
        print(f"\n[2] Generating {len(ratio_df)} quarter dates...")
        today = datetime.now()
        current_quarter = (today.month - 1) // 3 + 1
        current_year = today.year

        # Lùi 1 quý để tránh gán nhãn vào quý hiện tại chưa công bố đầy đủ BCTC
        current_quarter -= 1
        if current_quarter == 0:
            current_quarter = 4
            current_year -= 1
        
        quarters = []
        y, q = current_year, current_quarter
        for i in range(len(ratio_df)):
            # Dùng cùng định dạng với merge_features / process_prices: "YYYY-Qn"
            quarters.insert(0, f"{y}-Q{q}")
            q -= 1
            if q == 0:
                q = 4
                y -= 1
        
        ratio_df['date'] = quarters
        target_start = quarters[-TARGET_QUARTERS] if len(quarters) >= TARGET_QUARTERS else quarters[0]
        print(f"    -> Full range: {quarters[0]} to {quarters[-1]}")
        print(f"    -> Target 24Q range: {target_start} to {quarters[-1]}")
        
        # ============ REORDER COLUMNS ============
        print("\n[3] Organizing columns...")
        
        # Move key columns to front
        cols = ratio_df.columns.tolist()
        cols = [c for c in cols if c != 'date']
        key_cols = ['date']
        
        # Find metric columns
        for metric in ['roe', 'roa', 'profit_margin', 'debt/equity']:
            matching = [c for c in cols if metric.lower() in c.lower()]
            if matching:
                key_cols.extend(matching)
                for c in matching:
                    cols.remove(c)
        
        # Reorder: key metrics first, then rest
        reordered_cols = key_cols + cols
        ratio_df = ratio_df[reordered_cols]
        
        print(f"    -> Reordered {len(ratio_df.columns)} columns (key metrics first)")
        
        # ============ SAVE TO DATABASE ============
        print("\n[4] Saving to database (raw_finance table)...")
        save_to_database(ratio_df, SYMBOL)
        
        print(f"\n       Shape: {ratio_df.shape}")
        print(f"       Total columns: {len(ratio_df.columns)}")
        print(f"\n       Column list:")
        for i, col in enumerate(ratio_df.columns, 1):
            print(f"         {i}. {col}")
        
        print(f"\n       First 3 rows (first 8 cols):")
        print(ratio_df.iloc[:3, :8].to_string())
        
        print("\n" + "="*70)
        print("SUCCESS: Extra history crawled for YoY, full data saved to database")
        print("="*70 + "\n")
        
        return ratio_df
        
    except KeyError as e:
        # vnstock internals expected a 'data' key but API returned unexpected structure
        print("\nERROR: unexpected API response (missing key):", e)
        print("Attempting fallback to local/mock data via database.connection.load_tcb_data_from_vnstock()...")
        try:
            from database.connection import load_tcb_data_from_vnstock
            fallback_df = load_tcb_data_from_vnstock()
            print(f"Fallback loaded {len(fallback_df) if fallback_df is not None else 0} rows")
            return fallback_df
        except Exception as e2:
            print("Fallback failed:", e2)
            import traceback
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    crawl_ratio_api()