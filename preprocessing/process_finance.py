"""
Process financial data: Clean + Feature Engineering
═════════════════════════════════════════════════════

STAGE 2 + STAGE 3 Combined:
  ✓ Load full raw data from database (raw_finance table)
    ✓ Select ROE, ROA, debt_to_equity, net_profit_margin, financial_leverage
  ✓ Handle zero values (data quality)
  ✓ Forward fill missing quarters
  ✓ Calculate YoY change (4-quarter lag)
  ✓ Normalize (StandardScaler fit on train)
  ✓ Save features to database

Chạy: python preprocessing/process_finance.py
Input: database raw_finance table (24 rows)
Output: database features_finance table (24 rows × 11 cols)

Features used (5 metrics + YoY + lag):
    - ROE, ROA, debt_to_equity, net_profit_margin, financial_leverage
  - ROE_yoy, ROA_yoy (YoY change %, from Q5 onwards)
  - ROE_lag4, ROA_lag4 (4-quarter lag values)
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_connection
from database.schema import recreate_features_finance_table
from config.settings import SYMBOL

TARGET_QUARTERS = 24

def load_raw_from_database():
    """Đọc dữ liệu gốc từ database (raw_finance table)"""
    try:
        conn = get_connection()
        query = "SELECT * FROM raw_finance WHERE symbol = ? ORDER BY date"
        df = pd.read_sql_query(query, conn, params=(SYMBOL,))
        conn.close()
        
        if df.empty:
            print("ERROR: No data found in raw_finance table!")
            return None
        
        print(f"  -> Loaded {len(df)} rows from database (raw_finance)")
        return df
        
    except Exception as e:
        print(f"ERROR reading from database: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_features_to_database(df):
    """Lưu dữ liệu đã xử lý vào database (features_finance table)"""
    try:
        # Recreate table để đảm bảo có đủ schema mới
        recreate_features_finance_table()
        
        conn = get_connection()
        
        # Clear bảng cũ (cùng symbol)
        conn.execute("DELETE FROM features_finance WHERE symbol = ?", (SYMBOL,))
        
        # Insert từng row
        for _, row in df.iterrows():
            values = {
                'symbol': row.get('symbol', SYMBOL),
                'date': str(row.get('date')),
                'roe': row.get('roe'),
                'roa': row.get('roa'),
                'debt_to_equity': row.get('debt_to_equity'),
                'net_profit_margin': row.get('net_profit_margin'),
                'financial_leverage': row.get('financial_leverage'),
                'roe_yoy': row.get('roe_yoy'),
                'roa_yoy': row.get('roa_yoy'),
                'roe_lag4': row.get('roe_lag4'),
                'roa_lag4': row.get('roa_lag4'),
            }
            
            # Convert to proper types, handle None and Series conversions
            for key in values:
                if key not in ['symbol', 'date']:
                    val = values[key]
                    # Convert Series to scalar
                    if hasattr(val, 'item'):
                        val = val.item()
                    if pd.isna(val):
                        values[key] = None
                    else:
                        try:
                            values[key] = float(val)
                        except:
                            values[key] = None
            
            cols = ', '.join(values.keys())
            placeholders = ', '.join(['?' for _ in values])
            query = f"INSERT INTO features_finance ({cols}) VALUES ({placeholders})"

            def _to_sql_param(key, val):
                if val is None:
                    return None
                if hasattr(val, 'item') and not isinstance(val, (str, bytes)):
                    try:
                        val = val.item()
                    except Exception:
                        pass
                if pd.isna(val):
                    return None
                if key in ('symbol', 'date'):
                    return str(val)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            try:
                final_values = tuple(_to_sql_param(k, values[k]) for k in values.keys())
                conn.execute(query, final_values)
            except Exception as e:
                print(f"  WARN: insert row failed: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"  -> Saved {len(df)} rows to database (features_finance table)")
        
    except Exception as e:
        print(f"ERROR saving to database: {e}")
        import traceback
        traceback.print_exc()

def process_and_engineer_finance():
    """Process financial data: Clean + Feature Engineering"""
    print("\n" + "="*70)
    print("[STAGE 2+3] CLEAN & ENGINEER FEATURES")
    print("="*70)
    
    # ============ STAGE 2: CLEAN ============
    print("\n[STAGE 2] Cleaning financial data...")
    
    # Load from database instead of CSV
    df = load_raw_from_database()
    if df is None:
        return None
    
    print(f"  -> Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"  -> Will keep only last {TARGET_QUARTERS} quarters for training after YoY calculation")
    
    feature_cols = [
        'roe',
        'roa',
        'debt_to_equity',
        'net_profit_margin',
        'financial_leverage',
    ]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in raw_finance: {missing_cols}")
        return None

    keep_cols = ['symbol', 'date'] + feature_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    if 'symbol' not in df.columns:
        df.insert(0, 'symbol', SYMBOL)
    
    print(f"  -> Selected columns: {df.columns.tolist()}")
    
    rows_before = len(df)
    df = df.dropna(subset=feature_cols, how='all')
    print(f"  -> Dropped null rows: {rows_before} → {len(df)} rows")
    
    # ============ STAGE 3: FEATURE ENGINEERING ============
    print("\n[STAGE 3] Feature engineering...")
    
    # 1. Handle 0 values (data quality issue) - replace with NaN before fill
    for col in feature_cols:
        df.loc[df[col] == 0, col] = np.nan
    print(f"  -> Replaced 0 values with NaN")
    
    # 2. Fill NaN robustly for time series: forward-fill then back-fill leading gaps
    numeric_cols = feature_cols
    for col in numeric_cols:
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()
    print(f"  -> Forward/backward filled NaN")
    
    # 3. Remove outliers (Z-score > 3)
    rows_before = len(df)
    for col in numeric_cols:
        mask = df[col].notna()
        if mask.sum() > 2:
            z_scores = np.abs(stats.zscore(df.loc[mask, col]))
            outlier_mask = z_scores > 3
            if outlier_mask.any():
                df = df[~df.index.isin(df.loc[mask][outlier_mask].index)]
    
    print(f"  -> Outlier removal: {rows_before} → {len(df)} rows")
    
    # 4. YoY calculations (4-quarter lag = 1 year)
    print(f"  -> Calculating YoY changes (4Q lag)...")
    for col in ['roe', 'roa']:
        df[f'{col}_yoy'] = df[col].pct_change(periods=4) * 100  # percentage

    # 4.1 Remove synthetic 0 YoY values and forward-fill as requested
    yoy_cols = ['roe_yoy', 'roa_yoy']
    for col in yoy_cols:
        df.loc[df[col] == 0, col] = np.nan
        df[col] = df[col].ffill()
        df[col] = df[col].bfill()
    
    # 5. Calculate lag values
    # Also keep lag values for training purposes
    for col in ['roe', 'roa']:
        df[f'{col}_lag4'] = df[col].shift(4)  # Value 4Q ago
    
    print(f"  -> Raw columns: {numeric_cols}")
    print(f"  -> YoY columns: {[f'{c}_yoy' for c in ['roe', 'roa']]}")
    print(f"  -> Lag columns: {[f'{c}_lag4' for c in ['roe', 'roa']]}")
    
    # 6. Keep target window only (last 24Q) after using older data for YoY/lag
    if len(df) > TARGET_QUARTERS:
        rows_before = len(df)
        df = df.tail(TARGET_QUARTERS).reset_index(drop=True)
        print(f"\n[STAGE 3] Trimming to target window: {rows_before} → {len(df)} rows")

    # 7. Normalize (fit on past, apply to future - time series safe)
    print(f"\n[STAGE 3] Normalizing features...")
    
    # Split data for scaling (train: first 16Q, then apply to val/test)
    train_idx = 16  # 4Q before split
    if len(df) >= train_idx:
        train_df = df.iloc[:train_idx].copy()
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(train_df[numeric_cols])
        
        # Apply to all data
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
        
        print(f"  -> Fitted scaler on {len(train_df)} training rows")
        print(f"  -> Scaled all {len(df_scaled)} rows")
        
        df = df_scaled
    
    # 8. Save processed data to database
    print(f"\n[OUTPUT] Saving processed features to database...")
    
    save_features_to_database(df)
    
    print(f"  -> Shape: {df.shape}")
    print(f"  -> Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"\n  -> First 5 rows:")
    print(df.head().to_string())
    
    print("\n" + "="*70)
    print("SUCCESS: Features processed and saved to database")
    print("="*70 + "\n")
    
    return df


def process_finance():
    """Entry dùng bởi `python -m preprocessing.process_finance` và `fetch_and_process_tcb`."""
    return process_and_engineer_finance()


if __name__ == "__main__":
    process_finance()