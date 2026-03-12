"""
Tiền xử lý báo cáo tài chính + tính Financial Ratios.
═════════════════════════════════════════════════════
Phụ trách: Thành viên B
Branch: feature/process-finance
Chạy: python -m preprocessing.process_finance

Input:  raw_finance table
Output: clean_finance table (có financial ratios)
"""
import pandas as pd
from loguru import logger
from database.connection import read_table, write_table

logger.add("logs/process_finance.log", rotation="1 week")


def process_finance():
    """
    Đọc raw_finance → tính financial ratios → lưu clean_finance.

    Financial ratios cần tính (quan trọng cho ngân hàng TCB):
    - ROE (Return on Equity)
    - ROA (Return on Assets)
    - NIM (Net Interest Margin) — đặc trưng ngân hàng
    - P/E ratio
    - P/B ratio
    - Debt to Equity
    - NPL ratio (Non-performing loans) — đặc trưng ngân hàng
    - Cost to Income ratio
    - Revenue growth (YoY)
    - Profit growth (YoY)
    """
    logger.info("Tiền xử lý báo cáo tài chính...")

    df = read_table("raw_finance")
    logger.info(f"Loaded {len(df)} rows từ raw_finance")

    # TODO: Thành viên B implement
    # Gợi ý:
    # 1. Pivot raw_finance từ long format sang wide format (mỗi quarter 1 row)
    # 2. Tính các ratios từ số liệu gốc
    # 3. Tính growth rates so với cùng kỳ năm trước
    # 4. Xử lý missing values
    # 5. Lưu: write_table(df_clean, "clean_finance")
    #
    # Lưu ý:
    # - Format quarter: "2024-Q3"
    # - Một số ratios có thể không tính được nếu thiếu data
    # - NIM và NPL là metrics đặc trưng ngân hàng, rất quan trọng

    raise NotImplementedError("Thành viên B cần implement hàm này")


if __name__ == "__main__":
    process_finance()
