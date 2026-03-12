"""
Thu thập báo cáo tài chính TCB (3 năm, theo quý).
═══════════════════════════════════════════════════
Phụ trách: Thành viên B
Branch: feature/collect-finance
Chạy: python -m data_collection.collect_finance

Output: raw_finance table trong SQLite
"""
from vnstock import Vnstock
from loguru import logger
from database.connection import get_connection, write_table
from config.settings import SYMBOL, DATA_SOURCE

logger.add("logs/collect_finance.log", rotation="1 week")


def collect_finance():
    """
    Lấy báo cáo tài chính TCB (income, balance sheet, cash flow) và lưu vào raw_finance.

    Cần lấy 3 loại báo cáo:
    - Income Statement (Kết quả kinh doanh)
    - Balance Sheet (Bảng cân đối kế toán)
    - Cash Flow Statement (Lưu chuyển tiền tệ)

    Mỗi báo cáo lấy theo quý, 12 quý gần nhất (3 năm).
    """
    logger.info(f"Thu thập báo cáo tài chính {SYMBOL}...")

    # TODO: Thành viên B implement
    # Gợi ý:
    # 1. Dùng Vnstock().stock(symbol=SYMBOL, source=DATA_SOURCE)
    # 2. Gọi stock.finance.income_statement(period='quarter')
    # 3. Gọi stock.finance.balance_sheet(period='quarter')
    # 4. Gọi stock.finance.cash_flow(period='quarter')
    # 5. Chuyển đổi format phù hợp (xem schema raw_finance)
    # 6. Lưu bằng write_table(df, "raw_finance")
    #
    # Lưu ý: vnstock có thể trả về format khác nhau tuỳ version
    # Cần kiểm tra output thực tế và adjust cho phù hợp

    raise NotImplementedError("Thành viên B cần implement hàm này")


if __name__ == "__main__":
    collect_finance()
