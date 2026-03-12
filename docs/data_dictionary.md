# Data Dictionary — TCB Prediction

## raw_prices
| Column | Type | Mô tả |
|--------|------|-------|
| date | TEXT | Ngày giao dịch (YYYY-MM-DD) |
| open | REAL | Giá mở cửa (VND) |
| high | REAL | Giá cao nhất (VND) |
| low | REAL | Giá thấp nhất (VND) |
| close | REAL | Giá đóng cửa (VND) |
| volume | INTEGER | Khối lượng giao dịch |

## Technical Indicators (clean_prices)
| Column | Mô tả |
|--------|-------|
| sma_10, sma_20, sma_50 | Simple Moving Average |
| ema_12, ema_26 | Exponential Moving Average |
| rsi_14 | Relative Strength Index (0-100) |
| macd, macd_signal, macd_hist | MACD indicators |
| bb_upper, bb_middle, bb_lower | Bollinger Bands |
| atr_14 | Average True Range |
| obv | On-Balance Volume |

## Financial Ratios (clean_finance)
| Column | Mô tả |
|--------|-------|
| roe | Return on Equity |
| roa | Return on Assets |
| nim | Net Interest Margin (đặc trưng ngân hàng) |
| pe_ratio | Price to Earnings |
| pb_ratio | Price to Book |
| debt_to_equity | Tỷ lệ nợ trên vốn |

## Sentiment (clean_news)
| Column | Mô tả |
|--------|-------|
| sentiment_score | -1 (tiêu cực) đến +1 (tích cực) |
| sentiment_pos | Xác suất positive |
| sentiment_neg | Xác suất negative |
| sentiment_neu | Xác suất neutral |
