import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import pandas as pd
import numpy as np
import matplotlib
import os
from matplotlib import font_manager

# ==========================================
# 0. 網頁基本配置與字型處理
# ==========================================
st.set_page_config(page_title="2026 股市 AI 紅盤實作專案", layout="wide")

# 處理中文字型 (解決雲端 Linux 亂碼問題)
font_path = "NotoSansTC-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [prop.get_name()]
else:
    # 本機環境嘗試使用正黑體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

plt.rcParams['axes.unicode_minus'] = False

st.title("📊 2026 歡慶端午 2330 股市分析專案")
st.markdown("""
本專案演示了從資料獲取、多維度技術指標計算，到專業級 **6 大指標圖表（均線/布林帶、OBV、KDJ、MACD、RSI、BIAS）** 排版的完整流程。
""")

# ==========================================
# 1. 側邊欄與參數設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")
stock_id = st.sidebar.text_input("股票代號", "2330.TW")

# 加上 .date() 確保一開始就是純日期格式，避免 yfinance 初次載入報錯
default_start = datetime(2025, 10, 10).date()
default_end = datetime(2026, 5, 22).date()

target_start = st.sidebar.date_input("觀測起始日", default_start)
target_end = st.sidebar.date_input("觀測結束日", default_end)
warmup_days = st.sidebar.slider("指標預熱天數 (用於 EMA/RSI 準確度)", 30, 100, 60)

# 顯示字型狀態診斷
if os.path.exists(font_path):
    st.sidebar.success("✅ 已載入 NotoSansTC 中文字型")
else:
    st.sidebar.warning("⚠️ 使用系統預設中文字型 (雲端環境請確認已上傳 ttf 檔)")

# ==========================================
# 2. 步驟 1：資料獲取與「預熱」邏輯
# ==========================================
st.header("Step 1: 資料獲取與預熱處理")
with st.expander("📖 為什麼需要預熱資料？"):
    st.write("""
    - **預熱機制 (Warm-up)**：EMA、MACD 與 RSI 都是具備「延續性」的指標。如果直接從觀測日開始計算，初始值會產生嚴重的偏差。
    - 本程式自動向前抓取（預設 60 天）的資料進行「預熱」計算，確保在進入使用者選定的觀測區間時，所有指標已趨於穩定準確。
    - **避免格式錯誤**：強制將日期轉為 `YYYY-MM-DD` 格式再向 Yahoo Finance 請求，確保網頁一開啟就能正確載入資料。
    """)

@st.cache_data
def load_stock_data(symbol, start_dt, end_dt, warmup):
    # 向前推 warmup 天數
    fetch_start = start_dt - timedelta(days=warmup)
    
    # 強制轉換為字串格式，避免 datetime 物件造成 yfinance 解析異常
    start_str = fetch_start.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    df = yf.download(symbol, start=start_str, end=end_str, auto_adjust=False)
    
    if not df.empty:
        # 展平欄位名稱 (若 yfinance 回傳 MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return df

df_all = load_stock_data(stock_id, target_start, target_end, warmup_days)

if df_all.empty:
    st.error("找不到資料，請檢查代號或網路連線。")
    st.stop()

# ==========================================
# 3. 步驟 2：技術指標運算 (6大指標)
# ==========================================
st.header("Step 2: 技術指標運算 (Indicator Math)")
with st.expander("📖 查看 6 大指標運算邏輯說明"):
    st.markdown("""
    1. **均線與布林通道 (SMA & BBands)**：5日、10日、20日均線，以及 20日均線上下 2 倍標準差的軌道。
    2. **KDJ**：透過 9 日最高/最低價計算 RSV，再以 EWM (指數加權移動平均) 平滑出 K、D、J 值。
    3. **OBV (能量潮)**：利用成交量與股價漲跌的累計值，觀察資金進出動向。
    4. **MACD**：計算 12日與 26日 EMA 之差 (DIF)，以及其 9日訊號線 (MACD)。
    5. **RSI (相對強弱指標)**：使用 Yahoo Finance 官方標準公式 (修正平滑移動平均法) 計算 5日與 10日 RSI。
    6. **BIAS (乖離率)**：計算股價偏離 10日與 20日均線的百分比，並繪製兩者差距的柱狀圖判斷反轉點。
    """)

with st.spinner('各項指標計算中...'):
    df_calculated = df_all.copy()
    
    # 1. SMA & BBands
    df_calculated['SMA_5'] = df_calculated['Close'].rolling(window=5).mean()
    df_calculated['SMA_10'] = df_calculated['Close'].rolling(window=10).mean()
    df_calculated['SMA_20'] = df_calculated['Close'].rolling(window=20).mean()
    df_calculated['std_dev'] = df_calculated['Close'].rolling(window=20).std()
    df_calculated['upper_band'] = df_calculated['SMA_20'] + (df_calculated['std_dev'] * 2)
    df_calculated['lower_band'] = df_calculated['SMA_20'] - (df_calculated['std_dev'] * 2)

    # 2. KDJ (EWM 快速法)
    n = 9
    low_min = df_calculated['Low'].rolling(window=n).min()
    high_max = df_calculated['High'].rolling(window=n).max()
    df_calculated['RSV'] = ((df_calculated['Close'] - low_min) / (high_max - low_min)) * 100
    df_calculated['K'] = df_calculated['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df_calculated['D'] = df_calculated['K'].ewm(alpha=1/3, adjust=False).mean()
    df_calculated['J'] = 3 * df_calculated['D'] - 2 * df_calculated['K']

    # 3. OBV
    df_calculated['OBV'] = np.where(df_calculated['Close'] > df_calculated['Close'].shift(1), df_calculated['Volume'], -df_calculated['Volume']).cumsum()

    # 4. MACD
    df_calculated['EMA12'] = df_calculated['Close'].ewm(span=12, adjust=False).mean()
    df_calculated['EMA26'] = df_calculated['Close'].ewm(span=26, adjust=False).mean()
    df_calculated['DIF'] = df_calculated['EMA12'] - df_calculated['EMA26']
    df_calculated['MACD'] = df_calculated['DIF'].ewm(span=9, adjust=False).mean()
    df_calculated['MACD Histogram'] = df_calculated['DIF'] - df_calculated['MACD']

    # 5. RSI (Yahoo 靈魂公式)
    def yahoo_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df_calculated['RSI5'] = yahoo_rsi(df_calculated['Close'], 5)
    df_calculated['RSI10'] = yahoo_rsi(df_calculated['Close'], 10)

    # 6. BIAS 乖離率
    df_calculated['BIAS10'] = ((df_calculated['Close'] - df_calculated['SMA_10']) / df_calculated['SMA_10']) * 100
    df_calculated['BIAS20'] = ((df_calculated['Close'] - df_calculated['SMA_20']) / df_calculated['SMA_20']) * 100
    df_calculated['B10-B20'] = df_calculated['BIAS10'] - df_calculated['BIAS20']

# --- 過濾預熱資料 ---
# 確保 index 格式為 datetime，以利於進行時間過濾
df_calculated.index = pd.to_datetime(df_calculated.index)
mask_start = pd.Timestamp(target_start)
df = df_calculated.loc[mask_start:].copy()

# 將最終繪圖用的 DataFrame 索引轉為字串格式
df.index = df.index.map(lambda x: x.strftime('%y-%m-%d'))

with st.expander("🔍 查看已計算的指標數據 (觀測區間首五筆)"):
    st.dataframe(df.head())

# ==========================================
# 4. 步驟 3：專業多圖層視覺化
# ==========================================
st.header("Step 3: 綜合技術指標儀表板")
with st.expander("📖 查看圖表排版設計說明"):
    st.markdown("""
    本圖表使用 Matplotlib 的 `add_subplot(8, 1, ...)` 將畫布切分為 8 個單位高度：
    - **區塊 1-3 (主圖)**：顯示 K 線、5/10/20 日均線與布林通道。
    - **區塊 4 (OBV & Volume)**：結合能量潮曲線與成交量柱狀圖 (雙 Y 軸)。
    - **區塊 5 (KDJ)**：K、D、J 三線交叉觀察。
    - **區塊 6 (MACD)**：DIF、MACD 指標及其紅綠柱狀圖。
    - **區塊 7 (RSI)**：觀察 RSI5 與 RSI10 是否觸及超買 (70) 或超賣 (30) 虛線區間。
    - **區塊 8 (BIAS)**：10日與 20日乖離率，及兩者差距的柱狀圖 (輔助判斷極端行情)。
    - **視覺優化**：隱藏圖表間的重疊刻度，僅在底部的 BIAS 圖表顯示完整日期。
    """)

# 建立圖表畫布 (高度拉高到 16 以容納 6 個圖表)
fig = plt.figure(figsize=(14, 16), layout='constrained')

# 定義 6 大區塊 (總共 8 個單位)
ax1 = fig.add_subplot(8,1,(1,3)) # 主圖 (佔 3 單位)
ax2 = fig.add_subplot(8,1,4)     # OBV (佔 1 單位)
ax3 = fig.add_subplot(8,1,5)     # KDJ (佔 1 單位)
ax4 = fig.add_subplot(8,1,6)     # MACD (佔 1 單位)
ax5 = fig.add_subplot(8,1,7)     # RSI (佔 1 單位)
ax6 = fig.add_subplot(8,1,8)     # BIAS (佔 1 單位)

# 定義 X 軸刻度間隔 (每 15 根 K 棒顯示一次)
x_ticks_pos = range(0, len(df.index), 15)
x_ticks_labels = df.index[::15]

# --- Ax1: K線 + 均線 + 布林帶 ---
ax1.set_xticks(x_ticks_pos)
ax1.set_xticklabels(x_ticks_labels) # 隱藏重疊字體
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], 
                       width=0.8, colorup='r', colordown='g', alpha=1)
ax1.plot(df['SMA_5'],label='5日均線', color='cyan', lw=1)
ax1.plot(df['SMA_10'],label='10日均線', color='purple', lw=1)
ax1.plot(df['SMA_20'],label='20日均線', color='orange', lw=1)
ax1.plot(df['upper_band'], label='布林上軌', color='g', ls=':', lw=1)
ax1.plot(df['lower_band'], label='布林下軌', color='g', ls=':', lw=1)
ax1.legend(loc='upper left', fontsize='small')
ax1.set_title(f"【{stock_id}】綜合技術分析", fontsize=16)

# --- Ax2: OBV 與 成交量 ---
ax2.set_xticks(x_ticks_pos)
ax2.set_xticklabels([]) # 隱藏重疊字體
#vol_colors = np.where(df['Close'] > df['Close'].shift(1), 'r', 'g')
conditions = [
    df['Close'] > df['Close'].shift(1),  # 漲 -> 紅
    df['Close'] < df['Close'].shift(1)   # 跌 -> 綠
]
choices = ['r', 'g']
vol_colors = np.select(conditions, choices, default='gray')
ax2.plot(df['OBV'], color='purple', ls='--', label='OBV')
ax2_v = ax2.twinx()
ax2_v.bar(df.index, df['Volume'], color=vol_colors, alpha=0.7, width=0.8)
ax2.set_title("OBV 能量潮")
ax2.legend(loc='upper left', fontsize='small')

# --- Ax3: KDJ ---
ax3.plot(df['K'], label='K線', color='cyan', lw=1)
ax3.plot(df['D'], label='D線', color='purple', lw=1)
ax3.plot(df['J'], label='J線', color='orange', ls='--')
ax3.set_xticks(x_ticks_pos)
ax3.set_xticklabels(x_ticks_labels) # 隱藏重疊字體
ax3.set_title("KDJ 指標")
ax3.legend(loc='upper left', fontsize='small')

# --- Ax4: MACD ---
ax4.plot(df['DIF'], label='DIF', color='purple')
ax4.plot(df['MACD'], label='MACD', color='skyblue')
m_hist_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')
ax4.bar(df.index, df['MACD Histogram'], color=m_hist_colors, alpha=0.6)
ax4.axhline(0, color='gray', ls='--', lw=1)
ax4.set_xticks(x_ticks_pos)
ax4.set_xticklabels([]) # 隱藏重疊字體
ax4.set_title("MACD 指標")
ax4.legend(loc='upper left', fontsize='small')

# --- Ax5: RSI ---
ax5.plot(df['RSI5'], label='RSI5', color='cyan', lw=1)
ax5.plot(df['RSI10'], label='RSI10', color='purple', lw=1)
ax5.axhline(70, color='r', ls='--', lw=0.8, alpha=0.5) # 超買線
ax5.axhline(30, color='g', ls='--', lw=0.8, alpha=0.5) # 超賣線
ax5.set_ylim(0, 100)
ax5.set_xticks(x_ticks_pos)
ax5.set_xticklabels(x_ticks_labels) # 隱藏重疊字體
ax5.set_title("RSI 相對強弱指標")
ax5.legend(loc='upper left', fontsize='small')

# --- Ax6: BIAS (乖離率差距柱狀圖) ---
ax6.plot(df['BIAS10'], label='BIAS10', color='cyan', lw=1)
ax6.plot(df['BIAS20'], label='BIAS20', color='purple', lw=1)
bias_diff_colors = np.where(df['B10-B20'] >= 0, 'r', 'g')
ax6.bar(df.index, df['B10-B20'], color=bias_diff_colors, alpha=0.6)
ax6.axhline(0, color='gray', ls='--', lw=1)
ax6.set_xticks(x_ticks_pos)
ax6.set_xticklabels([]) # 最底部的圖表才顯示日期
ax6.set_title("BIAS 乖離率")
ax6.legend(loc='upper left', fontsize='small')

# 渲染到網頁
st.pyplot(fig)

st.divider()
st.info("💡 課程提示：觀察 MACD 柱狀圖與 RSI 超買超賣區，配合 BIAS 乖離率差距，可更全面判斷趨勢強弱。")
