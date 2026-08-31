import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import pandas as pd
import numpy as np
import os
from matplotlib import font_manager

# ==========================================
# 0. 網頁基本配置與字型處理
# ==========================================
st.set_page_config(
    page_title="2026 股市 AI 紅綠燈多空指標實作專案",
    page_icon="📈",
    layout="wide"
)

# 處理中文字型 (解決雲端 Linux 亂碼問題)
font_path = "NotoSansTC-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [prop.get_name()]
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False

st.title("📊 2026 股市 AI 紅綠燈多空指標分析系統")
st.markdown("""
本系統結合專業技術指標運算與**多空紅綠燈診斷模型**：
- 🔴 **紅燈 (看多 / +1 分)**：偏多訊號、黃金交叉、均線支撐或超賣底部反彈。
- 🟢 **綠燈 (看空 / -1 分)**：偏空訊號、死亡交叉、均線反壓或超買過熱修正。
- 🟡 **黃燈 (中立 / 0 分)**：盤整無明確方向或處於中性區間。
""")

# ==========================================
# 1. 側邊欄與參數設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")
stock_id = st.sidebar.text_input("股票代號", "2330.TW")

default_start = datetime(2025, 11, 19).date()
default_end = datetime(2026, 5, 22).date()

target_start = st.sidebar.date_input("觀測起始日", default_start)
target_end = st.sidebar.date_input("觀測結束日", default_end)
warmup_days = st.sidebar.slider("指標預熱天數 (用於 EMA/RSI 準確度)", 30, 100, 60)

# ==========================================
# 2. 步驟 1：資料獲取與「預熱」邏輯
# ==========================================
st.header("Step 1: 資料獲取與預熱處理")
with st.expander("📖 為什麼需要預熱資料？"):
    st.write("""
    - **預熱機制 (Warm-up)**：EMA、MACD 與 RSI 都是具備「延續性」的指標。如果直接從觀測日開始計算，初始值會產生嚴重的偏差。
    - 本程式自動向前抓取（預設 60 天）的資料進行「預熱」計算，確保在進入使用者選定的觀測區間時，所有指標已趨於穩定準確。
    - **避免格式錯誤**：強制將日期轉為 `YYYY-MM-DD` 格式向 Yahoo Finance 請求，確保解析穩定。
    """)

@st.cache_data
def load_stock_data(symbol, start_dt, end_dt, warmup):
    fetch_start = start_dt - timedelta(days=warmup)
    start_str = fetch_start.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    df = yf.download(symbol, start=start_str, end=end_str, auto_adjust=False)
    
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df_all = load_stock_data(stock_id, target_start, target_end, warmup_days)

if df_all.empty:
    st.error("找不到股票資料，請檢查股票代號或網路連線。")
    st.stop()

# ==========================================
# 3. 步驟 2：技術指標運算 (6大指標)
# ==========================================
st.header("Step 2: 技術指標運算 (Indicator Math)")

with st.spinner('各項技術指標運算中...'):
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
    df_calculated['RSV'] = ((df_calculated['Close'] - low_min) / (high_max - low_min + 1e-9)) * 100
    df_calculated['K'] = df_calculated['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df_calculated['D'] = df_calculated['K'].ewm(alpha=1/3, adjust=False).mean()
    df_calculated['J'] = 3 * df_calculated['D'] - 2 * df_calculated['K']

    # 3. OBV
    df_calculated['OBV'] = np.where(
        df_calculated['Close'] > df_calculated['Close'].shift(1),
        df_calculated['Volume'],
        np.where(df_calculated['Close'] < df_calculated['Close'].shift(1), -df_calculated['Volume'], 0)
    ).cumsum()
    df_calculated['OBV_EMA'] = df_calculated['OBV'].ewm(span=10, adjust=False).mean()

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
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df_calculated['RSI5'] = yahoo_rsi(df_calculated['Close'], 5)
    df_calculated['RSI10'] = yahoo_rsi(df_calculated['Close'], 10)

    # 6. BIAS 乖離率
    df_calculated['BIAS10'] = ((df_calculated['Close'] - df_calculated['SMA_10']) / df_calculated['SMA_10']) * 100
    df_calculated['BIAS20'] = ((df_calculated['Close'] - df_calculated['SMA_20']) / df_calculated['SMA_20']) * 100
    df_calculated['B10-B20'] = df_calculated['BIAS10'] - df_calculated['BIAS20']

# ==========================================
# 4. 步驟 3：指標多空紅綠燈分析與總分評定
# ==========================================
st.header("Step 3: 指標多空紅綠燈分析與總分評定 (Signal Diagnostics)")

# 多空規則定義函式
def analyze_signals(row, prev_row):
    signals = []
    
    # 1. 均線排列 (SMA Trend)
    if row['Close'] > row['SMA_5'] and row['SMA_5'] > row['SMA_20']:
        signals.append(("均線趨勢 (SMA)", 1, "🔴 多頭排列", f"收盤價 {row['Close']:.1f} 站上 5MA ({row['SMA_5']:.1f}) 且 5MA > 20MA"))
    elif row['Close'] < row['SMA_5'] and row['SMA_5'] < row['SMA_20']:
        signals.append(("均線趨勢 (SMA)", -1, "🟢 空頭排列", f"收盤價 {row['Close']:.1f} 跌破 5MA ({row['SMA_5']:.1f}) 且 5MA < 20MA"))
    elif row['Close'] > row['SMA_20']:
        signals.append(("均線趨勢 (SMA)", 1, "🔴 站上月線", f"收盤價站穩 20MA ({row['SMA_20']:.1f}) 支撐"))
    else:
        signals.append(("均線趨勢 (SMA)", -1, "🟢 跌破月線", f"收盤價落於 20MA ({row['SMA_20']:.1f}) 之下"))

    # 2. 布林通道 (Bollinger Bands)
    if row['Close'] >= row['upper_band']:
        signals.append(("布林通道 (BBands)", -1, "🟢 觸碰上軌警戒", f"股價觸及布林上軌 ({row['upper_band']:.1f})，需防高檔回檔"))
    elif row['Close'] <= row['lower_band']:
        signals.append(("布林通道 (BBands)", 1, "🔴 觸碰下軌超跌", f"股價落於布林下軌 ({row['lower_band']:.1f})，具超跌反彈契機"))
    elif row['Close'] > row['SMA_20']:
        signals.append(("布林通道 (BBands)", 1, "🔴 通道中軌之上", "處於布林中軌與上軌間的強勢多方區間"))
    else:
        signals.append(("布林通道 (BBands)", -1, "🟢 通道中軌之下", "處於布林中軌與下軌間的弱勢空方區間"))

    # 3. KDJ 指標
    if row['K'] > row['D'] and row['K'] < 80:
        signals.append(("KDJ 隨機指標", 1, "🔴 K > D 黃金交叉", f"K值 ({row['K']:.1f}) 高於 D值 ({row['D']:.1f})，動能偏多"))
    elif row['K'] < row['D'] and row['K'] > 20:
        signals.append(("KDJ 隨機指標", -1, "🟢 K < D 死亡交叉", f"K值 ({row['K']:.1f}) 低於 D值 ({row['D']:.1f})，動能偏空"))
    elif row['K'] >= 80:
        signals.append(("KDJ 隨機指標", -1, "🟢 KDJ 超買鈍化", f"K值 ({row['K']:.1f}) 進入 >80 超買警戒區"))
    elif row['K'] <= 20:
        signals.append(("KDJ 隨機指標", 1, "🔴 KDJ 超賣醞釀", f"K值 ({row['K']:.1f}) 進入 <20 超賣反彈區"))
    else:
        signals.append(("KDJ 隨機指標", 0, "🟡 中性整理", "KDJ 指標無明顯趨勢"))

    # 4. MACD 指標
    if row['MACD Histogram'] > 0 and row['DIF'] > row['MACD']:
        signals.append(("MACD 柱狀指標", 1, "🔴 紅柱偏多擴張", f"DIF ({row['DIF']:.2f}) > MACD ({row['MACD']:.2f})，紅柱動能向上"))
    elif row['MACD Histogram'] <= 0 and row['DIF'] < row['MACD']:
        signals.append(("MACD 柱狀指標", -1, "🟢 綠柱偏空延伸", f"DIF ({row['DIF']:.2f}) < MACD ({row['MACD']:.2f})，綠柱動能向下"))
    else:
        signals.append(("MACD 柱狀指標", 0, "🟡 零軸交界整理", "MACD 柱狀體收斂"))

    # 5. RSI 相對強弱
    if row['RSI5'] > 80:
        signals.append(("RSI 強弱指標", -1, "🟢 RSI 嚴重超買", f"5日 RSI ({row['RSI5']:.1f}) > 80，隨時有獲利了結壓力"))
    elif row['RSI5'] < 20:
        signals.append(("RSI 強弱指標", 1, "🔴 RSI 嚴重超賣", f"5日 RSI ({row['RSI5']:.1f}) < 20，短線超跌反彈訊號"))
    elif row['RSI5'] > row['RSI10']:
        signals.append(("RSI 強弱指標", 1, "🔴 5日 RSI > 10日 RSI", f"RSI5 ({row['RSI5']:.1f}) 站上 RSI10 ({row['RSI10']:.1f})，短多續強"))
    else:
        signals.append(("RSI 強弱指標", -1, "🟢 5日 RSI < 10日 RSI", f"RSI5 ({row['RSI5']:.1f}) 跌破 RSI10 ({row['RSI10']:.1f})，短線弱勢"))

    # 6. OBV 能量潮 / 量價動能
    if row['OBV'] > row['OBV_EMA']:
        signals.append(("OBV 能量潮", 1, "🔴 資金持續流入", f"OBV 位於 10日均線之上，量能支撐漲勢"))
    else:
        signals.append(("OBV 能量潮", -1, "🟢 資金流出警戒", f"OBV 跌破 10日均線，買盤後繼量能不足"))

    # 7. BIAS 乖離率差距
    if row['B10-B20'] > 0 and abs(row['BIAS10']) < 6:
        signals.append(("BIAS 乖離率差距", 1, "🔴 短期乖離擴散", f"B10-B20 為正 ({row['B10-B20']:.2f}%)，均線正向發散助漲"))
    elif row['B10-B20'] < 0 and abs(row['BIAS10']) < 6:
        signals.append(("BIAS 乖離率差距", -1, "🟢 短期乖離壓縮/偏空", f"B10-B20 為負 ({row['B10-B20']:.2f}%)，均線負向壓制"))
    elif row['BIAS10'] >= 6:
        signals.append(("BIAS 乖離率差距", -1, "🟢 正乖離過大", f"10日乖離高達 {row['BIAS10']:.2f}%，有技術性回調風險"))
    else:
        signals.append(("BIAS 乖離率差距", 1, "🔴 負乖離過大超跌", f"10日乖離為 {row['BIAS10']:.2f}%，負乖離過大醞釀反彈"))

    return signals

# 過濾預熱資料
df_calculated.index = pd.to_datetime(df_calculated.index)
mask_start = pd.Timestamp(target_start)
df = df_calculated.loc[mask_start:].copy()

if df.empty:
    st.warning("選取區間內無交易資料，請調整觀測起始日。")
    st.stop()

# 計算歷史每日總分
def compute_daily_total_score(df_input):
    daily_scores = []
    for i in range(len(df_input)):
        curr_row = df_input.iloc[i]
        prev_row = df_input.iloc[i-1] if i > 0 else curr_row
        sigs = analyze_signals(curr_row, prev_row)
        tot = sum(s[1] for s in sigs)
        daily_scores.append(tot)
    return daily_scores

df['Total_Score'] = compute_daily_total_score(df)

# 取出最後一天的最新數據進行診斷
latest_row = df.iloc[-1]
prev_row = df.iloc[-2] if len(df) > 1 else latest_row
latest_signals = analyze_signals(latest_row, prev_row)

total_score = sum(s[1] for s in latest_signals)
max_possible = len(latest_signals)
min_possible = -len(latest_signals)

# 評斷綜合多空狀態
if total_score >= 4:
    sentiment = "🚀 強烈看多 (Strong Bullish)"
    sentiment_color = "#d32f2f"
    sentiment_desc = "多數指標亮起紅燈，多方動能充沛，均線與動能指標皆處於上升軌道。"
elif 1 <= total_score <= 3:
    sentiment = "📈 偏多震盪 (Mild Bullish)"
    sentiment_color = "#e57373"
    sentiment_desc = "多方指標佔優勢，但部分動能指標顯示震盪或過熱，建議逢回低接。"
elif total_score == 0:
    sentiment = "⚖️ 多空平衡 / 盤整 (Neutral)"
    sentiment_color = "#ffa000"
    sentiment_desc = "多空力道勢均力敵，建議觀望等待帶量突破方向。"
elif -3 <= total_score <= -1:
    sentiment = "📉 偏空震盪 (Mild Bearish)"
    sentiment_color = "#81c784"
    sentiment_desc = "空方指標佔優勢，反彈面臨均線反壓，操作宜謹慎保守。"
else:
    sentiment = "🩸 強烈看空 (Strong Bearish)"
    sentiment_color = "#388e3c"
    sentiment_desc = "多數指標亮起綠燈，賣壓沉重且動能走疲，建議保守避開跌勢。"

# 頂部總結儀表板
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("最新收盤價", f"{latest_row['Close']:.2f}", f"{(latest_row['Close'] - prev_row['Close']):.2f}")
with kpi2:
    red_count = sum(1 for s in latest_signals if s[1] == 1)
    green_count = sum(1 for s in latest_signals if s[1] == -1)
    yellow_count = sum(1 for s in latest_signals if s[1] == 0)
    st.metric("紅燈 (多) / 綠燈 (空)", f"🔴 {red_count} / 🟢 {green_count}", f"🟡 中立 {yellow_count}")
with kpi3:
    st.metric("綜合指標總分", f"{total_score:+d} 分", f"滿分區間 [{min_possible}, +{max_possible}]")
with kpi4:
    st.metric("診斷評級", sentiment.split()[1], sentiment.split()[0])

st.markdown(
    f"""
    <div style="background-color: {sentiment_color}22; border-left: 6px solid {sentiment_color}; padding: 12px 16px; border-radius: 6px; margin: 10px 0 20px 0;">
        <h4 style="margin: 0; color: {sentiment_color}; font-size: 1.1rem;">評估結論：{sentiment} (總分: {total_score:+d})</h4>
        <p style="margin: 4px 0 0 0; color: #444; font-size: 0.95rem;">{sentiment_desc}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 顯示 7 大指標詳細紅綠燈列表
st.subheader(f"🚦 最新交易日 ({df.index[-1].strftime('%Y-%m-%d')}) 各指標信號明細")

cols = st.columns(2)
for idx, (name, score, light, detail) in enumerate(latest_signals):
    col_target = cols[idx % 2]
    with col_target:
        badge_bg = "#ffebee" if score == 1 else ("#e8f5e9" if score == -1 else "#fffde7")
        badge_border = "#ef5350" if score == 1 else ("#66bb6a" if score == -1 else "#fbc02d")
        score_text = f"+1 (多)" if score == 1 else (f"-1 (空)" if score == -1 else "0 (中立)")
        
        st.markdown(f"""
        <div style="background:{badge_bg}; border:1px solid {badge_border}; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <strong style="font-size:1.05rem;">{name}</strong>
                <span style="font-weight:bold; font-size:0.95rem;">{light} | 權重：{score_text}</span>
            </div>
            <div style="color:#555; font-size:0.9rem;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

with st.expander("🔍 查看已計算的指標數據 (觀測區間前 5 筆與後 5 筆)"):
    display_df = df.copy()
    display_df.index = display_df.index.strftime('%Y-%m-%d')
    st.dataframe(pd.concat([display_df.head(5), display_df.tail(5)]))

# 將繪圖用的 DataFrame 索引轉為字串格式
plot_df = df.copy()
plot_df.index = plot_df.index.map(lambda x: x.strftime('%y-%m-%d'))

# ==========================================
# 5. 步驟 4：專業多圖層視覺化 (8 單位高度排版)
# ==========================================
st.header("Step 4: 綜合技術指標儀表板 (含 6 大指標圖表)")
with st.expander("📖 查看圖表排版設計說明"):
    st.markdown("""
    本圖表使用 Matplotlib 的 `add_subplot(8, 1, ...)` 將畫布切分為 8 個單位高度：
    - **區塊 1-3 (主圖)**：顯示 K 線、5/10/20 日均線與布林通道。
    - **區塊 4 (OBV & Volume)**：結合能量潮曲線與成交量柱狀圖 (雙 Y 軸)。
    - **區塊 5 (KDJ)**：K、D、J 三線交叉觀察。
    - **區塊 6 (MACD)**：DIF、MACD 指標及其紅綠柱狀圖。
    - **區塊 7 (RSI)**：觀察 RSI5 與 RSI10 是否觸及超買 (70) 或超賣 (30) 虛線區間。
    - **區塊 8 (BIAS)**：10日與 20日乖離率，及兩者差距的柱狀圖。
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

# 定義 X 軸刻度間隔
step = max(1, len(plot_df.index) // 10)
x_ticks_pos = range(0, len(plot_df.index), step)
x_ticks_labels = plot_df.index[::step]

# --- Ax1: K線 + 均線 + 布林帶 ---
ax1.set_xticks(x_ticks_pos)
ax1.set_xticklabels([]) # 隱藏重疊字體
mpf.candlestick2_ochl(ax1, plot_df['Open'], plot_df['Close'], plot_df['High'], plot_df['Low'], 
                       width=0.8, colorup='r', colordown='g', alpha=1)
ax1.plot(plot_df['SMA_5'].values, label='5日均線', color='cyan', lw=1)
ax1.plot(plot_df['SMA_10'].values, label='10日均線', color='purple', lw=1)
ax1.plot(plot_df['SMA_20'].values, label='20日均線', color='orange', lw=1)
ax1.plot(plot_df['upper_band'].values, label='布林上軌', color='g', ls=':', lw=1)
ax1.plot(plot_df['lower_band'].values, label='布林下軌', color='g', ls=':', lw=1)
ax1.legend(loc='upper left', fontsize='small')
ax1.set_title(f"【{stock_id}】綜合技術分析圖表 (最新總分: {total_score:+d})", fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.3)

# --- Ax2: OBV 與 成交量 ---
ax2.set_xticks(x_ticks_pos)
ax2.set_xticklabels([])
conditions = [
    plot_df['Close'] > plot_df['Close'].shift(1),
    plot_df['Close'] < plot_df['Close'].shift(1)
]
choices = ['r', 'g']
colors = np.select(conditions, choices, default='gray')
ax2.plot(plot_df['OBV'].values, color='purple', ls='--', label='OBV')
ax2.plot(plot_df['OBV_EMA'].values, color='orange', ls=':', label='OBV EMA(10)')
ax2_v = ax2.twinx()
ax2_v.bar(range(len(plot_df)), plot_df['Volume'], color=colors, alpha=0.25, width=0.8)
ax2.set_title("OBV 能量潮 & 成交量")
ax2.legend(loc='upper left', fontsize='small')
ax2.grid(True, linestyle='--', alpha=0.3)

# --- Ax3: KDJ ---
ax3.plot(plot_df['K'].values, label='K線', color='cyan', lw=1)
ax3.plot(plot_df['D'].values, label='D線', color='purple', lw=1)
ax3.plot(plot_df['J'].values, label='J線', color='orange', ls='--')
ax3.axhline(80, color='r', ls=':', lw=0.8, alpha=0.5)
ax3.axhline(20, color='g', ls=':', lw=0.8, alpha=0.5)
ax3.set_xticks(x_ticks_pos)
ax3.set_xticklabels([])
ax3.set_title("KDJ 指標")
ax3.legend(loc='upper left', fontsize='small')
ax3.grid(True, linestyle='--', alpha=0.3)

# --- Ax4: MACD ---
ax4.plot(plot_df['DIF'].values, label='DIF', color='purple')
ax4.plot(plot_df['MACD'].values, label='MACD', color='skyblue')
m_hist_colors = np.where(plot_df['MACD Histogram'] >= 0, 'r', 'g')
ax4.bar(range(len(plot_df)), plot_df['MACD Histogram'], color=m_hist_colors, alpha=0.6)
ax4.axhline(0, color='gray', ls='--', lw=1)
ax4.set_xticks(x_ticks_pos)
ax4.set_xticklabels([])
ax4.set_title("MACD 指標")
ax4.legend(loc='upper left', fontsize='small')
ax4.grid(True, linestyle='--', alpha=0.3)

# --- Ax5: RSI ---
ax5.plot(plot_df['RSI5'].values, label='RSI5', color='cyan', lw=1)
ax5.plot(plot_df['RSI10'].values, label='RSI10', color='purple', lw=1)
ax5.axhline(70, color='r', ls='--', lw=0.8, alpha=0.5) # 超買線
ax5.axhline(30, color='g', ls='--', lw=0.8, alpha=0.5) # 超賣線
ax5.set_ylim(0, 100)
ax5.set_xticks(x_ticks_pos)
ax5.set_xticklabels([])
ax5.set_title("RSI 相對強弱指標")
ax5.legend(loc='upper left', fontsize='small')
ax5.grid(True, linestyle='--', alpha=0.3)

# --- Ax6: BIAS (乖離率差距柱狀圖) ---
ax6.plot(plot_df['BIAS10'].values, label='BIAS10', color='cyan', lw=1)
ax6.plot(plot_df['BIAS20'].values, label='BIAS20', color='purple', lw=1)
bias_diff_colors = np.where(plot_df['B10-B20'] >= 0, 'r', 'g')
ax6.bar(range(len(plot_df)), plot_df['B10-B20'], color=bias_diff_colors, alpha=0.6)
ax6.axhline(0, color='gray', ls='--', lw=1)
ax6.set_xticks(x_ticks_pos)
ax6.set_xticklabels(x_ticks_labels, rotation=30) # 最底部的圖表才顯示日期
ax6.set_title("BIAS 乖離率")
ax6.legend(loc='upper left', fontsize='small')
ax6.grid(True, linestyle='--', alpha=0.3)

# 渲染主圖表到網頁
st.pyplot(fig)

# ==========================================
# 6. 步驟 5：歷史紅綠燈綜合總分走勢圖
# ==========================================
st.header("Step 5: 歷史多空綜合總分走勢 (Historical Score Trend)")

fig_score, ax_score = plt.subplots(figsize=(14, 4), layout='constrained')
score_colors = np.where(df['Total_Score'] >= 0, 'crimson', 'forestgreen')
ax_score.bar(plot_df.index, df['Total_Score'], color=score_colors, alpha=0.7, width=0.6, label='每日多空總分')
ax_score.plot(plot_df.index, df['Total_Score'].rolling(3).mean(), color='darkblue', lw=1.5, ls='--', label='3日平滑趨勢')
ax_score.axhline(0, color='black', lw=1)
ax_score.axhline(3, color='red', ls=':', lw=0.8, label='多方強勢門檻 (+3)')
ax_score.axhline(-3, color='green', ls=':', lw=0.8, label='空方強勢門檻 (-3)')
ax_score.set_xticks(x_ticks_pos)
ax_score.set_xticklabels(x_ticks_labels, rotation=30)
ax_score.set_title(f"【{stock_id}】歷史每日技術指標多空總分累計走勢圖", fontsize=14)
ax_score.set_ylabel("多空總分 (-7 ~ +7)")
ax_score.legend(loc='upper left', fontsize='small')
ax_score.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig_score)

st.divider()
st.info("💡 **操作建議**：綜合指標總分高於 +3 分時代表多方趨勢明確；低於 -3 分時代表空方主導；在 -2 ~ +2 之間震盪時建議以區間震盪操作或減量因應。")