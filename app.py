import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import io
import ccxt
import warnings
from streamlit_autorefresh import st_autorefresh
warnings.filterwarnings('ignore')

# Sayfa ayarları - Kodun en üstünde olmalıdır
st.set_page_config(layout="wide", page_title="Yapay Zeka Çoklu Otomasyon Paneli")

# 60 saniyede bir sayfayı otomatik yeniler (Canlı alarmların arka planda çalışması için)
st_autorefresh(interval=60000, key="datarefresh")

# Session State Tanımlamaları
if "alarms" not in st.session_state:
    st.session_state.alarms = []
if "global_trade_history" not in st.session_state:
    st.session_state.global_trade_history = []

# --- 1. TELEGRAM BİLDİRİM FONKSiyonu ---
def send_telegram_signal(token, chat_id, message):
    if not token or not chat_id:
        return False
    try:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

# --- 2. GÜVENLİ VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=30)  # Sunucu kilitlenmelerini önlemek için 30 saniyelik önbellek
def get_crypto_data(symbol_name, prd_days, inv_str):
    try:
        exchange = ccxt.mexc({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 15000
        })
        limit_mapping = {"7 Gün": 200, "30 Gün": 750, "2 Ay": 1000, "1 Yıl": 1000, "3 Yıl": 1500}
        safe_limit = limit_mapping.get(prd_days, 500)
        ohlcv = exchange.fetch_ohlcv(symbol_name, timeframe=inv_str, limit=safe_limit)

        if ohlcv is not None and len(ohlcv) > 20:
            df_res = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df_res['Timestamp'] = pd.to_datetime(df_res['Timestamp'], unit='ms')
            df_res.set_index('Timestamp', inplace=True)
            return df_res[['Open', 'High', 'Low', 'Close']]
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# --- SAF PANDAS İLE RSI HESAPLAMA (vectorbt bağımlılığı kaldırıldı) ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# --- 3. PANDAS BACKTEST MATEMATİK MOTORU ---
def compute_strategy_performance(df_input, train_ratio):
    try:
        if df_input.empty or len(df_input) < 25:
            return None, 0.0, 10000.0, pd.DataFrame(), 0

        from sklearn.ensemble import RandomForestClassifier
        working_df = df_input.copy()
        working_df['Return'] = working_df['Close'].pct_change()
        
        # Saf matematiksel indikatör hesaplama
        working_df['RSI'] = compute_rsi(working_df['Close'], 14)
        working_df['SMA_20'] = working_df['Close'].rolling(window=20).mean()
        
        working_df['Price_to_SMA'] = working_df['Close'] / working_df['SMA_20']
        working_df['Signal_Target'] = (working_df['Close'].shift(-1) > working_df['Close']).astype(int)
        working_df.dropna(inplace=True)

        if len(working_df) < 10:
            return None, 0.0, 10000.0, pd.DataFrame(), 0

        X = working_df[['Return', 'RSI', 'Price_to_SMA']]
        y = working_df['Signal_Target']

        model = RandomForestClassifier(random_state=42, n_estimators=50)
        split_idx = int(len(X) * (train_ratio / 100))
        if split_idx == 0 or split_idx >= len(X):
            split_idx = int(len(X) * 0.8)

        model.fit(X[:split_idx], y[:split_idx])
        working_df['Predicted_Signal'] = model.predict(X)

        trade_logs = []
        in_pos = False
        ent_price = 0.0
        ent_date = None
        init_cash = 10000.0
        cash = init_cash
        units = 0.0
        last_entry_cost = 0.0

        for i in range(len(working_df)):
            c_date = working_df.index[i]
            c_price = float(working_df['Close'].iloc[i])
            c_sig = int(working_df['Predicted_Signal'].iloc[i])

            if c_sig == 1 and not in_pos:
                last_entry_cost = cash
                units = (cash / c_price) * 0.999
                ent_price = c_price
                ent_date = c_date
                cash = 0.0
                in_pos = True
            elif c_sig == 0 and in_pos:
                cash = (units * c_price) * 0.999
                pnl = cash - last_entry_cost 
                ret_pct = ((c_price - ent_price) / ent_price) * 100
                trade_logs.append({
                    "İşlem ID": len(trade_logs) + 1,
                    "Giriş Tarihi": ent_date.strftime('%Y-%m-%d %H:%M'),
                    "Çıkış Tarihi": c_date.strftime('%Y-%m-%d %H:%M'),
                    "Giriş Fiyatı ($)": round(ent_price, 4),
                    "Çıkış Fiyatı ($)": round(c_price, 4),
                    "Miktar (Adet)": round(units, 6),
                    "Net Kâr/Zarar ($)": round(pnl, 2),
                    "Getiri (%)": f"{ret_pct:.2f}%"
                })
                units = 0.0
                in_pos = False

        c_last_price = float(working_df['Close'].iloc[-1])
        final_val = cash if not in_pos else (units * c_last_price)
        total_ret_pct = ((final_val - init_cash) / init_cash) * 100
        latest_signal_out = int(working_df['Predicted_Signal'].iloc[-1])

        return working_df, total_ret_pct, final_val, pd.DataFrame(trade_logs), latest_signal_out
    except Exception as e:
        st.error(f"Hesaplama hatası: {str(e)}")
        return None, 0.0, 10000.0, pd.DataFrame(), 0

# --- 4. GRAFİK OLUŞTURMA FONKSİYONU ---
def build_candlestick_chart(data_df, label_text):
    try:
        if data_df.empty:
            return None
        candles = go.Candlestick(
            x=data_df.index,
            open=data_df['Open'],
            high=data_df['High'],
            low=data_df['Low'],
            close=data_df['Close'],
            name=label_text
        )
        fig_obj = go.Figure(data=[candles])
        fig_obj.update_layout(
            xaxis_rangeslider_visible=False,
            height=450,
            template="plotly_dark",
            title=label_text
        )
        fig_obj.update_xaxes(title_text='Zaman')
        fig_obj.update_yaxes(title_text='Fiyat (USDT)')
        return fig_obj
    except Exception:
        return None

# --- 5. CANLI ALARMLARI İŞLEYEN FONKSİYON ---
def process_live_alarms(b_token, c_id):
    if not st.session_state.alarms:
        return
    for alarm in st.session_state.alarms:
        if not alarm["is_active"]:
            continue
        alarm_raw = get_crypto_data(alarm["ticker"], alarm["period"], alarm["interval"])
        if alarm_raw.empty or len(alarm_raw) < 25:
            continue
        try:
            res = compute_strategy_performance(alarm_raw, 80)
            if res is not None:
                _, _, _, _, a_signal = res
                a_price = float(alarm_raw['Close'].iloc[-1])
                alarm["last_price"] = a_price
                if alarm["last_signal"] != a_signal:
                    action = ""
                    if a_signal == 1 and alarm["balance"] > 0:
                        alarm["crypto_amount"] = alarm["balance"] / a_price
                        action = f"🟢 ALIM YAPILDI: {alarm['crypto_amount']:.4f} adet."
                        alarm["balance"] = 0.0
                    elif a_signal == 0 and alarm["crypto_amount"] > 0:
                        alarm["balance"] = alarm["crypto_amount"] * a_price
                        action = f"🔴 SATIM YAPILDI: Nakte geçildi."
                        alarm["crypto_amount"] = 0.0
                    alarm["last_signal"] = a_signal
                    if action and b_token and c_id:
                        cur_val = alarm["balance"] if alarm["balance"] > 0 else (alarm["crypto_amount"] * a_price)
                        msg = f"⚡ *MEXC ALARM:* {alarm['ticker']} ({alarm['interval']})\n👉 {action}\n💰 Güncel Kasa: ${cur_val:,.2f}"
                        send_telegram_signal(b_token, c_id, msg)
                        st.session_state.global_trade_history.append(f"[{alarm['ticker']}] {action} | Kasa: ${cur_val:,.2f}")
        except Exception as e:
            st.session_state.global_trade_history.append(f"Alarm işleme hatası: {str(e)}")

# --- 6. ARAYÜZ BİLEŞENLERİ PANELİ ---
st.sidebar.header("⚙️ 1. Telegram Bağlantı Ayarları")
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", key="tg_token")
chat_id = st.sidebar.text_input("Telegram Chat ID", type="password", key="tg_chat_id")

st.sidebar.markdown("---")
st.sidebar.header("🔍 2. Kripto Seçimi & Backtest Ayarları")

crypto_list = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    "DOGE/USDT", "SHIB/USDT", "NEAR/USDT", "SUI/USDT", "LTC/USDT"
]

ticker = st.sidebar.selectbox("Kripto Para Seçin (MEXC Canlı)", crypto_list)
interval_label = st.sidebar.selectbox("Veri Sıklığı (Grafik Mum Tipi)", ["1 Saat", "1 Gün"])

interval_mapping = {"1 Saat": "1h", "1 Gün": "1d"}
time_period = st.sidebar.selectbox("Geçmiş Test Süresi", ["7 Gün", "30 Gün", "2 Ay", "1 Yıl", "3 Yıl"], index=3)
train_size = st.sidebar.slider("Yapay Zeka Eğitim Verisi Oranı (%)", 50, 90, 80)

st.sidebar.markdown("---")
st.sidebar.header("🚨 3. Alarm Oluşturma")
alarm_init_balance = st.sidebar.number_input("Bu Alarma Özel Başlangıç Bakiyesi ($)", min_value=10.0, value=1000.0, step=100.0)

if st.sidebar.button("🚨 SEÇİLİ COİNİ ALARMLARA EKLE", use_container_width=True):
    if not any(a["ticker"] == ticker and a["interval"] == interval_mapping[interval_label] for a in st.session_state.alarms):
        new_alarm = {
            "id": len(st.session_state.alarms) + 1,
