import streamlit as st
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
import requests
import io
from sklearn.ensemble import RandomForestClassifier

# 1. Sayfa Konfigürasyonu
st.set_page_config(layout="wide", page_title="Yapay Zeka Çoklu Otomasyon Paneli")
st.title("🤖 ML Kripto Backtest & Çoklu Canlı Alarm Yönetim Platformu")

# --- OTURUM BELLEĞİ (SESSION STATE) ---
if "alarms" not in st.session_state:
    st.session_state.alarms = [] 
if "global_trade_history" not in st.session_state:
    st.session_state.global_trade_history = []

# --- TELEGRAM SİNYAL GÖNDERME ---
def send_telegram_signal(token, chat_id, message):
    try:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

# --- GÜVENLİ VERİ ÇEKME ---
def get_crypto_data(symbol_name, prd, inv):
    try:
        yf_data = vbt.YFData.download(symbol_name, period=prd, interval=inv)
        res_df = pd.DataFrame({
            'Open': yf_data.get('Open'), 
            'High': yf_data.get('High'), 
            'Low': yf_data.get('Low'), 
            'Close': yf_data.get('Close')
        })
        for col in res_df.columns:
            if isinstance(res_df[col], pd.DataFrame):
                res_df[col] = res_df[col].iloc[:, 0]
        return res_df
    except:
        return pd.DataFrame()

# 2. Yan Panel (Sidebar) Ayarları
st.sidebar.header("⚙️ 1. Telegram Bağlantı Ayarları")
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password")
chat_id = st.sidebar.text_input("Telegram Chat ID", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🔍 2. Kripto Seçimi & Backtest Ayarları")

crypto_list = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", 
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "WIF/USDT", "BONK/USDT",
    "NEAR/USDT", "SUI/USDT", "APT/USDT", "OP/USDT", "ARB/USDT",
    "LTC/USDT", "BCH/USDT", "UNI/USDT", "ATOM/USDT", "ICP/USDT",
    "FIL/USDT", "RNDR/USDT", "FET/USDT", "INJ/USDT", "TIA/USDT",
    "IMX/USDT", "STX/USDT", "GRT/USDT", "THETA/USDT", "FTM/USDT",
    "ALGO/USDT", "VET/USDT", "EGLD/USDT", "SAND/USDT", "MANA/USDT"
]

ticker = st.sidebar.selectbox("Kripto Para Seçin", crypto_list)
interval_label = st.sidebar.selectbox("Veri Sıklığı (Grafik Mum Tipi)", ["1 Dakika", "5 Dakika", "15 Dakika", "1 Saat", "1 Gün"])

interval_mapping = {"1 Dakika": "1m", "5 Dakika": "5m", "15 Dakika": "15m", "1 Saat": "1h", "1 Gün": "1d"}
period_mapping = {"7 Gün": "7d", "30 Gün": "30d", "2 Ay": "2mo", "1 Yıl": "1y", "3 Yıl": "3y"}

time_period = st.sidebar.selectbox(
    "Geçmiş Test Süresi", 
    ["7 Gün", "30 Gün", "2 Ay", "1 Yıl", "3 Yıl"],
    index=["7 Gün", "30 Gün", "2 Ay", "1 Yıl", "3 Yıl"].index(
        "7 Gün" if interval_label == "1 Dakika" else 
        "30 Gün" if "Dakika" in interval_label else 
        "2 Ay" if interval_label == "1 Saat" else "1 Yıl"
    )
)

train_size = st.sidebar.slider("Yapay Zeka Eğitim Verisi Oranı (%)", 50, 90, 80)

st.sidebar.markdown("---")
st.sidebar.header("🚨 3. Alarm Oluşturma")
alarm_init_balance = st.sidebar.number_input("Bu Alarma Özel Başlangıç Bakiyesi ($)", min_value=10.0, value=1000.0, step=100.0)

if st.sidebar.button("🚨 SEÇİLİ COİNİ ALARMLARA EKLE", use_container_width=True):
    new_alarm = {
        "id": len(st.session_state.alarms) + 1,
        "ticker": ticker,
        "interval": interval_label,
        "period": time_period,
        "balance": float(alarm_init_balance),
        "crypto_amount": 0.0,
        "last_signal": None,
        "last_price": 0.0,
        "is_active": True
    }
    st.session_state.alarms.append(new_alarm)
    st.sidebar.success(f"Başarılı! {ticker} alarm havuzuna eklendi.")

# --- ANA PROGRAM AKIŞI ---
symbol = ticker.replace("/", "-").replace("USDT", "USD")
df = get_crypto_data(symbol, period_mapping[time_period], interval_mapping[interval_label])

if df.empty or len(df) < 20:
    st.error("Seçili borsa verisi yüklenemedi. Lütfen yan panelden zaman ayarlarını değiştirin.")
else:
    # İndikatör Hesaplama (ML Şartları)
    df['Return'] = df['Close'].pct_change()
    df['RSI'] = vbt.RSI.run(df['Close'], window=14).rsi
    df['SMA_20'] = vbt.MA.run(df['Close'], window=20).ma
    df['Price_to_SMA'] = df['Close'] / df['SMA_20']
    df['Signal_Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # ML Model Eğitimi (Random Forest)
    X = df[['Return', 'RSI', 'Price_to_SMA']]
    y = df['Signal_Target']
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    split_index = int(len(X) * (train_size / 100))
    model.fit(X[:split_index], y[:split_index])
    df['Predicted_Signal'] = model.predict(X)

    current_price = float(df['Close'].iloc[-1])
    latest_signal = int(df['Predicted_Signal'].iloc[-1])

    # --- PANDAS TABANLI HAFİF VE ÇÖKMEZ BACKTEST HESAPLAYICI ---
    trade_logs_list = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    initial_cash = 10000.0
    cash = initial_cash
    crypto_units = 0.0

    for i in range(len(df)):
        current_date_idx = df.index[i]
        price_at_idx = float(df['Close'].iloc[i])
        sig = int(df['Predicted_Signal'].iloc[i])

        if sig == 1 and not in_position:
            # ALIM İŞLEMİ
            crypto_units = (cash / price_at_idx) * 0.999 # %0.1 Komisyon düşüldü
            entry_price = price_at_idx
            entry_date = current_date_idx
            cash = 0.0
            in_position = True
        elif sig == 0 and in_position:
            # SATIM İŞLEMİ
            cash = (crypto_units * price_at_idx) * 0.999 # %0.1 Komisyon düşüldü
            pnl_val = cash - initial_cash if len(trade_logs_list) == 0 else cash - (initial_cash + sum([t['Net Kâr/Zarar ($)'] for t in trade_logs_list]))
            ret_pct = ((price_at_idx - entry_price) / entry_price) * 100
            
            trade_logs_list.append({
                "İşlem ID": len(trade_logs_list) + 1,
                "Giriş Tarihi": entry_date.strftime('%Y-%m-%d %H:%M'),
                "Çıkış Tarihi": current_date_idx.strftime('%Y-%m-%d %H:%M'),
                "Giriş Fiyatı ($)": round(entry_price, 4),
                "Çıkış Fiyatı ($)": round(price_at_idx, 4),
                "Miktar (Adet)": round(crypto_units, 6),
                "Net Kâr/Zarar ($)": round(pnl_val, 2),
                "Getiri (%)": f"{ret_pct:.2f}%"
            })
            crypto_units = 0.0
            in_position = False

    # Son cüzdan değeri hesabı
    final_wallet_value = cash if not in_position else (crypto_units * current_price)
    total_net_return_pct = ((final_wallet_value - initial_cash) / initial_cash) * 100

    backtest_logs = pd.DataFrame(trade_logs_list)

    # --- CANLI ALARMLARI GÜNCELLEME DÖNGÜSÜ ---
    for alarm in st.session_state.alarms:
        if not alarm["is_active"]:
            continue
        
        alm_symbol = alarm["ticker"].replace("/", "-").replace("USDT", "USD")
        alarm_raw = get_crypto_data(alm_symbol, period_mapping[alarm["period"]], interval_mapping[alarm["interval"]])
        
        if not alarm_raw.empty and len(alarm_raw) > 20:
            try:
                alarm_raw['Return'] = alarm_raw['Close'].pct_change()
                alarm_raw['RSI'] = vbt.RSI.run(alarm_raw['Close'], window=14).rsi
                alarm_raw['SMA_20'] = vbt.MA.run(alarm_raw['Close'], window=20).ma
                alarm_raw['Price_to_SMA'] = alarm_raw['Close'] / alarm_raw['SMA_20']
                alarm_raw['Signal_Target'] = (alarm_raw['Close'].shift(-1) > alarm_raw['Close']).astype(int)
                alarm_raw.dropna(inplace=True)
                
                a_X = alarm_raw[['Return', 'RSI', 'Price_to_SMA']]
                a_model = RandomForestClassifier(random_state=42, n_estimators=50)
                a_model.fit(a_X, alarm_raw['Signal_Target'])
                
                a_price = float(alarm_raw['Close'].iloc[-1])
                a_signal = int(a_model.predict(a_X.iloc[[-1]]))
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
                    
                    if action and bot_token and chat_id:
                        cur_val = alarm["balance"] if alarm["balance"] > 0 else (alarm["crypto_amount"] * a_price)
                        msg = f"⚡ *ALARM:* {alarm['ticker']} ({alarm['interval']})\n👉 {action}\n💰 Güncel Kasa: ${cur_val:,.2f}"
                        send_telegram_signal(bot_token, chat_id, msg)
                        st.session_state.global_trade_history.append(f"[{alarm['ticker']}] {action} | Kasa: ${cur_val:,.2f}")
            except:
                pass

    # --- SEKMELİ ÖN YÜZ TASARIMI ---
    tab1, tab2, tab3 = st.tabs(["📊 1. Gelişmiş Backtest Alanı", "🚨 2. Canlı Alarm Havuzu & Excel", "🕒 3. Global İşlem Günlüğü"])

    with tab1:
        st.write(f"### 📈 {ticker} Strateji Analiz Paneli")
        
        st.write("#### 🕯️ İnteraktif Mum Grafiği")
        fig = go.Figure(data=[go.Candlestick(
