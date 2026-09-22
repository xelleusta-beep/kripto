import streamlit as st
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
import requests
import io
from sklearn.ensemble import RandomForestClassifier

# 1. Web Sayfası Başlığı ve Geniş Ekran Ayarı
st.set_page_config(layout="wide", page_title="Yapay Zeka Çoklu Otomasyon Paneli")
st.title("🤖 ML Kripto Backtest & Çoklu Canlı Alarm Yönetim Platformu")

# --- INITIAL SESSIONS (Oturum Belleği) ---
if "alarms" not in st.session_state:
    st.session_state.alarms = [] 
if "global_trade_history" not in st.session_state:
    st.session_state.global_trade_history = []

# --- TELEGRAM SİNYAL GÖNDERME FONKSİYONU ---
def send_telegram_signal(token, chat_id, message):
    try:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# 2. Yan Panel (Sidebar) - Kullanıcı Seçimleri ve Zaman Ayarları
st.sidebar.header("⚙️ 1. Telegram Bağlantı Ayarları")
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", help="BotFather'dan aldığınız token")
chat_id = st.sidebar.text_input("Telegram Chat ID", type="password", help="Userinfo botundan aldığınız ID")

st.sidebar.markdown("---")
st.sidebar.header("🔍 2. Kripto Seçimi & Backtest Ayarları")

ticker = st.sidebar.selectbox(
    "Kripto Para Seçin", 
    ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "AVAX/USDT", "LINK/USDT", "BNB/USDT", "ADA/USDT"]
)

interval_label = st.sidebar.selectbox(
    "Veri Sıklığı (Grafik Mum Tipi)", 
    ["1 Dakika", "5 Dakika", "15 Dakika", "1 Saat", "1 Gün"]
)

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
        "is_active": True,
        "last_signal": None,
        "last_price": 0.0
    }
    st.session_state.alarms.append(new_alarm)
    st.sidebar.success(f"Başarılı! {ticker} alarm havuzuna eklendi.")

# --- ARKA PLAN VERİ VE ML İŞLEME ALANI ---
symbol = ticker.replace("/", "-")
if "USDT" in symbol:
    symbol = symbol.replace("USDT", "USD")

try:
    yf_data = vbt.YFData.download(symbol, period=period_mapping[time_period], interval=interval_mapping[interval_label])
    df = pd.DataFrame({'Open': yf_data.get('Open'), 'High': yf_data.get('High'), 'Low': yf_data.get('Low'), 'Close': yf_data.get('Close')})
    
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    df['Return'] = df['Close'].pct_change()
    df['RSI'] = vbt.RSI.run(df['Close'], window=14).rsi
    df['SMA_20'] = vbt.MA.run(df['Close'], window=20).ma
    df['Price_to_SMA'] = df['Close'] / df['SMA_20']
    df['Signal_Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    X = df[['Return', 'RSI', 'Price_to_SMA']]
    y = df['Signal_Target']
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    split_index = int(len(X) * (train_size / 100))
    model.fit(X[:split_index], y[:split_index])
    df['Predicted_Signal'] = model.predict(X)

    current_price = float(df['Close'].iloc[-1])
    latest_signal = int(df['Predicted_Signal'].iloc[-1])

    portfolio = vbt.Portfolio.from_signals(df['Close'], entries=(df['Predicted_Signal'] == 1), exits=(df['Predicted_Signal'] == 0), fees=0.001)

    # --- 3 ANA SEKMELİ ÖN YÜZ TASARIMI ---
    tab1, tab2, tab3 = st.tabs(["📊 1. Gelişmiş Backtest Alanı", "🚨 2. Canlı Alarm Havuzu & Excel", "🕒 3. Global İşlem Günlüğü"])

    with tab1:
        st.write(f"### 📈 {ticker} Strateji Analiz Paneli")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 🕯️ İnteraktif Mum Grafiği")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker
            )])
            fig.update_layout(xaxis_rangeslider_visible=False, height=450, template="plotly_dark")
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.write("#### 📊 Geçmiş Dönem Performans Sonuçları")
            total_ret = portfolio.total_return() * 100
            st.metric(label="Yapay Zeka Toplam Net Kâr/Zarar", value=f"{total_ret:.2f}%", delta=f"{total_ret:.2f}%")
            
            st.write("##### 🛠️ Detaylı Backtest İstatistikleri")
            stats_df = pd.DataFrame(portfolio.stats(), columns=["Değer"]).astype(str)
            st.dataframe(stats_df, width="stretch")

    with tab2:
        st.write("### 🗃️ Tanımlı Yapay Zeka Alarmlarınız ve Canlı Bakiyeleri")
        
        # AKTİF ALARMLARIN OTOMATİK ML MOTORU TARAFINDAN GÜNCELLENMESİ
        for alarm in st.session_state.alarms:
            if not alarm["is_active"]:
                continue
            
            alm_symbol = alarm["ticker"].replace("/", "-").replace("USDT", "USD")
            try:
                a_data = vbt.YFData.download(alm_symbol, period=period_mapping[alarm["period"]], interval=interval_mapping[alarm["interval"]])
                a_df = pd.DataFrame({'Close': a_data.get('Close')})
                if isinstance(a_df['Close'], pd.DataFrame): a_df['Close'] = a_df['Close'].iloc[:, 0]
                
                a_df['Return'] = a_df['Close'].pct_change()
                a_df['RSI'] = vbt.RSI.run(a_df['Close'], window=14).rsi
                a_df['SMA_20'] = vbt.MA.run(a_df['Close'], window=20).ma
                a_df['Price_to_SMA'] = a_df['Close'] / a_df['SMA_20']
                a_df['Signal_Target'] = (a_df['Close'].shift(-1) > a_df['Close']).astype(int)
                a_df.dropna(inplace=True)
                
                a_X = a_df[['Return', 'RSI', 'Price_to_SMA']]
                a_model = RandomForestClassifier(random_state=42, n_estimators=50)
                a_model.fit(a_X, a_df['Signal_Target'])
                
                a_price = float(a_df['Close'].iloc[-1])
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
            except Exception:
                pass

        if not st.session_state.alarms:
            st.info("Havuzda alarm bulunmuyor. Yan taraftan coin seçip listeye ekleyebilirsiniz.")
        else:
            for idx, alm in enumerate(st.session_state.alarms):
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.write(f"**#{alm['id']}**")
                c2.write(f"💱 {alm['ticker']}")
                c3.write(f"⏱️ {alm['interval']}")
                live_val = alm['balance'] if alm['balance'] > 0 else (alm['crypto_amount'] * alm['last_price'])
                c4.write(f"💰 Kasa: **${live_val:,.2f}**")
                
                if alm['last_signal'] == 1:
                    c5.success("🤖 Sinyal: AL")
                elif alm['last_signal'] == 0:
                    c5.error("🤖 Sinyal: SAT")
                else:
                    c5.warning("⏳ Hesaplanıyor")
                    
                alm["is_active"] = c6.toggle("Açık", value=alm["is_active"], key=f"tgl_{idx}")
                if c7.button("🗑️", key=f"del_{idx}"):
                    st.session_state.alarms.pop(idx)
                    st.rerun()
            
            st.markdown("---")
            
            # --- HATALI IF ELSE BLOKLARI TEK SATIRA İNDİRGENEREK KESİN ÇÖZÜM SAĞLANDI ---
            report_data = []
            for a in st.session_state.alarms:
                calculated_balance = a['balance'] if a['balance'] > 0 else (a['crypto_amount'] * a['last_price'])
