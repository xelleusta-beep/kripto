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

# --- YARDIMCI FONKSİYONLAR ---
def send_telegram_signal(token, chat_id, message):
    try:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

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

def run_ml_and_backtest(df_input, train_ratio):
    try:
        working_df = df_input.copy()
        working_df['Return'] = working_df['Close'].pct_change()
        working_df['RSI'] = vbt.RSI.run(working_df['Close'], window=14).rsi
        working_df['SMA_20'] = vbt.MA.run(working_df['Close'], window=20).ma
        working_df['Price_to_SMA'] = working_df['Close'] / working_df['SMA_20']
        working_df['Signal_Target'] = (working_df['Close'].shift(-1) > working_df['Close']).astype(int)
        working_df.dropna(inplace=True)

        if len(working_df) < 10:
            return None, None

        X = working_df[['Return', 'RSI', 'Price_to_SMA']]
        y = working_df['Signal_Target']
        
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        split_idx = int(len(X) * (train_ratio / 100))
        
        model.fit(X[:split_idx], y[:split_idx])
        working_df['Predicted_Signal'] = model.predict(X)

        portfolio = vbt.Portfolio.from_signals(
            working_df['Close'], 
            entries=(working_df['Predicted_Signal'] == 1), 
            exits=(working_df['Predicted_Signal'] == 0), 
            fees=0.001
        )
        return working_df, portfolio
    except:
        return None, None

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
    ["7 Gün", "30 Gün", "2 Ay", "1 Yıl", "3 Year"],
    index=1 if "Dakika" in interval_label else 3
)

# Yahoo finance period kelime uyumu düzeltmesi
safe_period_str = "3y" if time_period == "3 Year" else period_mapping.get(time_period, "30d")

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
raw_df = get_crypto_data(symbol, safe_period_str, interval_mapping[interval_label])

if raw_df.empty or len(raw_df) < 20:
    st.error("Seçili borsa verisi yüklenemedi. Lütfen zaman ayarlarını değiştirin.")
else:
    processed_df, portfolio = run_ml_and_backtest(raw_df, train_size)
    
    if processed_df is not None and portfolio is not None:
        
        # --- CANLI ALARMLARI GÜNCELLEME DÖNGÜSÜ ---
        for alarm in st.session_state.alarms:
            if not alarm["is_active"]:
                continue
            
            alm_symbol = alarm["ticker"].replace("/", "-").replace("USDT", "USD")
            alm_period_safe = "3y" if alarm["period"] == "3 Year" else period_mapping.get(alarm["period"], "30d")
            alarm_raw = get_crypto_data(alm_symbol, alm_period_safe, interval_mapping[alarm["interval"]])
            
            if not alarm_raw.empty and len(alarm_raw) > 20:
                try:
                    a_proc, _ = run_ml_and_backtest(alarm_raw, 80)
                    if a_proc is not None:
                        a_price = float(a_proc['Close'].iloc[-1])
                        a_signal = int(a_proc['Predicted_Signal'].iloc[-1])
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
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### 🕯️ İnteraktif Mum Grafiği")
                fig = go.Figure(data=[go.Candlestick(
                    x=processed_df.index, open=processed_df['Open'], high=processed_df['High'], low=processed_df['Low'], close=processed_df['Close'], name=ticker
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
                
            st.markdown("---")
            st.write("### 📜 Yapay Zekanın Geçmiş Tüm İşlemlerinin Detaylı Listesi (Trade Logs)")
            
            try:
                # KESİN ÇÖZÜM: records_df yerine to_df() fonksiyonu kullanıldı
                trades_df = portfolio.trades.to_df()
                if not trades_df.empty:
                    entry_dates = processed_df.index[trades_df['Entry Index']]
                    exit_dates = processed_df.index[trades_df['Exit Index']]
                    
                    backtest_logs = pd.DataFrame()
                    backtest_logs["İşlem ID"] = trades_df['Trade ID'] + 1
                    backtest_logs["Giriş Tarihi"] = entry_dates.strftime('%Y-%m-%d %H:%M')
