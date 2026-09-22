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

# --- 1. TELEGRAM BİLDİRİM FONKSİYONU ---
def send_telegram_signal(token, chat_id, message):
    try:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

# --- 2. GÜVENLİ VERİ ÇEKME FONKSİYONU ---
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

# --- 3. TEMİZ VE HAFİF PANDAS BACKTEST MATEMATİK MOTORU ---
def compute_strategy_performance(df_input, train_ratio):
    try:
        working_df = df_input.copy()
        working_df['Return'] = working_df['Close'].pct_change()
        working_df['RSI'] = vbt.RSI.run(working_df['Close'], window=14).rsi
        working_df['SMA_20'] = vbt.MA.run(working_df['Close'], window=20).ma
        working_df['Price_to_SMA'] = working_df['Close'] / working_df['SMA_20']
        working_df['Signal_Target'] = (working_df['Close'].shift(-1) > working_df['Close']).astype(int)
        working_df.dropna(inplace=True)

        if len(working_df) < 15:
            return None, 0.0, 10000.0, pd.DataFrame(), 0

        X = working_df[['Return', 'RSI', 'Price_to_SMA']]
        y = working_df['Signal_Target']
        
        model = RandomForestClassifier(random_state=42, n_estimators=50)
        split_idx = int(len(X) * (train_ratio / 100))
        model.fit(X[:split_idx], y[:split_idx])
        working_df['Predicted_Signal'] = model.predict(X)

        trade_logs = []
        in_pos = False
        ent_price = 0.0
        ent_date = None
        init_cash = 10000.0
        cash = init_cash
        units = 0.0

        for i in range(len(working_df)):
            c_date = working_df.index[i]
            c_price = float(working_df['Close'].iloc[i])
            c_sig = int(working_df['Predicted_Signal'].iloc[i])

            if c_sig == 1 and not in_pos:
                units = (cash / c_price) * 0.999
                ent_price = c_price
                ent_date = c_date
                cash = 0.0
                in_pos = True
            elif c_sig == 0 and in_pos:
                cash = (units * c_price) * 0.999
                pnl = cash - init_cash if len(trade_logs) == 0 else cash - (init_cash + sum([t['Net Kâr/Zarar ($)'] for t in trade_logs]))
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
    except:
        return None, 0.0, 10000.0, pd.DataFrame(), 0

# --- 4. GRAFİK OLUŞTURMA FONKSİYONU ---
def build_candlestick_chart(data_df, label_text):
    try:
        candles = go.Candlestick(
            x=data_df.index, 
            open=data_df['Open'], 
            high=data_df['High'], 
            low=data_df['Low'], 
            close=data_df['Close'], 
            name=label_text
        )
        fig_obj = go.Figure(data=[candles])
        fig_obj.update_layout(xaxis_rangeslider_visible=False, height=450, template="plotly_dark")
        return fig_obj
    except:
        return None

# --- 5. CANLI ALARMLARI İŞLEYEN FONKSİYON ---
def process_live_alarms(b_token, c_id, p_mapping, i_mapping):
    if not st.session_state.alarms:
        return

    for alarm in st.session_state.alarms:
        if not alarm["is_active"]:
            continue
        
        alm_symbol = alarm["ticker"].replace("/", "-").replace("USDT", "USD")
        alarm_raw = get_crypto_data(alm_symbol, p_mapping.get(alarm["period"], "30d"), i_mapping.get(alarm["interval"], "1h"))
        
        if alarm_raw.empty or len(alarm_raw) < 20:
            continue
            
        try:
            _, _, _, _, a_signal = compute_strategy_performance(alarm_raw, 80)
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
                    msg = f"⚡ *ALARM:* {alarm['ticker']} ({alarm['interval']})\n👉 {action}\n💰 Güncel Kasa: ${cur_val:,.2f}"
                    send_telegram_signal(b_token, c_id, msg)
                    st.session_state.global_trade_history.append(f"[{alarm['ticker']}] {action} | Kasa: ${cur_val:,.2f}")
        except:
            pass

# --- 6. GÜVENLİ VE HİZALANMIŞ ANA PROGRAM KATMANI ---
def main():
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
        index=3
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
        st.sidebar.success(f"Başarılı! {ticker} havuzunuza eklendi.")

    # İşlem Akışı
    symbol = ticker.replace("/", "-").replace("USDT", "USD")
    raw_df = get_crypto_data(symbol, period_mapping[time_period], interval_mapping[interval_label])

    if raw_df.empty or len(raw_df) < 20:
        st.error("Seçili borsa verisi yüklenemedi. Lütfen yan panelden zaman ayarlarını değiştirin.")
        return

    processed_df, total_net_return_pct, final_wallet_value, backtest_logs, latest_signal = compute_strategy_performance(raw_df, train_size)
    
    if processed_df is None:
        st.error("Strateji hesaplamaları çalıştırılamadı.")
        return
