import streamlit as st
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
import requests
import io
from sklearn.ensemble import RandomForestClassifier

# 1. Web Sayfası Başlığı ve Geniş Ekran Ayarı
st.set_page_config(layout="wide", page_title="Yapay Zeka Çoklu Otomasyon Paneli")
st.title("🤖 Çoklu ML Alarm Yönetimi & Canlı Portföy Paneli")

# --- INITIAL SESSIONS (Gelişmiş Oturum Belleği) ---
if "alarms" not in st.session_state:
    st.session_state.alarms = [] # Eklenen tüm dinamik alarmları tutar
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

# 2. Yan Panel (Sidebar) - Genel ve Telegram Bağlantı Ayarları
st.sidebar.header("⚙️ 1. Telegram Bağlantı Ayarları")
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", help="BotFather'dan aldığınız token")
chat_id = st.sidebar.text_input("Telegram Chat ID", type="password", help="Userinfo botundan aldığınız ID")

st.sidebar.markdown("---")
st.sidebar.header("🚨 2. Yeni Alarm Tanımlama Paneli")

# Alarm Oluşturma Form Elemanları
alarm_ticker = st.sidebar.selectbox("Kripto Para", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
alarm_interval = st.sidebar.selectbox("Veri Sıklığı (Mum)", ["1 Dakika", "5 Dakika", "15 Dakika", "1 Saat", "1 Gün"])
alarm_period = st.sidebar.selectbox("Geçmiş Test Süresi", ["7 Gün", "30 Gün", "2 Ay", "1 Yıl", "3 Yıl"], index=1)
alarm_init_balance = st.sidebar.number_input("Bu Alarma Özel Bakiye ($)", min_value=10.0, value=1000.0, step=100.0)

# Butona basıldığında yeni alarmı session_state havuzuna ekler
if st.sidebar.button("🚨 YENİ ALARM EKLE", use_container_width=True):
    new_alarm = {
        "id": len(st.session_state.alarms) + 1,
        "ticker": alarm_ticker,
        "interval": alarm_interval,
        "period": alarm_period,
        "balance": float(alarm_init_balance),
        "crypto_amount": 0.0,
        "is_active": True,
        "last_signal": None,
        "last_price": 0.0
    }
    st.session_state.alarms.append(new_alarm)
    st.sidebar.success(f"{alarm_ticker} için {alarm_interval} alarmı listeye eklendi!")

# --- ARKA PLAN: TÜM ALARMLARI OTOMATİK İŞLEYEN ML MOTORU ---
interval_mapping = {"1 Dakika": "1m", "5 Dakika": "5m", "15 Dakika": "15m", "1 Saat": "1h", "1 Gün": "1d"}
period_mapping = {"7 Gün": "7d", "30 Gün": "30d", "2 Ay": "2mo", "1 Yıl": "1y", "3 Yıl": "3y"}

# Eklenmiş olan tüm alarmları tek tek dönerek ML tahminlerini yapar ve bakiyeleri simüle eder
for alarm in st.session_state.alarms:
    if not alarm["is_active"]:
        continue
        
    symbol = alarm["ticker"].replace("/", "-")
    if "USDT" in symbol:
        symbol = symbol.replace("USDT", "USD")
        
    try:
        # ML için dinamik veri çekimi
        yf_data = vbt.YFData.download(symbol, period=period_mapping[alarm["period"]], interval=interval_mapping[alarm["interval"]])
        df = pd.DataFrame({'Close': yf_data.get('Close')})
        
        for col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
                
        if len(df) > 20:
            df['Return'] = df['Close'].pct_change()
            df['RSI'] = vbt.RSI.run(df['Close'], window=14).rsi
            df['SMA_20'] = vbt.MA.run(df['Close'], window=20).ma
            df['Price_to_SMA'] = df['Close'] / df['SMA_20']
            df['Signal_Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df.dropna(inplace=True)
            
            # ML Model Eğitimi (RandomForest Arka Planda Aktif Çalışıyor)
            X = df[['Return', 'RSI', 'Price_to_SMA']]
            y = df['Signal_Target']
            model = RandomForestClassifier(random_state=42, n_estimators=50)
            model.fit(X, y)
            
            current_price = float(df['Close'].iloc[-1])
            latest_signal = int(model.predict(X.iloc[[-1]])[0])
            
            alarm["last_price"] = current_price
            
            # --- SİNYAL DEĞİŞİMİ VE BAKİYE GÜNCELLEME ALGORİTMASI ---
            if alarm["last_signal"] != latest_signal:
                action_text = ""
                
                # AL Sinyali Geldiğinde Nakit ile Kripto Alımı Simülasyonu
                if latest_signal == 1 and alarm["balance"] > 0:
                    alarm["crypto_amount"] = alarm["balance"] / current_price
                    action_text = f"🟢 İŞLEM AÇILIŞI (AL): {alarm['crypto_amount']:.4f} adet alındı."
                    alarm["balance"] = 0.0
                    
                # SAT Sinyali Geldiğinde Kriptoları Nakite Çevirme Simülasyonu
                elif latest_signal == 0 and alarm["crypto_amount"] > 0:
                    alarm["balance"] = alarm["crypto_amount"] * current_price
                    action_text = f"🔴 İŞLEM KAPANIŞI (SAT): Nakite geçildi."
                    alarm["crypto_amount"] = 0.0
                
                alarm["last_signal"] = latest_signal
                
                # Değişiklik varsa Telegram Bildirimi Tetikle
                if action_text and bot_token and chat_id:
                    current_wallet_value = alarm["balance"] if alarm["balance"] > 0 else (alarm["crypto_amount"] * current_price)
                    msg = (
                        f"⚡ *ALARM SİNYAL TETİKLENDİ*\n\n"
                        f"▪️ *Varlık:* {alarm['ticker']}\n"
                        f"▪️ *Periyot/Sıklık:* {alarm['period']} / {alarm['interval']}\n"
                        f"▪️ *Durum:* {action_text}\n"
                        f"▪️ *Fiyat:* ${current_price:,.2f}\n"
                        f"▪️ *Güncel Güncel Bakiye:* ${current_wallet_value:,.2f}\n"
                    )
                    send_telegram_signal(bot_token, chat_id, msg)
                    st.session_state.global_trade_history.append(f"[{alarm['ticker']} - {alarm['interval']}] {action_text} Fiyat: ${current_price:,.2f} | Portföy Değeri: ${current_wallet_value:,.2f}")
    except Exception:
        pass

# 3. Ön Yüz Tasarımı: Sekmeli Yapı
tab1, tab2 = st.tabs(["🚨 Alarm Havuzu ve Excel Yönetimi", "🕒 Global İşlem Günlüğü"])

with tab1:
    st.write("### 🗃️ Tanımlı Yapay Zeka Alarmlarınız")
    
    if not st.session_state.alarms:
        st.info("Henüz eklenmiş bir alarm bulunmuyor. Yan paneli kullanarak ilk alarmınızı ekleyebilirsiniz.")
    else:
        # Her alarm için ekranda interaktif yönetim satırları oluşturma
        for idx, alm in enumerate(st.session_state.alarms):
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 2, 2, 2, 3, 2, 1])
            
            c1.write(f"**#{alm['id']}**")
            c2.write(f"💱 {alm['ticker']}")
            c3.write(f"⏱️ {alm['interval']} ({alm['period']})")
            
            # Anlık toplam cüzdan değerinin hesabı
            live_val = alm['balance'] if alm['balance'] > 0 else (alm['crypto_amount'] * alm['last_price'])
            c4.write(f"💰 Bakiye: **${live_val:,.2f}**")
            
            # Sinyal Durum Rozeti
            if alm['last_signal'] == 1:
                c5.success("🤖 Sinyal: AL (Pozisyonda)")
            elif alm['last_signal'] == 0:
                c5.error("🤖 Sinyal: SAT (Nakit)")
            else:
                c5.warning("⏳ Sinyal Hesaplanıyor...")
                
            # Aktif / Pasif Açma Kapatma Anahtarı
            alm["is_active"] = c6.toggle("Aktif", value=alm["is_active"], key=f"tgl_{idx}")
            
            # Alarmı Kaldırma Butonu
            if c7.button("🗑️", key=f"del_{idx}"):
                st.session_state.alarms.pop(idx)
                st.contexthint = "Alarm silindi"
                st.rerun()
        
        st.markdown("---")
        st.write("### 📊 Excel Raporlama Alanı")
        
        # Alarmları toplu liste tablosuna dönüştürme
        report_data = []
        for alm in st.session_state.alarms:
            live_val = alm['balance'] if alm['balance'] > 0 else (alm['crypto_amount'] * alm['last_price'])
            report_data.append({
                "Alarm ID": alm["id"],
                "Kripto Para": alm["ticker"],
                "Veri Sıklığı": alm["interval"],
                "Test Süresi": alm["period"],
                "Güncel Bakiye ($)": round(live_val, 2),
                "Durum": "AKTİF" if alm["is_active"] else "PASİF",
                "Son Sinyal": "AL" if alm["last_signal"] == 1 else "SAT" if alm["last_signal"] == 0 else "Belirsiz"
            })
        
        df_report = pd.DataFrame(report_data)
        
        # Site üzerinde indirmeden görme alanı
        st.dataframe(df_report, width="stretch")
        
        # Excel İndirme Butonu Optimizasyonu
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_report.to_excel(writer, index=False, sheet_name='Alarmlar Raporu')
        
        st.download_button(
            label="📥 ALARMLARI EXCEL OLARAK İNDİR",
            data=buffer.getvalue(),
            file_name="kripto_yapay_zeka_alarmlar.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

with tab2:
    st.write("### 🕒 Gerçek Zamanlı Sinyal Tetiklenme ve Bakiye Geçmişi")
    if st.session_state.global_trade_history:
        for log in reversed(st.session_state.global_trade_history):
            st.info(log)
    else:
        st.info("Alarmlarda henüz bir sinyal değişimi veya bakiye güncellemesi yaşanmadı. Sinyal değiştikçe işlem logları buraya düşecektir.")
