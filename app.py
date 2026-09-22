import streamlit as st
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. Web Sayfası Başlığı ve Geniş Ekran Ayarı
st.set_page_config(layout="wide", page_title="Yapay Zeka & Gelişmiş Zaman Ayarlı Trade Paneli")
st.title("🤖 ML Backtest & Esnek Zaman Aralıklı Trading Platformu")

# --- PAPER TRADING SESSİON STATE BAŞLATMA ---
if "balance" not in st.session_state:
    st.session_state.balance = 10000.0
if "crypto_amount" not in st.session_state:
    st.session_state.crypto_amount = 0.0
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

# 2. Yan Panel (Sidebar) - Kullanıcı Seçimleri
st.sidebar.header("🎛️ Strateji ve Zaman Ayarları")
ticker = st.sidebar.selectbox("Kripto Para Seçin", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

# YENİ ÖZELLİK: Veri Sıklığı (Interval) Seçici
interval_label = st.sidebar.selectbox(
    "Veri Sıklığı (Grafik Mum Tipi)", 
    ["1 Dakika", "5 Dakika", "15 Dakika", "1 Saat", "1 Gün", "1 Hafta"]
)

# Seçilen sıklığa göre Yahoo Finance interval kodları ve güvenli maksimum geçmiş süreleri
interval_mapping = {
    "1 Dakika": {"code": "1m", "default_period": "7d"},
    "5 Dakika": {"code": "5m", "default_period": "30d"},
    "15 Dakika": {"code": "15m", "default_period": "30d"},
    "1 Saat": {"code": "1h", "default_period": "2mo"},
    "1 Gün": {"code": "1d", "default_period": "1y"},
    "1 Hafta": {"code": "1wk", "default_period": "3y"}
}

chosen_interval = interval_mapping[interval_label]["code"]
safe_period = interval_mapping[interval_label]["default_period"]

# YENİ ÖZELLİK: Seçilen sıklığa uyumlu geçmiş test süresi
time_period = st.sidebar.selectbox(
    f"Geçmiş Test Süresi (Seçilen {interval_label} için Önerilen)", 
    ["7 Gün", "30 Gün", "2 Ay", "1 Yıl", "3 Yıl"],
    index=["7 Gün", "30 Gün", "2 Ay", "1 Yıl", "3 Yıl"].index(
        "7 Gün" if interval_label == "1 Dakika" else 
        "30 Gün" if "Dakika" in interval_label else 
        "2 Ay" if interval_label == "1 Saat" else 
        "1 Yıl" if interval_label == "1 Gün" else "3 Yıl"
    )
)

period_mapping = {"7 Gün": "7d", "30 Gün": "30d", "2 Ay": "2mo", "1 Yıl": "1y", "3 Yıl": "3y"}
selected_period = period_mapping[time_period]

train_size = st.sidebar.slider("Yapay Zeka Eğitim Verisi Oranı (%)", 50, 90, 80)

st.write(f"### 📈 {ticker} ({time_period} boyunca {interval_label} Verileri) Analizi")

# 3. Arka Plan: Dinamik Veri Çekme
symbol = ticker.replace("/", "-")
if "USDT" in symbol:
    symbol = symbol.replace("USDT", "USD")

try:
    # Kullanıcının seçtiği interval ve period değerlerine göre dinamik veri çekimi
    yf_data = vbt.YFData.download(symbol, period=selected_period, interval=chosen_interval)
    df = pd.DataFrame({
        'Open': yf_data.get('Open'),
        'High': yf_data.get('High'),
        'Low': yf_data.get('Low'),
        'Close': yf_data.get('Close')
    })

    # Çok katmanlı (MultiIndex) yapıları temizleme garantisi
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    if df.empty or len(df) < 30:
        st.error("Seçilen zaman aralığında yeterli veri bulunamadı! Lütfen daha kısa bir geçmiş süre veya daha yüksek bir grafik mum tipi seçin.")
    else:
        # 4. Feature Engineering: Yapay Zekaya İndikatör Öğretme
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = vbt.RSI.run(df['Close'], window=14).rsi
        df['SMA_20'] = vbt.MA.run(df['Close'], window=20).ma
        df['Price_to_SMA'] = df['Close'] / df['SMA_20']

        # Yapay Zekanın Hedefi: Bir sonraki mumda fiyat yükselecek mi?
        df['Signal_Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # 5. ML Model Çalıştırma Alanı (Aktif Çalışıyor)
        features = ['Return', 'RSI', 'Price_to_SMA']
        X = df[features]
        y = df['Signal_Target']

        model = RandomForestClassifier(random_state=42, n_estimators=100)
        split_index = int(len(X) * (train_size / 100))

        model.fit(X[:split_index], y[:split_index])
        df['Predicted_Signal'] = model.predict(X)

        current_price = float(df['Close'].iloc[-1])
        latest_signal = int(df['Predicted_Signal'].iloc[-1])

        # 6. Arka Plan: Vectorbt ile Backtest Simülasyonu
        portfolio = vbt.Portfolio.from_signals(
            df['Close'], 
            entries=(df['Predicted_Signal'] == 1), 
            exits=(df['Predicted_Signal'] == 0),
            fees=0.001
        )

        # 7. Ön Yüz Tasarımı: Sekmeli Yapı
        tab1, tab2 = st.tabs(["📊 Backtest ve Grafik Analizi", "💰 Canlı Paper Trading (Demo Hesap)"])

        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### 🕯️ İnteraktif Mum Grafiği")
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker
                )])
                fig.update_layout(xaxis_rangeslider_visible=False, height=450, template="plotly_dark")
                st.plotly_chart(fig, width="stretch")

            with col2:
                st.write("#### 📈 Strateji Geçmiş Performans Raporu")
                total_ret = portfolio.total_return() * 100
                st.metric(label="Yapay Zeka Toplam Net Getiri", value=f"{total_ret:.2f}%", delta=f"{total_ret:.2f}%")
                
                st.write("##### 🛠️ Detaylı Finansal İstatistikler")
                stats_df = pd.DataFrame(portfolio.stats(), columns=["Değer"]).astype(str)
                st.dataframe(stats_df, width="stretch")

        with tab2:
            st.write("### 💵 Sanal Portföy Yönetimi")
            
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("Sanal Nakit Bakiye", f"${st.session_state.balance:,.2f}")
            p_col2.metric(f"Varlık Miktarı ({ticker.split('/')[0]})", f"{st.session_state.crypto_amount:.4f}")
            
            total_asset_value = st.session_state.crypto_amount * current_price
            total_portfolio_value = st.session_state.balance + total_asset_value
            p_col3.metric("Toplam Portföy Değeri", f"${total_portfolio_value:,.2f}")
            
            if latest_signal == 1:
                p_col4.success("🤖 YAPAY ZEKA TAVSİYESİ: AL (BUY)")
            else:
                p_col4.error("🤖 YAPAY ZEKA TAVSİYESİ: SAT (SELL)")
                
            st.write(f"**Anlık Güncel Fiyat ({interval_label} Kapanışı):** ${current_price:,.2f}")
            
            trade_col1, trade_col2 = st.columns(2)
            
            with trade_col1:
                if st.button("🟢 Sanal BAKİYE İLE MAKSİMUM AL", use_container_width=True):
                    if st.session_state.balance > 10:
                        amount_to_buy = st.session_state.balance / current_price
                        st.session_state.crypto_amount += amount_to_buy
                        st.session_state.trade_history.append(f"ALINDI: {amount_to_buy:.4f} {ticker} @ ${current_price:,.2f} ({interval_label})")
                        st.session_state.balance = 0.0
                        st.rerun()
                    else:
                        st.warning("Yetersiz Nakit Bakiye!")
                        
            with trade_col2:
                if st.button("🔴 TÜM VARLIKLARI SANAL METAYA SAT", use_container_width=True):
                    if st.session_state.crypto_amount > 0:
                        gain = st.session_state.crypto_amount * current_price
                        st.session_state.balance += gain
                        st.session_state.trade_history.append(f"SATILDI: {st.session_state.crypto_amount:.4f} {ticker} @ ${current_price:,.2f} ({interval_label})")
                        st.session_state.crypto_amount = 0.0
                        st.rerun()
                    else:
                        st.warning("Satılacak Kripto Varlığınız Bulunmuyor!")

            st.write("##### 🕒 Sanal İşlem Geçmişi (Aktivite Günlüğü)")
            if st.session_state.trade_history:
                for log in reversed(st.session_state.trade_history):
                    st.text(log)
            else:
                st.info("Henüz simüle edilmiş bir işlem gerçekleştirilmedi.")

except Exception as e:
    st.error(f"Veri çekme hatası! Yahoo Finance bu zaman dilimini desteklemiyor olabilir. Hata ayrıntısı: {e}")
    st.info("İpucu: '1 Dakika' gibi kısa zaman dilimleri için geçmiş test süresini '7 Gün' yapmayı deneyin.")
