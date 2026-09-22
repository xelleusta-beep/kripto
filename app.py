import streamlit as st
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. Web Sayfası Başlığı ve Geniş Ekran Ayarı
st.set_page_config(layout="wide", page_title="Yapay Zeka & Paper Trading Paneli")
st.title("🤖 Gelişmiş ML Backtest & Canlı Paper Trading Platformu")

# --- PAPER TRADING SESSİON STATE BAŞLATMA ---
if "balance" not in st.session_state:
    st.session_state.balance = 10000.0  # Başlangıç sanal dolar bakiyesi
if "crypto_amount" not in st.session_state:
    st.session_state.crypto_amount = 0.0  # Sahip olunan kripto miktarı
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []  # İşlem geçmişi kayıtları

# 2. Yan Panel (Sidebar) - Kullanıcı Seçimleri
st.sidebar.header("🎛️ Strateji ve Zaman Ayarları")
ticker = st.sidebar.selectbox("Kripto Para Seçin", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

# YENİ OZELLİK: Geçmiş test sürelerini dinamik değiştirmek için zaman aralığı seçici
time_period = st.sidebar.selectbox(
    "Geçmiş Test Süresi (Backtest Range)", 
    ["1 Ay", "3 Ay", "6 Ay", "1 Yıl", "3 Yıl"], 
    index=3
)

train_size = st.sidebar.slider("Yapay Zeka Eğitim Verisi Oranı (%)", 50, 90, 80)

# Seçilen periyodu Yahoo Finance formatına çevirme
period_mapping = {"1 Ay": "1mo", "3 Ay": "3mo", "6 Ay": "6mo", "1 Yıl": "1y", "3 Yıl": "3y"}
selected_period = period_mapping[time_period]

st.write(f"### 📈 {ticker} ({time_period} Verisi) Teknik Analiz ve Yapay Zeka Sonuçları")

# 3. Arka Plan: Genişletilmiş Veri Çekme
symbol = ticker.replace("/", "-")
if "USDT" in symbol:
    symbol = symbol.replace("USDT", "USD")

# Seçilen dinamik zaman aralığına göre veri indiriliyor
yf_data = vbt.YFData.download(symbol, period=selected_period)
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

# 4. Feature Engineering: Yapay Zekaya İndikatör Öğretme
df['Return'] = df['Close'].pct_change()
df['RSI'] = vbt.RSI.run(df['Close'], window=14).rsi
df['SMA_20'] = vbt.MA.run(df['Close'], window=20).ma
df['Price_to_SMA'] = df['Close'] / df['SMA_20']

# Yapay Zekanın Hedefi: Yarın fiyat bugün kapanıştan yüksek olacak mı?
df['Signal_Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# 5. Arka Plan: Giriş Özellikleri ve Model Eğitme
features = ['Return', 'RSI', 'Price_to_SMA']
X = df[features]
y = df['Signal_Target']

model = RandomForestClassifier(random_state=42, n_estimators=100)
split_index = int(len(X) * (train_size / 100))

model.fit(X[:split_index], y[:split_index])
df['Predicted_Signal'] = model.predict(X)

# En güncel anlık fiyat ve son üretilen yapay zeka sinyali
current_price = float(df['Close'].iloc[-1])
latest_signal = int(df['Predicted_Signal'].iloc[-1])

# 6. Arka Plan: Vectorbt ile Komisyonlu Backtest Simülasyonu
portfolio = vbt.Portfolio.from_signals(
    df['Close'], 
    entries=(df['Predicted_Signal'] == 1), 
    exits=(df['Predicted_Signal'] == 0),
    fees=0.001
)

# 7. Ön Yüz Tasarımı: Sekmeli Yapı (Backtest ve Paper Trading Ayrı Sekmelerde)
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
    
    # Portföy Özet Metrikleri
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Sanal Nakit Bakiye", f"${st.session_state.balance:,.2f}")
    p_col2.metric(f"Varlık Miktarı ({ticker.split('/')[0]})", f"{st.session_state.crypto_amount:.4f}")
    
    total_asset_value = st.session_state.crypto_amount * current_price
    total_portfolio_value = st.session_state.balance + total_asset_value
    p_col3.metric("Toplam Portföy Değeri", f"${total_portfolio_value:,.2f}")
    
    # Anlık Yapay Zeka Tavsiyesi Sinyal Kutusu
    if latest_signal == 1:
        p_col4.success("🤖 YAPAY ZEKA TAVSİYESİ: AL (BUY)")
    else:
        p_col4.error("🤖 YAPAY ZEKA TAVSİYESİ: SAT (SELL)")
        
    st.write(f"**Anlık Güncel Fiyat:** ${current_price:,.2f}")
    
    # Alım Satım Butonları ve Emir Kontrolleri
    trade_col1, trade_col2 = st.columns(2)
    
    with trade_col1:
        if st.button("🟢 Sanal BAKİYE İLE MAKSİMUM AL", use_container_width=True):
            if st.session_state.balance > 10:
                amount_to_buy = st.session_state.balance / current_price
                st.session_state.crypto_amount += amount_to_buy
                st.session_state.trade_history.append(f"ALINDI: {amount_to_buy:.4f} {ticker} @ ${current_price:,.2f}")
                st.session_state.balance = 0.0
                st.rerun()
            else:
                st.warning("Yetersiz Nakit Bakiye!")
                
    with trade_col2:
        if st.button("🔴 TÜM VARLIKLARI SANAL METAYA SAT", use_container_width=True):
            if st.session_state.crypto_amount > 0:
                gain = st.session_state.crypto_amount * current_price
                st.session_state.balance += gain
                st.session_state.trade_history.append(f"SATILDI: {st.session_state.crypto_amount:.4f} {ticker} @ ${current_price:,.2f}")
                st.session_state.crypto_amount = 0.0
                st.rerun()
            else:
                st.warning("Satılacak Kripto Varlığınız Bulunmuyor!")

    # İşlem Geçmişi Listesi
    st.write("##### 🕒 Sanal İşlem Geçmişi (Aktivite Günlüğü)")
    if st.session_state.trade_history:
        for log in reversed(st.session_state.trade_history):
            st.text(log)
    else:
        st.info("Henüz simüle edilmiş bir işlem gerçekleştirilmedi.")
