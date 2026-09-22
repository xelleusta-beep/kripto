import streamlit as st
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. Web Sayfası Başlığı ve Geniş Ekran Ayarı
st.set_page_config(layout="wide", page_title="Yapay Zeka Trade Paneli")
st.title("🤖 ML Tabanlı Kripto Backtest ve Analiz Platformu")

# 2. Yan Panel (Sidebar) - Kullanıcı Seçimleri
st.sidebar.header("🎛️ Strateji ve Model Ayarları")
ticker = st.sidebar.selectbox("Kripto Para Seçin", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
train_size = st.sidebar.slider("Yapay Zeka Eğitim Verisi Oranı (%)", 50, 90, 80)

st.write(f"### 📈 {ticker} Gelişmiş Teknik Analiz ve Yapay Zeka Sonuçları")

# 3. Arka Plan: Genişletilmiş Veri Çekme (Mum grafikleri için tüm veriyi alıyoruz)
symbol = ticker.replace("/", "-")
if "USDT" in symbol:
    symbol = symbol.replace("USDT", "USD")

# Tüm OHLCV (Açılış, Yüksek, Düşük, Kapanış) verilerini indiriyoruz
yf_data = vbt.YFData.download(symbol, period="1y")
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

# RSI Hesaplama (vectorbt entegre hızıyla)
df['RSI'] = vbt.RSI.run(df['Close'], window=14).rsi

# Hareketli Ortalama (SMA 20) ve Fiyata Oranı
df['SMA_20'] = vbt.MA.run(df['Close'], window=20).ma
df['Price_to_SMA'] = df['Close'] / df['SMA_20']

# Yapay Zekanın Hedefi: Yarın fiyat bugün kapanıştan yüksek olacak mı? (1 veya 0)
df['Signal_Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# 5. Arka Plan: Gelişmiş Giriş Özellikleri (Model artık fiyata değil indikatörlere bakıyor)
features = ['Return', 'RSI', 'Price_to_SMA']
X = df[features]
y = df['Signal_Target']

# 6. Arka Plan: Makine Öğrenimi Modeli Eğitme
model = RandomForestClassifier(random_state=42, n_estimators=100)
split_index = int(len(X) * (train_size / 100))

model.fit(X[:split_index], y[:split_index])
df['Predicted_Signal'] = model.predict(X)

# 7. Arka Plan: Vectorbt ile Komisyonlu Backtest Simülasyonu
portfolio = vbt.Portfolio.from_signals(
    df['Close'], 
    entries=(df['Predicted_Signal'] == 1), 
    exits=(df['Predicted_Signal'] == 0),
    fees=0.001 # %0.1 Standart Spot Borsa Komisyonu
)

# 8. Ön Yüz: Gelişmiş Görsel Paneller
col1, col2 = st.columns([3, 2]) # Sol grafik alanını biraz daha geniş yapıyoruz

with col1:
    st.write("#### 🕯️ TradingView Tarzı İnteraktif Mum Grafiği")
    # Plotly ile Profesyonel Candlestick Grafiği Çizimi
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("#### 📊 Strateji Performans Raporu")
    
    total_ret = portfolio.total_return() * 100
    
    st.metric(
        label="Yapay Zeka Toplam Net Getiri", 
        value=f"{total_ret:.2f}%",
        delta=f"{total_ret:.2f}%"
    )
    
    st.write("##### 🛠️ Detaylı Finansal İstatistikler")
    # vectorbt çıktılarını tabloya dönüştürüp Arrow hatasını engelliyoruz
    stats_df = pd.DataFrame(portfolio.stats(), columns=["Değer"]).astype(str)
    st.dataframe(stats_df, use_container_width=True)
