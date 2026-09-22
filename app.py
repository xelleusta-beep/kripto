import streamlit as st
import pandas as pd
import vectorbt as vbt
from sklearn.ensemble import RandomForestClassifier

# 1. Web Sayfası Başlığı ve Ayarları
st.set_page_config(layout="wide")
st.title("🤖 ML Tabanlı Kripto Backtest Platformu")

# 2. Yan Panel (Sidebar) - Kullanıcı Kontrolleri
st.sidebar.header("Strateji Parametreleri")
ticker = st.sidebar.selectbox("Kripto Para", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
train_size = st.sidebar.slider("Model Eğitim Verisi (%)", 50, 90, 80)

st.write(f"### {ticker} Verileri ve Yapay Zeka Analizi")

# 3. Arka Plan: Veri Çekme ve Temizleme
symbol = ticker.replace("/", "-")
if "USDT" in symbol:
    symbol = symbol.replace("USDT", "USD") # Yahoo Finance formatı

# Veriyi indir ve sadece kapanış fiyatını al
data = vbt.YFData.download(symbol, period="1y").get('Close')

# Çok katmanlı index yapısını tek boyuta indirge
if isinstance(data, pd.DataFrame):
    data = data.iloc[:, 0]

df = pd.DataFrame({'Close': data})

# Feature Engineering (Özellik Çıkarımı)
df['Return'] = df['Close'].pct_change()
df['Signal_Target'] = (df['Return'].shift(-1) > 0).astype(int) # Yarın fiyat artacak mı?
df.dropna(inplace=True)

# 4. Arka Plan: Makine Öğrenimi Giriş ve Çıkış Değişkenleri (Eksik olan kısım)
X = df[['Return']] 
y = df['Signal_Target']

# 5. Arka Plan: Makine Öğrenimi Modeli Eğitme
model = RandomForestClassifier(random_state=42)
split_index = int(len(X) * (train_size / 100))

model.fit(X[:split_index], y[:split_index])
df['Predicted_Signal'] = model.predict(X)

# 6. Arka Plan: Vectorbt ile Backtest Simülasyonu
portfolio = vbt.Portfolio.from_signals(
    df['Close'], 
    entries=(df['Predicted_Signal'] == 1), 
    exits=(df['Predicted_Signal'] == 0),
    fees=0.001 # %0.1 borsa komisyonu eklendi
)

# 7. Ön Yüz: Sonuçları Web Sitesine Basma
col1, col2 = st.columns(2)
with col1:
    st.write("#### Fiyat Grafiği")
    st.line_chart(df['Close'])

with col2:
    st.write("#### Backtest Performans Sonuçları")
    st.dataframe(portfolio.total_return())
