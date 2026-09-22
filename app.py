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

# 3. Arka Plan: Veri Çekme (Simüle edilmiş veri, CCXT ile borsadan da çekilebilir)
# Gerçek senaryoda buraya veri çekme fonksiyonu eklenir.
# 3. Arka Plan: Veri Çekme (Yahoo Finance için Ticker formatı düzeltildi)
symbol = ticker.replace("/", "-")
if "USDT" in symbol:
    symbol = symbol.replace("USDT", "USD") # Yahoo Finance BTC-USD kullanır

# Veriyi indir ve sadece kapanış fiyatını (Close) temiz bir Seri olarak al
data = vbt.YFData.download(symbol, period="1y").get('Close')

# Eğer veri çok katmanlı geldiyse tek boyuta indirge
if isinstance(data, pd.DataFrame):
    data = data.iloc[:, 0]

df = pd.DataFrame({'Close': data})

# Feature Engineering (Özellik Çıkarımı)
df['Return'] = df['Close'].pct_change()
df['Signal_Target'] = (df['Return'].shift(-1) > 0).astype(int) # Yarın fiyat artacak mı?
df.dropna(inplace=True)
y = df['Signal_Target']

model = RandomForestClassifier()
model.fit(X[:int(len(X)*(train_size/100))], y[:int(len(y)*(train_size/100))])
df['Predicted_Signal'] = model.predict(X)

# 5. Arka Plan: Vectorbt ile Backtest
# Model 1 (Al) ürettiğinde pozisyona gir, 0 ürettiğinde nakte geç
portfolio = vbt.Portfolio.from_signals(df['Close'], entries=(df['Predicted_Signal'] == 1), exits=(df['Predicted_Signal'] == 0))

# 6. Ön Yüz: Sonuçları Web Sitesine Basma
col1, col2 = st.columns(2)
with col1:
    st.write("#### Fiyat Grafiği")
    st.line_chart(df['Close'])

with col2:
    st.write("#### Backtest Performans Sonuçları")
    st.dataframe(portfolio.total_return())
