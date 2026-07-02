import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Dinamik Fiyat Optimizasyonu", layout="wide")
st.title("📊 Yapay Zekâ Destekli Dinamik Fiyat Optimizasyon Simülatörü")
st.markdown("Bu uygulama, geçmiş satış verilerinden talep eğrisini öğrenir ve kârı maksimum yapan fiyatı hesaplar.")

veri_sozlugu = {
    'Satis_Fiyati': [300, 400, 500, 600, 700, 800],
    'Satis_Adedi': [100, 85, 60, 40, 20, 5]
}
df = pd.DataFrame(veri_sozlugu)

st.sidebar.header("⚙️ Ürün ve Maliyet Ayarları")
maliyet = st.sidebar.number_input("Ürün Birim Maliyeti (TL):", min_value=10, max_value=1000, value=200, step=10)
kullanici_fiyati = st.sidebar.slider("Manuel Fiyat Test Et (TL):", min_value=200, max_value=800, value=400, step=10)

X = df['Satis_Fiyati'].values.reshape(-1, 1)
y = df['Satis_Adedi'].values
model = LinearRegression()
model.fit(X, y)
a_katsayisi = model.coef_[0]
b_sabiti = model.intercept_

def negatif_kar_fonksiyonu(fiyat):
    tahmini_satis_adedi = (a_katsayisi * fiyat) + b_sabiti
    if tahmini_satis_adedi < 0: return 0
    toplam_kar = tahmini_satis_adedi * (fiyat - maliyet)
    return -toplam_kar

sonuc = minimize_scalar(negatif_kar_fonksiyonu, bounds=(maliyet, 1000), method='bounded')
optimal_fiyat = sonuc.x
maksimum_kar = -sonuc.fun
optimal_adet = (a_katsayisi * optimal_fiyat) + b_sabiti

manuel_adet = (a_katsayisi * kullanici_fiyati) + b_sabiti
if manuel_adet < 0: manuel_adet = 0
manuel_kar = manuel_adet * (kullanici_fiyati - maliyet)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🎯 Yapay Zekânın Önerdiği Optimal Fiyat", value=f"{optimal_fiyat:.2f} TL")
with col2:
    st.metric(label="📦 Tahmini Satış Adedi", value=f"{int(optimal_adet)} Adet")
with col3:
    st.metric(label="💰 Hedeflenen Maksimum Kâr", value=f"{maksimum_kar:,.2f} TL")

st.divider()

st.subheader("🔍 Sizin Seçtiğiniz Fiyatın Analizi")
st.write(f"Fiyatı elle **{kullanici_fiyati} TL** yaptığınızda; tahmini **{int(manuel_adet)} adet** satarsınız ve toplam kârınız **{manuel_kar:,.2f} TL** olur.")
st.write(f"Yapay zekanın optimal fiyatına geçerek kârınızı **{maksimum_kar - manuel_kar:,.2f} TL** artırabilirsiniz!")

fiyat_araligi = np.linspace(maliyet, 800, 100)
kar_listesi = [(lambda f: max(0, (a_katsayisi * f) + b_sabiti) * (f - maliyet))(f) for f in fiyat_araligi]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(fiyat_araligi, kar_listesi, color='green', label='Kâr Eğrisi')
ax.scatter(optimal_fiyat, maksimum_kar, color='red', s=100, marker='*', zorder=5, label=f'Optimal Nokta ({optimal_fiyat:.1f} TL)')
ax.scatter(kullanici_fiyati, manuel_kar, color='blue', s=80, marker='o', zorder=5, label=f'Sizin Fiyatınız ({kullanici_fiyati} TL)')
ax.set_xlabel("Fiyat (TL)")
ax.set_ylabel("Toplam Kâr (TL)")
ax.legend()
st.pyplot(fig)
