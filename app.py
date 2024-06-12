import streamlit as st
from PIL import Image
import html

def main():
    st.title("Bank Customer Churn Prediction and Visualization")
    image = Image.open('churn bank prediction.png')
    st.image(image)
    
    st.header("**Pendahuluan**")
    latar_belakang = """
    <div style='text-align: justify'>
    Dalam era digital, industri perbankan menghadapi peningkatan churn, yaitu kehilangan pelanggan ke bank lain, akibat persaingan ketat, perubahan preferensi 
    konsumen, dan teknologi yang memudahkan perbandingan layanan. Churn ini mengurangi laba dan memerlukan prediksi tepat untuk retensi pelanggan. Pelanggan adalah aset penting, 
    sehingga menjaga loyalitas melalui Customer Relationship Management (CRM) menjadi krusial. CRM menekankan pentingnya hubungan efektif dengan pelanggan untuk kesuksesan bisnis. 
    Fokus kami adalah memprediksi churn menggunakan model machine learning untuk mengurangi kehilangan nasabah dan mempertahankan pelanggan berharga.
    </div>
    """
    st.markdown(latar_belakang, unsafe_allow_html=True)
    
    st.header("**Dataset dan Machine Learning**")
    dataset = """
    <div style='text-align: justify'>
    Dataset yang kami gunakan berasal dari sumber data Kaggle. Kami memilih dataset bernama “Bank Churn Data Exploration And Churn Prediction”. Dan untuk Machine Learning nya kami memakai Machine Learning
    bernama XGBoost.
    </div>
    """
    st.markdown(dataset, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


