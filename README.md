# 🏦 Bank Customer Churn Prediction

## 👥 Team Members
- Agisni Zahra Latifa (1305213038)  
- Ahmad Jundi Khairurrijal (1305213037)  
- Aina Rosada Almardhiya (1305213013)  
- Agung Hadi Winoto (1305213027)  

---

## 📌 Project Overview
Customer churn (pelanggan berhenti menggunakan layanan) menjadi salah satu tantangan terbesar di industri perbankan modern.  
Dalam proyek ini, kami membangun **model prediksi churn** menggunakan **XGBoost** dan membandingkannya dengan model lain (Random Forest, Extra Trees, Gradient Boosting).  

Selain itu, kami juga mengembangkan **dashboard interaktif berbasis Streamlit** untuk memvisualisasikan data dan menampilkan hasil prediksi churn secara langsung.  

---

## 📊 Dataset
- **Source**: Kaggle – *Bank Churn Data Exploration and Prediction*  
- **Size**: 10,127 rows × 23 columns  
- **Features**: usia, jenis kelamin, status perkawinan, pendidikan, pendapatan, jenis kartu, pengeluaran, dsb.  
- **Target**:  
  - `Existing Customer` (84%)  
  - `Attrited Customer` (16%)  

---

## ⚙️ Methodology
1. **Data Preprocessing**  
   - Data cleaning (handling missing values & outliers)  
   - Normalization (StandardScaler)  
   - Feature selection (feature importance)  

2. **Exploratory Data Analysis (EDA)**  
   - Distribusi gender terhadap churn  
   - Rentang usia pelanggan  
   - Pola transaksi & atribut penting  

3. **Data Balancing**  
   - Menggunakan **SMOTE** untuk mengatasi class imbalance  

4. **Machine Learning Models**  
   - Random Forest  
   - Extra Trees  
   - Gradient Boosting  
   - XGBoost (main model)  

5. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  

---

## 🏆 Results
| Model               | Accuracy |
|---------------------|----------|
| Extra Trees         | 85–92%   |
| Random Forest       | 90–95%   |
| Gradient Boosting   | 94–98%   |
| **XGBoost**         | **98.56%** ✅ |

📌 **XGBoost** achieved the highest accuracy (98.56%), making it the best model for churn prediction in this project.  

---

## 📊 Streamlit Application
We deployed our model and visualization dashboard on Streamlit Cloud.  

👉 [Bank Churn Prediction App](https://bankchurnpredict.streamlit.app/)  

Features:  
- **Prediction Page** → Predict churn from input customer data  
- **Visualization Page** → Explore dataset with interactive charts  
- **App Info Page** → Project background & methodology  

---

