#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
loaded_model = joblib.load('logistic_reg_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Fungsi untuk memproses teks
def preprocess_text_input(text):
    text = re.sub(r'http\S+', '', text)  # Menghapus URL
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Menghapus karakter khusus dan angka
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = text.strip()  # Menghapus spasi di awal/akhir teks
    return text

# Streamlit UI
st.title('Hate Speech Detection')

st.write('Masukkan teks untuk menganalisis apakah itu Hate Speech atau tidak.')

# Input teks dari pengguna
new_text = st.text_area("Teks yang ingin dianalisis:")

if st.button('Prediksi'):
    if new_text:
        # Memproses teks
        processed_text = preprocess_text_input(new_text)

        # Mengubah teks menjadi vektor menggunakan TF-IDF Vectorizer yang telah diload
        vectorized_text = loaded_vectorizer.transform([processed_text])

        # Melakukan prediksi menggunakan model yang telah diload
        prediction = loaded_model.predict(vectorized_text)

        # Menampilkan hasil prediksi
        if prediction[0] == 1:
            st.write("Hasil: Hate Speech")
        else:
            st.write("Hasil: Not Hate Speech")
    else:
        st.write("Masukkan teks terlebih dahulu.")


# In[ ]:




