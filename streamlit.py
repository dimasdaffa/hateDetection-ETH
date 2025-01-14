#!/usr/bin/env python
# coding: utf-8

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
st.set_page_config(page_title="Hate Speech Detection", page_icon="⚡", layout="centered")

# Styling menggunakan CSS
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            font-family: 'Helvetica', sans-serif;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #777;
            font-family: 'Arial', sans-serif;
        }
        .result {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        .text-input {
            font-size: 16px;
            width: 80%;
            margin: auto;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            width: 100%;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Erik Ten Hag Twitter Hate Speech Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan teks atau Copy sebuah Tweet untuk menganalisis apakah itu Hate Speech atau  [GUNAKAN ENGLISH].</div>', unsafe_allow_html=True)

# Input teks dari pengguna
new_text = st.text_area("Teks yang ingin dianalisis:", height=200, max_chars=1000, key="input_text", label_visibility="collapsed")
st.markdown('<div class="text-input"></div>', unsafe_allow_html=True)

# Styling button
if st.button('Prediksi', key="predict_button"):
    if new_text:
        # Memproses teks
        processed_text = preprocess_text_input(new_text)

        # Mengubah teks menjadi vektor menggunakan TF-IDF Vectorizer yang telah diload
        vectorized_text = loaded_vectorizer.transform([processed_text])

        # Melakukan prediksi menggunakan model yang telah diload
        prediction = loaded_model.predict(vectorized_text)

        # Menampilkan hasil prediksi dengan styling
        if prediction[0] == 1:
            st.markdown('<div class="result" style="color: red;">Hasil: Hate Speech</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result" style="color: green;">Hasil: Not Hate Speech</div>', unsafe_allow_html=True)
    else:
        st.write("Masukkan teks terlebih dahulu.")
