import streamlit as st
import pandas as pd
import io
import pickle
import os

pilihan = st.sidebar.selectbox(
    "Pilih Jenis Prediksi",
    ("Prediksi Kategori Properti", "Prediksi Harga Properti")
)

def baca_csv():
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
    return None

def input_data_properti():
    with st.form("form_properti"):
        st.write("Masukkan data properti:")
        
        squaremeters = st.number_input("Luas Tanah (m²)", min_value=0.0)
        numberofrooms = st.number_input("Jumlah Kamar", min_value=0, step=1)
        hasyard = st.checkbox("Ada Halaman")
        haspool = st.checkbox("Ada Kolam Renang")
        floors = st.number_input("Jumlah Lantai", min_value=1, step=1)
        citycode = st.text_input("Kode Lokasi")
        citypartrange = st.selectbox("citypartrange Kawasan", ["Rendah", "Sedang", "Tinggi"])
        numprevowners = st.number_input("Jumlah Pemilik Sebelumnya", min_value=0, step=1)
        made = st.number_input("Tahun Pembuatan", min_value=1800, max_value=2023, step=1)
        isnewbuilt = st.checkbox("Gedung Baru")
        hasstormprotector = st.checkbox("Ada Pelindung Badai")
        basement = st.number_input("Luas Basement (m²)", min_value=0.0)
        attic = st.number_input("Luas Loteng (m²)", min_value=0.0)
        garage = st.number_input("Luas Garasi (m²)", min_value=0.0)
        hasstorageroom = st.checkbox("Ada Gudang")
        hasguestroom = st.checkbox("Ada Ruang Tamu")

        submitted = st.form_submit_button("Prediksi")

        if submitted:
            return {
                "squaremeters": squaremeters,
                "numberofrooms": numberofrooms,
                "hasyard": hasyard,
                "haspool": haspool,
                "floors": floors,
                "citycode": citycode,
                "citypartrange": citypartrange,
                "numprevowners": numprevowners,
                "made": made,
                "isnewbuilt": isnewbuilt,
                "hasstormprotector": hasstormprotector,
                "basement": basement,
                "attic": attic,
                "garage": garage,
                "hasstorageroom": hasstorageroom,
                "hasguestroom": hasguestroom
            }
    return None

def muat_model(nama_file):
    path_file = nama_file
    try:
        with open(path_file, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

def format_data(data):
    return pd.DataFrame([data])

def lakukan_prediksi(model, data):
    try:
        prediksi = model.predict(data)
        return prediksi[0]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return None

if pilihan == "Prediksi Kategori Properti":
    st.title("Prediksi Kategori Properti")
    
    model_kategori = muat_model("BestModel_CLF_RF_Bokeh.pkl")
    
    if model_kategori is None:
        st.error("Gagal memuat model prediksi kategori.")
    else:
        metode_input = st.radio("Pilih metode input:", ("Input Manual", "Unggah CSV"), key="input_kategori")
        
        if metode_input == "Input Manual":
            data = input_data_properti()
            if data:
                formatted_data = format_data(data)
                prediksi = lakukan_prediksi(model_kategori, formatted_data)
                if prediksi is not None:
                    st.success(f"Prediksi kategori properti: {prediksi}")
                st.write("Data yang diinput:", data)
        else:
            df = baca_csv()
            if df is not None:
                st.write("Data dari file CSV:")
                st.write(df)
                prediksi = lakukan_prediksi(model_kategori, df)
                if prediksi is not None:
                    st.success(f"Prediksi kategori properti: {prediksi}")

elif pilihan == "Prediksi Harga Properti":
    st.title("Prediksi Harga Properti")
    
    model_harga = muat_model("BestModel_REG_Ridge_Bokeh.pkl")
    
    if model_harga is None:
        st.error("Gagal memuat model prediksi harga.")
    else:
        metode_input = st.radio("Pilih metode input:", ("Input Manual", "Unggah CSV"), key="input_harga")
        
        if metode_input == "Input Manual":
            data = input_data_properti()
            if data:
                formatted_data = format_data(data)
                prediksi = lakukan_prediksi(model_harga, formatted_data)
                if prediksi is not None:
                    st.success(f"Prediksi harga properti: Rp {prediksi:,.2f}")
                st.write("Data yang diinput:", data)
        else:
            df = baca_csv()
            if df is not None:
                st.write("Data dari file CSV:")
                st.write(df)
                prediksi = lakukan_prediksi(model_harga, df)
                if prediksi is not None:
                    st.success(f"Prediksi harga properti: Rp {prediksi:,.2f}")
