import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
import warnings

# Mengabaikan peringatan
warnings.filterwarnings('ignore')

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Penyakit Pernapasan",
    page_icon="🩺",
    layout="wide"
)

# --- 2. Fungsi Memuat Model ---
@st.cache_data
def load_and_train_model():
    """
    Memuat dataset, melakukan preprocessing, 
    dan melatih model Bernoulli Naive Bayes.
    """
    try:
        # --- (INI YANG DIUBAH) ---
        file_path = 'dataset_prediksi_penyakit_pernapasan.csv'
        df = pd.read_csv(file_path)

        # Preprocessing (sesuai notebook Anda)
        if 'Umur' in df.columns:
            df = df.drop(columns=['Umur'])

        # Pisahkan Fitur (X) dan Target (y)
        X = df.drop(columns=['Diagnosis'])
        y = df['Diagnosis']
        feature_names = X.columns.to_list()

        # Encoding Target (y)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Melatih Model
        model = BernoulliNB()
        model.fit(X, y_encoded)
        
        # Mengembalikan model, encoder (le), dan nama fitur
        return model, le, feature_names

    except FileNotFoundError:
        # --- (PESAN ERROR INI JUGA DIUBAH) ---
        st.error(f"Error: File '{file_path}' tidak ditemukan.")
        st.info(f"Pastikan file '{file_path}' berada di folder yang sama dengan file .py ini.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau melatih model: {e}")
        return None, None, None

# --- 3. Memuat Model ---
model, le, feature_names = load_and_train_model()

# --- 4. Antarmuka (UI) Aplikasi ---

# Hanya tampilkan UI jika model berhasil dimuat
if model is not None and le is not None and feature_names is not None:

    # --- Bagian Header dan Penjelasan ---
    st.title("🩺 Sistem Prediksi Diagnosis Penyakit Pernapasan")
    st.markdown("""
    Selamat datang di aplikasi prediksi penyakit. Aplikasi ini menggunakan model *Machine Learning* (Bernoulli Naive Bayes)
    untuk menganalisis probabilitas **diagnosis penyakit pernapasan** berdasarkan 16 gejala klinis.
    
    **Project ini ditujukan untuk memenuhi tugas Komputasi Cerdas Dalam Fisika.**
    """)
    st.markdown("---")

    # --- Bagian Input Data Diri ---
    st.subheader("1. Masukkan Data Diri Pasien")
    
    # Menggunakan 2 kolom agar input Nama dan Umur sejajar
    col1, col2 = st.columns(2)
    with col1:
        nama_pasien = st.text_input("Nama Lengkap Pasien", placeholder="Contoh: Budi Santoso")
    with col2:
        # Note: Model Anda tidak menggunakan Umur, tapi kita tetap menambahkannya
        # untuk kelengkapan data diri sesuai permintaan Anda.
        umur_pasien = st.number_input("Umur (Tahun)", min_value=0, max_value=120, value=30, step=1)
    
    st.info("Catatan: Input 'Umur' saat ini hanya untuk kelengkapan data. Model ini dilatih **hanya** berdasarkan 16 gejala yang dipilih di bawah.")
    st.markdown("---")

    # --- Bagian Input Gejala (Di Halaman Utama) ---
    st.subheader("2. Pilih Gejala yang Dialami")

    user_input = {}
    
    # Menggunakan 'expander' agar daftar gejala tidak memakan tempat
    with st.expander("Klik di sini untuk melihat dan memilih gejala", expanded=True):
        
        # Membuat 4 kolom untuk menata 16 gejala
        cols = st.columns(4)
        
        for i, feature in enumerate(feature_names):
            # Membagi 16 gejala ke 4 kolom (0, 1, 2, 3)
            col_index = i % 4
            with cols[col_index]:
                friendly_name = feature.replace("_", " ") # Membuat nama lebih rapi
                user_input[feature] = st.checkbox(friendly_name, key=feature)
    
    st.markdown("---")

    # --- Bagian Tombol Prediksi ---
    st.subheader("3. Mulai Analisis")
    
    # Tombol dibuat besar dan jelas
    if st.button("Analisis Sekarang", type="primary", use_container_width=True):
        
        # --- Validasi Input ---
        if not nama_pasien:
            st.warning("Harap masukkan 'Nama Lengkap Pasien' terlebih dahulu.")
        
        # Cek jika tidak ada gejala yang dipilih sama sekali
        elif all(value == False for value in user_input.values()):
            st.warning("Harap pilih minimal satu gejala yang dialami.")
        
        else:
            # --- Proses Prediksi ---
            try:
                # 1. Kumpulkan input gejala (list 0 dan 1)
                input_data = [int(user_input[feature]) for feature in feature_names]
                input_array = [input_data]

                # 2. Ambil PREDIKSI (hasilnya)
                prediction_encoded = model.predict(input_array)
                prediction_text = le.inverse_transform(prediction_encoded)[0]

                # 3. Ambil PROBABILITAS (Persentase)
                probabilities = model.predict_proba(input_array)[0]
                
                # 4. Ambil skor keyakinan dari kelas yang diprediksi
                confidence_score = probabilities[prediction_encoded[0]]
                
                # --- Tampilan Hasil ---
                st.subheader(f"Hasil Analisis untuk: {nama_pasien} ({umur_pasien} Tahun)")
                
                # Menggunakan container agar hasil terlihat rapi dalam satu boks
                with st.container(border=True):
                    col_res1, col_res2 = st.columns(2)
                    
                    # Kolom 1: Menampilkan Hasil Prediksi
                    with col_res1:
                        st.metric(label="Hasil Prediksi Diagnosis", value=prediction_text)
                    
                    # Kolom 2: Menampilkan Persentase Keyakinan
                    with col_res2:
                        st.metric(label="Tingkat Keyakinan Model", value=f"{confidence_score * 100:.2f} %")
                
                st.markdown("---")

                # --- Rincian Probabilitas (Diagram Batang) ---
                st.subheader("Rincian Probabilitas (Semua Diagnosis)")
                
                # Membuat DataFrame untuk visualisasi
                prob_df = pd.DataFrame({
                    'Diagnosis': le.classes_, # Mengambil semua nama diagnosis
                    'Probabilitas': probabilities * 100 # Ubah ke persentase
                }).sort_values(by='Probabilitas', ascending=False)
                
                # Mengganti nama kolom agar lebih jelas di diagram
                prob_df.rename(columns={'Probabilitas': 'Probabilitas (%)'}, inplace=True)
                
                # Tampilkan diagram batang
                st.bar_chart(prob_df.set_index('Diagnosis'))

                # --- Disclaimer Medis (Sangat Penting) ---
                st.warning(f"""
                **DISCLAIMER MEDIS (PENTING):**
                
                * Hasil prediksi **{prediction_text}** dengan keyakinan **{confidence_score * 100:.2f}%** adalah murni output dari model *machine learning* berdasarkan data latih.
                * Aplikasi ini **BUKAN** pengganti diagnosis medis profesional. 
                * Selalu konsultasikan dengan dokter atau tenaga medis berlisensi untuk mendapatkan diagnosis dan penanganan yang akurat.
                """)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# Jika model gagal dimuat di awal
else:
    # --- (PESAN ERROR INI JUGA DIUBAH) ---
    st.error("Aplikasi tidak dapat dimuat. Pastikan file 'pernapasan.csv' sudah benar.")