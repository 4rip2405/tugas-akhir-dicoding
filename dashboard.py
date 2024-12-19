import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
# Load data yang telah dibersihkan
alldata_df = pd.read_csv('all_data.csv')

# --- Streamlit Dashboard ---
st.title("Dashboard Analisis Bike Sharing")
st.markdown("""
Dashboard ini dirancang untuk menjawab dua pertanyaan utama:
1. Apakah ada perbedaan signifikan dalam jumlah peminjaman sepeda pada hari kerja dibandingkan akhir pekan/libur?
2. Apakah kondisi cuaca memengaruhi jumlah peminjaman sepeda?
""")

# Sidebar untuk Navigasi
option = st.sidebar.selectbox(
    "Pilih Pertanyaan untuk Dianalisis:",
    ["Perbedaan Jumlah Peminjaman (Hari Kerja dan Akhir Pekan)", "Pengaruh Kondisi Cuaca", "Analisis Lanjutan menggunakan Metode Cluster"]
)

# --- Analisis 1: Hari Kerja dan Akhir Pekan ---
if option == "Perbedaan Jumlah Peminjaman (Hari Kerja dan Akhir Pekan)":
    st.header("Perbedaan Jumlah Peminjaman (Hari Kerja dan Akhir Pekan)")

    # Data agregasi untuk hari libur
    holiday_df = alldata_df.groupby(by="holiday").cnt.mean().sort_values(ascending=False)
    holiday_df = holiday_df.rename(index={
        0: "Tidak Meminjam",
        1: "Meminjam"
    }).to_frame(name="total_peminjaman")

    # Visualisasi data hari libur
    st.subheader("- Rata-rata Peminjaman Sepeda pada Hari Libur")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        y="total_peminjaman",
        x=holiday_df.index,
        data=holiday_df,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Rata-rata Peminjaman Sepeda pada Hari Libur", loc="center", fontsize=15)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=12)
    st.pyplot(fig)

    # Data agregasi untuk hari kerja
    workingday_df = alldata_df.groupby(by="workingday").cnt.mean().sort_values(ascending=True)
    workingday_df = workingday_df.rename(index={
        0: "Tidak Meminjam",
        1: "Meminjam"
    }).to_frame(name="total_peminjaman")

    # Visualisasi data hari kerja
    st.subheader("- Rata-rata Peminjaman Sepeda pada Hari Kerja")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        y="total_peminjaman",
        x=workingday_df.index,
        data=workingday_df,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Rata-rata Peminjaman Sepeda pada Hari Kerja", loc="center", fontsize=15)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=12)
    st.pyplot(fig)

    # Kesimpulan
    st.markdown("""
    **Kesimpulan:**
    - Rata-rata jumlah peminjaman sepeda lebih tinggi pada hari kerja dibandingkan akhir pekan/libur.
    - variasi peminjaman sepeda pada akhir pekan/libur lebih besar daripada hari kerja, yang menunjukkan bahwa faktor kegiatan rekreasi memengaruhi penggunaan sepeda pada waktu tersebut.
    """)

# --- Analisis 2: Pengaruh Kondisi Cuaca ---
if option == "Pengaruh Kondisi Cuaca":
    st.header("Pengaruh Kondisi Cuaca terhadap Jumlah Peminjaman")

    # Data agregasi untuk kondisi cuaca
    weathersit_stats = alldata_df.groupby('weathersit')['cnt'].mean().reset_index()
    weathersit_stats['weathersit'] = weathersit_stats['weathersit'].map({
        1: 'Cerah',
        2: 'Berawan',
        3: 'Hujan Ringan',
        4: 'Hujan Deras'
    })

    # Visualisasi data kondisi cuaca
    st.subheader("Rata-rata Jumlah Peminjaman Berdasarkan Kondisi Cuaca")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='weathersit', y='cnt', data=weathersit_stats, palette='coolwarm', ax=ax)
    ax.set_title("Rata-rata Jumlah Peminjaman Berdasarkan Kondisi Cuaca")
    ax.set_xlabel("Kondisi Cuaca")
    ax.set_ylabel("Rata-rata Jumlah Peminjaman")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    st.pyplot(fig)
    # Kesimpulan
    st.markdown("""
    **Kesimpulan:**
    - Jumlah peminjaman sepeda tertinggi terjadi pada kondisi cuaca cerah atau berawan.
    - Cuaca buruk (hujan lebat, salju) menyebabkan penurunan drastis jumlah peminjaman.
    """)

# --- Analisis 3: Analisis Lanjutan menggunakan Metode Cluster ---
if option == "Analisis Lanjutan menggunakan Metode Cluster":
    st.header("Analisis Lanjutan menggunakan Metode Cluster")

    # Import library yang dibutuhkan
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # --- Data Preparation for Clustering ---
    # Fitur yang relevan untuk clustering
    features = alldata_df[['cnt', 'temp', 'hum', 'windspeed', 'workingday', 'weathersit']]

    # Normalisasi data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # --- Apply K-Means Clustering ---
    # Tentukan jumlah cluster (misalnya, 3 atau 4)
    kmeans = KMeans(n_clusters=3, random_state=42)
    alldata_df['Cluster'] = kmeans.fit_predict(features_scaled)

    # --- Visualisasi Cluster ---
    # Scatter plot untuk visualisasi clustering (gunakan 2D representasi)
    st.subheader("Visualisasi Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=features_scaled[:, 0],  # Fitur pertama (misal: cnt)
        y=features_scaled[:, 1],  # Fitur kedua (misal: temp)
        hue=alldata_df['Cluster'],
        palette='Set1',
        legend='full',
        ax=ax
    )
    ax.set_title('Clustering Analytics pada Data Bike Sharing')
    ax.set_xlabel('Jumlah Peminjaman (Normalized)')
    ax.set_ylabel('Suhu (Normalized)')
    st.pyplot(fig)

    # Kesimpulan
    st.markdown("""
    **Kesimpulan:**
    
    Pada metode Cluster ini dapat diketahui bahwa :

    Analisis rata-rata setiap fitur pada kelompok (cluster) untuk memahami karakteristiknya.
    Contoh: Apakah Cluster 1 didominasi oleh cuaca cerah dan hari kerja, sementara Cluster 2 terkait dengan cuaca buruk dan akhir pekan?

    Mengidentifikasi strategi untuk meningkatkan peminjaman sepeda pada kondisi cuaca tertentu atau jenis hari tertentu.
    Menemukan potensi untuk penjadwalan atau promosi pada waktu tertentu.
    """) 
    st.markdown("""
    **Alasan Menggunakan Metode Clustering :**
    - Memberikan wawasan baru dengan segmentasi berbasis pola.
    - Membantu menjawab kedua pertanyaan dengan mengelompokkan data yang memiliki pola peminjaman serupa.
    - Meningkatkan analisis prediktif di masa depan.
    
    """) 
    
