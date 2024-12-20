import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

# --- Sidebar: Filter Interaktif ---
st.sidebar.header("Filter Data")
date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(pd.to_datetime("2011-01-01"), pd.to_datetime("2012-12-31"))
)

season_options = {
    1: "Musim Semi",
    2: "Musim Panas",
    3: "Musim Gugur",
    4: "Musim Dingin"
}
selected_season = st.sidebar.multiselect(
    "Pilih Musim",
    options=list(season_options.keys()),
    format_func=lambda x: season_options[x]
)

# --- Filter Data Berdasarkan Input ---
filtered_data = alldata_df[
    (pd.to_datetime(alldata_df['dteday']) >= pd.to_datetime(date_range[0])) &
    (pd.to_datetime(alldata_df['dteday']) <= pd.to_datetime(date_range[1]))
]
if selected_season:
    filtered_data = filtered_data[filtered_data['season'].isin(selected_season)]



# Sidebar untuk Navigasi
option = st.sidebar.selectbox(
    "Pilih Pertanyaan untuk Dianalisis:",
    ["Perbedaan Jumlah Peminjaman (Hari Kerja dan Akhir Pekan)", "Pengaruh Kondisi Cuaca", "Analisis Lanjutan menggunakan Metode Cluster"]
)

# --- Analisis 1: Hari Kerja dan Akhir Pekan ---
if option == "Perbedaan Jumlah Peminjaman (Hari Kerja dan Akhir Pekan)":
    st.header("Perbedaan Jumlah Peminjaman (Hari Kerja dan Akhir Pekan)")

    # Data agregasi untuk hari libur
    holiday_df = filtered_data.groupby(by="holiday").cnt.mean().sort_values(ascending=False)
    holiday_df = holiday_df.rename(index={
        0: "Tidak Hari Libur",
        1: "Hari Libur"
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
    workingday_df = filtered_data.groupby(by="workingday").cnt.mean().sort_values(ascending=True)
    workingday_df = workingday_df.rename(index={
        0: "Tidak Hari Libur",
        1: "Hari Libur"
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
elif option == "Pengaruh Kondisi Cuaca":
    st.header("Pengaruh Kondisi Cuaca terhadap Jumlah Peminjaman")

    # Data agregasi untuk kondisi cuaca
    weathersit_stats = filtered_data.groupby('weathersit')['cnt'].mean().reset_index()
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
elif option == "Analisis Lanjutan menggunakan Metode Cluster":
    st.header("Analisis Lanjutan menggunakan Metode Cluster")

    # --- Data Preparation for Clustering ---
    # Fitur yang relevan untuk clustering
    features = filtered_data[['cnt', 'temp', 'hum', 'windspeed', 'workingday', 'weathersit']]

    # Normalisasi data (Manual Scaling)
    features_normalized = (features - features.min()) / (features.max() - features.min())

    # --- Rule-Based Clustering ---
    # Membuat aturan clustering sederhana
    # Contoh aturan: Cluster berdasarkan jumlah peminjaman (cnt) dan suhu (temp)
    conditions = [
        (features_normalized['cnt'] <= 0.33) & (features_normalized['temp'] <= 0.33),
        (features_normalized['cnt'] > 0.33) & (features_normalized['cnt'] <= 0.66),
        (features_normalized['cnt'] > 0.66) & (features_normalized['temp'] > 0.66)
    ]

    # Berikan label cluster berdasarkan aturan
    cluster_labels = np.select(conditions, [0, 1, 2], default=3)  # Default ke cluster 3 jika tidak memenuhi aturan
    filtered_data['Cluster'] = cluster_labels

    # --- Visualisasi Cluster ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=features_normalized['cnt'],  # Fitur pertama (misal: cnt)
        y=features_normalized['temp'],  # Fitur kedua (misal: temp)
        hue=filtered_data['Cluster'],
        palette='Set1',
        legend='full'
    )
    plt.title('Clustering Analytics pada Data Bike Sharing')
    plt.xlabel('Jumlah Peminjaman (Normalized)')
    plt.ylabel('Suhu (Normalized)')
    st.pyplot()
    
# Kesimpulan
    st.markdown("""
    **Kesimpulan:**
    
Pada metode Cluster ini dapat diketahui bahwa :

- Analisis rata-rata setiap fitur pada kelompok (cluster) untuk memahami karakteristiknya.
Contoh: Apakah Cluster 1 didominasi oleh cuaca cerah dan hari kerja, sementara Cluster 2 terkait dengan cuaca buruk dan akhir pekan?
- Mengidentifikasi strategi untuk meningkatkan peminjaman sepeda pada kondisi cuaca tertentu atau jenis hari tertentu.
- Menemukan potensi untuk penjadwalan atau promosi pada waktu tertentu.
    """)
    
# Kesimpulan
    st.markdown("""
    **Alasan Menggunakan Metode Cluster:**

- Pendekatan ini transparan dan tidak memerlukan pustaka tambahan.
- Aturan clustering eksplisit dan mudah dipahami oleh siapa saja, termasuk mereka yang bukan ahli data.
- Ketika pola dalam data sudah jelas (misalnya, rentang tertentu untuk setiap fitur), metode ini sangat cocok.
    """)

