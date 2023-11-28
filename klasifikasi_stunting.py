import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import altair as alt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import base64
from io import BytesIO

st.markdown(
    """<h1 style='text-align: center;'> Klasifikasi Penentuan Status Stunting</h1> """,
    unsafe_allow_html=True,
)


# 1. as sidevar menu
with st.sidebar:
    # Function to convert image to base64
    def image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Calculate the space to center the image
    space = st.sidebar.empty()

    img = Image.open("bayi1.png")

    # Center the image using Markdown
    st.sidebar.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{image_to_base64(img)}" style="width:150px;"></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title="Status Stunting",  # required
        options=[
            "Beranda",
            "Dataset",
            "Preprocessing",
            "Permodelan",
            "Implementasi",
        ],  # required
        icons=[
            "house-door-fill",
            "book-half",
            "bi bi-file-earmark-arrow-up-fill",
            "arrow-repeat",
            "medium",
            "folder-fill",
            "bookmark-fill",
        ],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        styles={
            "container": {"padding": "0!important", "background-color": "white"},
            "icon": {"color": "black", "font-size": "17px"},
            "nav-link": {
                "font-size": "17px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#4169E1",
            },
            "nav-link-selected": {"background-color": "Royalblue"},
        },
    )

if selected == "Beranda":
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.write("")

    with col2:
        img = Image.open("stunting.jpg")
        st.image(img, use_column_width=False, width=300)

    with col3:
        st.write("")

    st.write(""" """)

    st.write(""" """)

    st.header("Apakah itu stunting?")

    st.write(
        """
    Menurut WHO (2015), stunting adalah gangguan pertumbuhan dan perkembangan anak akibat kekurangan 
    gizi kronis dan infeksi berulang, yang ditandai dengan panjang atau tinggi badannya berada di bawah 
    standar. Selanjutnya menurut WHO (2020) stunting adalah pendek atau sangat pendek berdasarkan panjang / 
    tinggi badan menurut usia yang kurang dari -2 standar deviasi (SD) pada kurva pertumbuhan WHO yang 
    terjadi dikarenakan kondisi irreversibel akibat asupan nutrisi yang tidak adekuat dan/atau infeksi 
    berulang / kronis yang terjadi dalam 1000 HPK.
    """
    )

    st.header("Apakah semua balita pendek itu pasti stunting?")
    st.write(
        """Perlu diketahui bahwa tidak semua balita pendek itu stunting, sehingga perlu dibedakan oleh 
        dokter anak, tetapi anak yang stunting pasti pendek.
    """
    )

    st.header("Dampak Masalah Stunting di Indonesia")
    st.subheader("1. Dampak Kesehatan")
    st.write(
        """a. Gagal tumbuh (berat lahir rendah, kecil, pendek, kurus), hambatan perkembangan kognitif dan motoric.
           b. Gangguan metabolik pada saat dewasa → risiko penyakit tidak menular (diabetes, obesitas, stroke, penyakit jantung, dan lain sebagainya).
    """
    )
    st.subheader("2. Dampak Ekonomi")
    st.write(
        """Berpotensi menimbulkan kerugian setiap tahunnya : 2-3 persen GDP.
    """
    )
    st.header("Penyebab Stunting")
    st.subheader("1. Asupan Kalori yang tidak adekuat")
    st.write(
        """
        a.      Faktor sosio-ekonomi (kemiskinan).

        b.      Pendidikan dan pengetahuan yang rendah mengenai praktik pemberian makan untuk bayi dan batita (kecukupan ASI).

        c.      Peranan protein hewani dalam MPASI.

        d.      Penelantaran

        e.      Pengaruh budaya

        f.       Ketersediaan bahan makanan setempat.
    """
    )
    st.subheader("2. Kebutuhan yang Meningkat")
    st.write(
        """
        a.      Penyakit jantung bawaan.

        b.      Alergi susu sapi.

        c.      Bayi berat badan lahir sangat rendah.

        d.      Kelainan metabolisme bawaan.

        e.      Infeksi kronik yang disebabkan kebersihan personal dan lingkungan yang buruk (diare kronis) dan 
        penyakit-penyakit yang dapat dicegah oleh imunisasi (Tuberculosis / TBC, difteri, pertussis, dan campak).
    """
    )

    st.header("Cara Mencegah Stunting")
    st.subheader("1. Saat Remaja Putri")
    st.write(
        """
        Skrining anemia dan konsumsi tablet tambah darah. 
    """
    )
    st.subheader("2. Saat Masa Kehamilan")
    st.write(
        """
        Disarankan untuk rutin memeriksakan kondisi kehamilan ke dokter. Perlu juga memenuhi asupan nutrisi yang baik selama kehamilan. Dengan makanan sehat dan juga asupan mineral seperti zat besi, asam folat, dan yodium harus tercukupi. 
    """
    )
    st.subheader("3. Balita")
    st.write(
        """
       a.      Terapkan Inisiasi Menyusui Dini (IMD).

        Sesaat setelah bayi lahir, segera lakukan IMD agar berhasil menjalankan ASI Eksklusif. Setelah itu, lakukan pemeriksaan 
        ke dokter atau ke Posyandu dan Puskesmas secara berkala untuk memantau pertumbuhan dan perkembangan anak.

        b.      Imunisasi

        Perhatikan jadwal imunisasi rutin yang diterapkan oleh Pemerintah agar anak terlindungi dari berbagai macam penyakit.

        c.      ASI Eksklusif

        Berikan ASI eksklusif sampai anak berusia 6 (enam) bulan dan diteruskan dengan MPASI yang sehat dan bergizi.

        d.      Pemantauan tumbuh kembang à weight faltering.
    """
    )
    st.subheader("4. Gaya Hidup dan Sehat")
    st.write(
        """
        Terapkan gaya hidup bersih dan sehat, seperti mencuci tangan sebelum makan, memastikan air yang diminum merupakan air 
        bersih, buang air besar di jamban, sanitasi sehat, dan lain sebagainya.
    """
    )


if selected == "Dataset":
    st.markdown(
        """<h2 style='text-align: center; color:grey;'> Dataset Stunting Puskesmas Kalianget </h1> """,
        unsafe_allow_html=True,
    )
    df = pd.read_csv(
        "https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/Data_stunting_kalianget_sistem.csv"
    )
    c1, c2, c3 = st.columns([1, 5, 1])

    with c1:
        st.write("")

    with c2:
        df

    with c3:
        st.write("")

if selected == "Preprocessing":
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image(
        "https://ilmudatapy.com/wp-content/uploads/2020/07/normalisasi-3.png",
        use_column_width=False,
        width=250,
    )
    st.markdown(
        """
    Dimana :
    - X old= data yang akan dinormalisasi atau data asli
    - X min = nilai minimum semua data asli
    - X max = nilai maksimum semua data asli
    """
    )
    df = pd.read_csv(
        "https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/Data_stunting_kalianget_sistem.csv"
    )
    # drop kolom nama lengkap
    data = df.drop(columns=["Nama Lengkap"])
    # values
    X = data.iloc[:, 0:7]
    # classes
    y = data.iloc[:, 7]

    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    # features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader("Hasil Normalisasi Data")
    st.write(scaled_features)
    st.subheader("Target Label")
    data = df.rename(columns={"Status Stunting": "status_stunting"})
    dumies = pd.get_dummies(data.status_stunting).columns.values.tolist()
    dumies = np.array(dumies)
    labels = pd.DataFrame({"Stunting": [dumies[0]], "Normal": [dumies[-1]]})
    st.write(labels)


if selected == "Permodelan":
    df = pd.read_csv(
        "https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/Data_stunting_kalianget_sistem.csv"
    )
    # Mendefinisikan Varible X dan Y
    # drop kolom nama lengkap
    data = df.drop(columns=["Nama Lengkap"])
    # values
    X = data.iloc[:, 0:7]
    # classes
    y = data.iloc[:, 7]

    # SELEKSI FITUR
    num_iterations = 100  # Jumlah iterasi
    information_gains = []
    for _ in range(num_iterations):
        information_gain = mutual_info_classif(X, y)
        information_gains.append(information_gain)
    # Rata-rata hasil Information Gain untuk semua fitur
    average_information_gain = np.mean(information_gains, axis=0)
    # Buat DataFrame untuk menampilkan hasil
    st.subheader("Hasil Perangkingan Fitur Information Gain")
    info_gain_df = pd.DataFrame(
        {"Fitur": X.columns, "Average Information Gain": average_information_gain}
    )
    # Mengurutkan hasil berdasarkan Information Gain secara menurun
    info_gain_df = info_gain_df.sort_values(
        by="Average Information Gain", ascending=False
    )
    st.write(info_gain_df)
    K = 4  # Ganti nilai K sesuai dengan jumlah fitur terbaik yang ingin Anda pilih
    st.subheader("Fitur Terpilih")
    st.write(info_gain_df.head(K))
    selected_features = info_gain_df.head(K)["Fitur"]
    # Filter matriks fitur berdasarkan fitur terbaik
    X_new = X[selected_features]

    # NORMALISASI DATA
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X_new)
    features_names = X_new.columns.copy()
    # features_names.remove('label')
    scaled_features_SF = pd.DataFrame(scaled, columns=features_names)

    # MODEL
    training, test = train_test_split(
        scaled_features_SF, test_size=0.083, random_state=1
    )  # Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.083, random_state=1)
    st.subheader("Modeling")
    st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
    SVM = st.checkbox("SVM")
    submitted = st.button("Submit")

    # Nilai Y training dan Nilai Y testing
    clf = svm.SVC(C=1, kernel="poly", gamma=1, max_iter=100, degree=5, coef0=0.1)
    clf.fit(training, training_label)
    y_predict = clf.predict(test)
    acc = accuracy_score(test_label, y_predict)

    if submitted:
        if SVM:
            st.write("Model SVM accuracy score: {0:0.2f}".format(acc))


if selected == "Implementasi":
    # Inputan Form
    with st.form("my_form"):
        df = pd.read_csv(
            "https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/Data_stunting_kalianget_sistem.csv"
        )
        # Mendefinisikan fitur dan kelas
        data = df.drop(columns=["Nama Lengkap"])
        X = data.iloc[:, 0:7]
        y = data.iloc[:, 7]

        # SELEKSI FITUR

        # iterasi
        num_iterations = 100
        information_gains = []
        for _ in range(num_iterations):
            information_gain = mutual_info_classif(X, y)
            information_gains.append(information_gain)

        # Rata-rata hasil Information Gain untuk semua fitur
        average_information_gain = np.mean(information_gains, axis=0)

        # Buat DataFrame untuk menampilkan hasil
        info_gain_df = pd.DataFrame(
            {"Fitur": X.columns, "Average Information Gain": average_information_gain}
        )

        # Mengurutkan hasil berdasarkan Information Gain secara menurun
        info_gain_df = info_gain_df.sort_values(
            by="Average Information Gain", ascending=False
        )

        K = 4  # Ganti nilai K sesuai dengan jumlah fitur terbaik yang ingin Anda pilih
        # Filter matriks fitur berdasarkan fitur terbaik
        selected_features = info_gain_df.head(K)["Fitur"]
        X_new = X[selected_features]

        # NORMALISASI DATA
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X_new)
        features_names = X_new.columns.copy()
        # features_names.remove('label')
        scaled_features_SF = pd.DataFrame(scaled, columns=features_names)

        # MODEL
        training, test = train_test_split(
            scaled_features_SF, test_size=0.083, random_state=1
        )
        # Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(
            y, test_size=0.083, random_state=1
        )
        # Nilai Y training dan Nilai Y testing
        clf = svm.SVC(C=1, kernel="poly", gamma=1, max_iter=100, degree=5, coef0=0.1)
        clf.fit(training, training_label)
        y_predict = clf.predict(test)
        acc = accuracy_score(test_label, y_predict)

        st.subheader("Implementasi")
        nama = st.text_input("Masukan Nama Lengkap")
        jenis_kelamin = st.selectbox(
            "Masukkan Jenis Kelamin", ("Laki-laki", "Perempuan")
        )
        BB_lahir = st.number_input("Masukkan Berat Badan Lahir (dalam satuan kilogram)")
        PB_lahir = st.number_input(
            "Masukkan Panjang Badan Lahir (dalam satuan centimeter)"
        )
        TB_Saatini = st.number_input(
            "Masukan Berat Badan Saat Ini (dalam satuan kilogram)"
        )
        BB_Saatini = st.number_input(
            "Masukan Tinggi Badan Saat Ini (dalam satuan centimeter)"
        )
        umur = st.number_input("Masukan Umur Saat Ini (dalam bulan)")

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            # Fungsi untuk menghitung Z_score dan menentukan Status
            def hitung_status_umur(jenis_kelamin, umur, TB_Saatini):
                if jenis_kelamin == "Laki-laki":
                    if umur == 0:
                        return None
                    elif umur == 1:
                        Z_score = (TB_Saatini - 54.7) / (54.7 - 52.8)
                    elif umur == 2:
                        Z_score = (TB_Saatini - 58.4) / (58.4 - 56.4)
                    elif umur == 3:
                        Z_score = (TB_Saatini - 61.4) / (61.4 - 59.4)
                    elif umur == 4:
                        Z_score = (TB_Saatini - 63.9) / (63.9 - 61.8)
                    elif umur == 5:
                        Z_score = (TB_Saatini - 65.9) / (65.9 - 63.8)
                    elif umur == 6:
                        Z_score = (TB_Saatini - 67.6) / (67.6 - 65.5)
                    elif umur == 7:
                        Z_score = (TB_Saatini - 69.2) / (69.2 - 67.0)
                    elif umur == 8:
                        Z_score = (TB_Saatini - 70.6) / (70.6 - 68.4)
                    elif umur == 9:
                        Z_score = (TB_Saatini - 72.0) / (72.0 - 69.7)
                    elif umur == 10:
                        Z_score = (TB_Saatini - 73.3) / (73.3 - 71.0)
                    elif umur == 11:
                        Z_score = (TB_Saatini - 74.5) / (74.5 - 72.2)
                    elif umur == 12:
                        Z_score = (TB_Saatini - 75.7) / (75.7 - 73.4)
                    elif umur == 13:
                        Z_score = (TB_Saatini - 76.9) / (76.9 - 74.5)
                    elif umur == 14:
                        Z_score = (TB_Saatini - 78.0) / (78.0 - 75.6)
                    elif umur == 15:
                        Z_score = (TB_Saatini - 79.1) / (79.1 - 76.6)
                    elif umur == 16:
                        Z_score = (TB_Saatini - 80.2) / (80.2 - 77.6)
                    elif umur == 17:
                        Z_score = (TB_Saatini - 81.2) / (81.2 - 78.6)
                    elif umur == 18:
                        Z_score = (TB_Saatini - 82.3) / (82.3 - 79.6)
                    elif umur == 19:
                        Z_score = (TB_Saatini - 83.2) / (83.2 - 80.5)
                    elif umur == 20:
                        Z_score = (TB_Saatini - 84.2) / (84.2 - 81.4)
                    elif umur == 21:
                        Z_score = (TB_Saatini - 85.1) / (85.1 - 82.3)
                    elif umur == 22:
                        Z_score = (TB_Saatini - 86.0) / (86.0 - 83.1)
                    elif umur == 23:
                        Z_score = (TB_Saatini - 86.9) / (86.9 - 83.9)
                    elif umur == 24:
                        Z_score = (TB_Saatini - 87.8) / (87.8 - 84.8)
                    elif umur == 25:
                        Z_score = (TB_Saatini - 88.0) / (88.0 - 84.9)
                    elif umur == 26:
                        Z_score = (TB_Saatini - 88.8) / (88.8 - 85.6)
                    elif umur == 27:
                        Z_score = (TB_Saatini - 89.6) / (89.6 - 86.4)
                    elif umur == 28:
                        Z_score = (TB_Saatini - 90.4) / (90.4 - 87.1)
                    elif umur == 29:
                        Z_score = (TB_Saatini - 91.2) / (91.2 - 87.8)
                    elif umur == 30:
                        Z_score = (TB_Saatini - 91.9) / (91.9 - 88.5)
                    elif umur == 31:
                        Z_score = (TB_Saatini - 92.7) / (92.7 - 89.2)
                    elif umur == 32:
                        Z_score = (TB_Saatini - 93.4) / (93.4 - 89.9)
                    elif umur == 33:
                        Z_score = (TB_Saatini - 94.1) / (94.1 - 90.5)
                    elif umur == 34:
                        Z_score = (TB_Saatini - 94.8) / (94.8 - 91.1)
                    elif umur == 35:
                        Z_score = (TB_Saatini - 95.4) / (95.4 - 91.8)
                    elif umur == 36:
                        Z_score = (TB_Saatini - 96.1) / (96.1 - 92.4)
                    elif umur == 37:
                        Z_score = (TB_Saatini - 96.7) / (96.7 - 93.0)
                    elif umur == 38:
                        Z_score = (TB_Saatini - 97.4) / (97.4 - 93.6)
                    elif umur == 39:
                        Z_score = (TB_Saatini - 98.0) / (98.0 - 94.2)
                    elif umur == 40:
                        Z_score = (TB_Saatini - 98.6) / (98.6 - 94.7)
                    elif umur == 41:
                        Z_score = (TB_Saatini - 99.2) / (99.2 - 95.3)
                    elif umur == 42:
                        Z_score = (TB_Saatini - 99.9) / (99.9 - 95.9)
                    elif umur == 43:
                        Z_score = (TB_Saatini - 100.4) / (100.4 - 96.4)
                    elif umur == 44:
                        Z_score = (TB_Saatini - 101.0) / (101.0 - 97.0)
                    elif umur == 45:
                        Z_score = (TB_Saatini - 101.6) / (101.6 - 97.5)
                    elif umur == 46:
                        Z_score = (TB_Saatini - 102.2) / (102.2 - 98.1)
                    elif umur == 47:
                        Z_score = (TB_Saatini - 102.8) / (102.8 - 98.6)
                    elif umur == 48:
                        Z_score = (TB_Saatini - 103.3) / (103.3 - 99.1)
                    elif umur == 59:
                        Z_score = (TB_Saatini - 103.9) / (103.9 - 99.7)
                    elif umur == 50:
                        Z_score = (TB_Saatini - 104.4) / (104.4 - 100.2)
                    elif umur == 51:
                        Z_score = (TB_Saatini - 105.0) / (105.0 - 100.7)
                    elif umur == 52:
                        Z_score = (TB_Saatini - 105.6) / (105.6 - 101.2)
                    elif umur == 53:
                        Z_score = (TB_Saatini - 106.1) / (106.1 - 101.7)
                    elif umur == 54:
                        Z_score = (TB_Saatini - 106.7) / (106.7 - 102.3)
                    elif umur == 55:
                        Z_score = (TB_Saatini - 107.2) / (107.2 - 102.8)
                    elif umur == 56:
                        Z_score = (TB_Saatini - 107.8) / (107.8 - 103.3)
                    elif umur == 57:
                        Z_score = (TB_Saatini - 108.3) / (108.3 - 103.8)
                    elif umur == 58:
                        Z_score = (TB_Saatini - 108.9) / (108.9 - 104.3)
                    elif umur == 59:
                        Z_score = (TB_Saatini - 109.4) / (109.4 - 104.8)
                    elif umur == 60:
                        Z_score = (TB_Saatini - 110.0) / (110.0 - 105.3)
                    else:
                        Z_score = (TB_Saatini - 110.0) / (110.0 - 105.3)

                    # Tentukan Status berdasarkan Z_score
                    if Z_score < -3:
                        return "Sangat Pendek"
                    elif -3 < Z_score < -2:
                        return "Pendek"
                    elif -2 < Z_score < 3:
                        return "Normal"
                    else:
                        return "Tinggi"

                # Perempuan
                elif jenis_kelamin == "Perempuan":
                    if umur == 0:
                        return None
                    elif umur == 1:
                        Z_score = (TB_Saatini - 53.7) / (53.7 - 51.7)
                    elif umur == 2:
                        Z_score = (TB_Saatini - 57.1) / (57.1 - 55.0)
                    elif umur == 3:
                        Z_score = (TB_Saatini - 59.8) / (59.8 - 57.0)
                    elif umur == 4:
                        Z_score = (TB_Saatini - 62.1) / (62.1 - 59.9)
                    elif umur == 5:
                        Z_score = (TB_Saatini - 64.0) / (64.0 - 61.8)
                    elif umur == 6:
                        Z_score = (TB_Saatini - 65.7) / (65.7 - 63.5)
                    elif umur == 7:
                        Z_score = (TB_Saatini - 67.3) / (67.3 - 65.0)
                    elif umur == 8:
                        Z_score = (TB_Saatini - 68.7) / (68.7 - 66.4)
                    elif umur == 9:
                        Z_score = (TB_Saatini - 70.1) / (70.1 - 67.7)
                    elif umur == 10:
                        Z_score = (TB_Saatini - 71.5) / (71.5 - 69.0)
                    elif umur == 11:
                        Z_score = (TB_Saatini - 72.8) / (72.8 - 70.3)
                    elif umur == 12:
                        Z_score = (TB_Saatini - 74.0) / (74.0 - 71.4)
                    elif umur == 13:
                        Z_score = (TB_Saatini - 75.2) / (75.2 - 72.6)
                    elif umur == 14:
                        Z_score = (TB_Saatini - 76.4) / (76.4 - 73.7)
                    elif umur == 15:
                        Z_score = (TB_Saatini - 77.5) / (77.5 - 74.8)
                    elif umur == 16:
                        Z_score = (TB_Saatini - 78.6) / (78.6 - 75.8)
                    elif umur == 17:
                        Z_score = (TB_Saatini - 79.7) / (79.7 - 76.8)
                    elif umur == 18:
                        Z_score = (TB_Saatini - 80.7) / (80.7 - 77.8)
                    elif umur == 19:
                        Z_score = (TB_Saatini - 81.7) / (81.7 - 78.8)
                    elif umur == 20:
                        Z_score = (TB_Saatini - 82.7) / (82.7 - 79.7)
                    elif umur == 21:
                        Z_score = (TB_Saatini - 83.7) / (83.7 - 80.6)
                    elif umur == 22:
                        Z_score = (TB_Saatini - 84.6) / (84.6 - 81.5)
                    elif umur == 23:
                        Z_score = (TB_Saatini - 53.7) / (53.7 - 82.3)
                    elif umur == 24:
                        Z_score = (TB_Saatini - 86.4) / (86.4 - 83.2)
                    elif umur == 25:
                        Z_score = (TB_Saatini - 86.6) / (86.6 - 83.3)
                    elif umur == 26:
                        Z_score = (TB_Saatini - 87.4) / (87.4 - 84.1)
                    elif umur == 27:
                        Z_score = (TB_Saatini - 84.9) / (88.3 - 84.9)
                    elif umur == 28:
                        Z_score = (TB_Saatini - 89.1) / (89.1 - 85.7)
                    elif umur == 29:
                        Z_score = (TB_Saatini - 89.9) / (89.9 - 86.4)
                    elif umur == 30:
                        Z_score = (TB_Saatini - 90.7) / (90.7 - 87.1)
                    elif umur == 31:
                        Z_score = (TB_Saatini - 91.4) / (91.4 - 87.9)
                    elif umur == 32:
                        Z_score = (TB_Saatini - 92.2) / (92.2 - 88.6)
                    elif umur == 33:
                        Z_score = (TB_Saatini - 92.9) / (92.9 - 89.3)
                    elif umur == 34:
                        Z_score = (TB_Saatini - 93.6) / (93.6 - 89.9)
                    elif umur == 35:
                        Z_score = (TB_Saatini - 94.4) / (94.4 - 90.6)
                    elif umur == 36:
                        Z_score = (TB_Saatini - 95.1) / (95.1 - 91.2)
                    elif umur == 37:
                        Z_score = (TB_Saatini - 95.7) / (95.7 - 91.9)
                    elif umur == 38:
                        Z_score = (TB_Saatini - 96.4) / (96.4 - 92.5)
                    elif umur == 39:
                        Z_score = (TB_Saatini - 97.1) / (97.1 - 93.1)
                    elif umur == 40:
                        Z_score = (TB_Saatini - 97.7) / (97.7 - 93.8)
                    elif umur == 41:
                        Z_score = (TB_Saatini - 98.4) / (98.4 - 94.4)
                    elif umur == 42:
                        Z_score = (TB_Saatini - 99.0) / (99.0 - 95.0)
                    elif umur == 43:
                        Z_score = (TB_Saatini - 99.7) / (99.7 - 95.6)
                    elif umur == 44:
                        Z_score = (TB_Saatini - 100.3) / (100.3 - 96.2)
                    elif umur == 45:
                        Z_score = (TB_Saatini - 100.9) / (100.9 - 96.7)
                    elif umur == 46:
                        Z_score = (TB_Saatini - 101.5) / (101.5 - 97.3)
                    elif umur == 47:
                        Z_score = (TB_Saatini - 102.1) / (102.1 - 97.9)
                    elif umur == 48:
                        Z_score = (TB_Saatini - 102.7) / (102.7 - 98.4)
                    elif umur == 49:
                        Z_score = (TB_Saatini - 103.3) / (103.3 - 99.0)
                    elif umur == 50:
                        Z_score = (TB_Saatini - 103.9) / (103.9 - 99.5)
                    elif umur == 51:
                        Z_score = (TB_Saatini - 104.5) / (104.5 - 100.1)
                    elif umur == 52:
                        Z_score = (TB_Saatini - 105.0) / (105.0 - 100.6)
                    elif umur == 53:
                        Z_score = (TB_Saatini - 105.6) / (105.6 - 101.1)
                    elif umur == 54:
                        Z_score = (TB_Saatini - 106.2) / (106.2 - 101.6)
                    elif umur == 55:
                        Z_score = (TB_Saatini - 106.7) / (106.7 - 102.2)
                    elif umur == 56:
                        Z_score = (TB_Saatini - 107.3) / (107.3 - 102.7)
                    elif umur == 57:
                        Z_score = (TB_Saatini - 107.8) / (107.8 - 103.2)
                    elif umur == 58:
                        Z_score = (TB_Saatini - 108.4) / (108.4 - 103.7)
                    elif umur == 59:
                        Z_score = (TB_Saatini - 108.9) / (108.9 - 104.2)
                    elif umur == 60:
                        Z_score = (TB_Saatini - 109.4) / (109.4 - 104.7)
                    else:
                        return None  # Umur di luar rentang yang ditangani

                    # Tentukan Status berdasarkan Z_score
                    if Z_score < -3:
                        return "Sangat Pendek"
                    elif -3 < Z_score < -2:
                        return "Pendek"
                    elif -2 < Z_score < 3:
                        return "Normal"
                    else:
                        return "Tinggi"
                else:
                    return None  # Kondisi untuk jenis kelamin selain 'Laki-laki' tidak ditangani

            status = hitung_status_umur(jenis_kelamin, int(umur), TB_Saatini)

            # Menampilkan Status jika hasil tidak None
            if status is not None:
                if status == "Normal":
                    status_gizi = 1
                elif status == "Pendek":
                    status_gizi = 2
                elif status == "Sangat Pendek":
                    status_gizi = 3
                elif status == "Tinggi":
                    status_gizi = 4

                if jenis_kelamin == "Laki-laki":
                    jk = 1
                else:
                    jk = 0

                inputs = np.array([status_gizi, TB_Saatini, BB_Saatini, umur])
                # Normalisasi data input
                df_min = X_new.iloc[:, 0:4].min()
                df_max = X_new.iloc[:, 0:4].max()
                input_norm = (inputs - df_min) / (df_max - df_min)
                input_norm = pd.DataFrame(np.array(input_norm).reshape(1, -1))

                # menampilkan prediksi
                input_pred = clf.predict(input_norm)
                st.subheader("Hasil Penentuan Status Stunting")
                st.write("Nama        :", nama)
                st.write("Status Gizi :", status)
                st.write("Diprediksi  :")
                if input_pred == 1:
                    st.success("Normal")
                else:
                    st.error("Stunting")
