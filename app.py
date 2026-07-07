# =====================================================
# IMPORT LIBRARY
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Glioma Grading Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Glioma Grading Classification")
st.write("Machine Learning untuk Klasifikasi Grade Glioma")

# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv("TCGA_InfoWithGrade.csv")

    return df

df = load_data()

# =====================================================
# PREPROCESSING
# =====================================================

# Menghapus kolom ID jika ada
if "Case_ID" in df.columns:
    df = df.drop(columns=["Case_ID"])

# Memisahkan fitur dan target
X = df.drop("Grade", axis=1)
y = df["Grade"]

# Mengubah data kategorikal menjadi numerik
X = pd.get_dummies(X)

# Label Encoder target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Standardisasi
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.selectbox(

    "Pilih Menu",

    [

        "Dataset",

        "Machine Learning",

        "Tentang"

    ]

)

# =====================================================
# HALAMAN DATASET
# =====================================================

if menu == "Dataset":

    st.header("📊 Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", df.shape[0])

    col2.metric("Jumlah Fitur", df.shape[1]-1)

    col3.metric("Jumlah Kelas", df["Grade"].nunique())

    st.subheader("Preview Dataset")

    st.dataframe(df.head())

    st.subheader("Informasi Dataset")

    st.write(df.describe())

    st.subheader("Distribusi Grade")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(data=df, x="Grade", ax=ax)

    st.pyplot(fig)

# =====================================================
# HALAMAN MACHINE LEARNING
# =====================================================

elif menu == "Machine Learning":

    st.header("🤖 Machine Learning")

    model_name = st.selectbox(

        "Pilih Model",

        [

            "Logistic Regression",

            "Decision Tree",

            "Random Forest"

        ]

    )

    # ===============================
    # MEMILIH MODEL
    # ===============================

    if model_name == "Logistic Regression":

        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":

        model = DecisionTreeClassifier(random_state=42)

    else:

        model = RandomForestClassifier(random_state=42)

    # ===============================
    # TRAIN MODEL
    # ===============================

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    # ===============================
    # HASIL AKURASI
    # ===============================

    st.success(

        f"Akurasi Model : {acc*100:.2f}%"

    )

    # ===============================
    # CONFUSION MATRIX
    # ===============================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(

        y_test,

        pred

    )

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        ax=ax

    )

    ax.set_xlabel("Prediksi")

    ax.set_ylabel("Aktual")

    st.pyplot(fig)

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================

    st.subheader("Classification Report")

    report = classification_report(

        y_test,

        pred,

        target_names=encoder.classes_,

        output_dict=True

    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # ===============================
    # SIMPAN MODEL
    # ===============================

    st.session_state["model"] = model

# =====================================================
# HALAMAN PREDIKSI
# =====================================================

elif menu == "Prediksi":

    st.header("🔮 Prediksi Grade Glioma")

    if "model" not in st.session_state:

        st.warning("Silakan jalankan Machine Learning terlebih dahulu.")

    else:

        model = st.session_state["model"]

        st.write("Masukkan nilai setiap fitur")

        input_data = {}

        # Menggunakan data asli sebelum preprocessing
        fitur = df.drop("Grade", axis=1)

        for kolom in fitur.columns:

            if fitur[kolom].dtype == "object":

                pilihan = fitur[kolom].unique()

                nilai = st.selectbox(kolom, pilihan)

            else:

                nilai = st.number_input(

                    kolom,

                    value=float(fitur[kolom].mean())

                )

            input_data[kolom] = nilai

        if st.button("Prediksi"):

            input_df = pd.DataFrame([input_data])

            # One Hot Encoding
            input_df = pd.get_dummies(input_df)

            # Menyamakan kolom dengan data training
            X_dummy = pd.get_dummies(df.drop("Grade", axis=1))

            input_df = input_df.reindex(
                columns=X_dummy.columns,
                fill_value=0
            )

            # Standardisasi
            input_scaled = scaler.transform(input_df)

            # Prediksi
            hasil = model.predict(input_scaled)

            hasil_label = encoder.inverse_transform(hasil)

            st.success(
                f"Hasil Prediksi : **{hasil_label[0]}**"
            )


# =====================================================
# HALAMAN TENTANG
# =====================================================

elif menu == "Tentang":

    st.header("ℹ️ Tentang")

    st.write("""
Aplikasi ini dibuat untuk melakukan klasifikasi
**Glioma Grading** menggunakan algoritma
Machine Learning.

### Dataset
TCGA Glioma Grading Clinical and Mutation Features

### Algoritma

- Logistic Regression
- Decision Tree
- Random Forest

### Library

- Streamlit
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

### Dibuat Untuk

Responsi Praktikum Data Sains for Health
""")

    st.divider()

    st.success("Terima kasih telah menggunakan aplikasi ini.")
