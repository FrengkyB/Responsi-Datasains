# =====================================================
# IMPORT LIBRARY
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Glioma Grading Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Glioma Grading Classification")
st.markdown("### Responsi Praktikum Data Sains for Health")
st.divider()

# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    glioma = fetch_ucirepo(id=759)

    X = glioma.data.features
    y = glioma.data.targets

    return X, y

X, y = load_dataset()

# =====================================================
# PREPROCESSING
# =====================================================

# Target menjadi angka
target = y.iloc[:, 0]

encoder = LabelEncoder()
target = encoder.fit_transform(target)

# One Hot Encoding
X = pd.get_dummies(X)

# Mengatasi missing value
X = X.fillna(X.mean(numeric_only=True))

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.selectbox(
    "Pilih Halaman",
    [
        "Dataset",
        "Machine Learning",
        "Prediksi",
        "Tentang"
    ]
)

# =====================================================
# HALAMAN DATASET
# =====================================================

if menu == "Dataset":

    st.header("📊 Dataset Glioma")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Data",
        X.shape[0]
    )

    col2.metric(
        "Jumlah Fitur",
        X.shape[1]
    )

    col3.metric(
        "Jumlah Kelas",
        len(np.unique(target))
    )

    st.divider()

    st.subheader("Preview Dataset")

    st.dataframe(X.head())

    st.divider()

    st.subheader("Informasi Dataset")

    buffer = io.StringIO()

    X.info(buf=buffer)

    info = buffer.getvalue()

    st.text(info)

    st.divider()

    st.subheader("Statistik Dataset")

    st.dataframe(X.describe())

    st.divider()

    st.subheader("Missing Value")

    st.dataframe(
        X.isnull().sum().to_frame("Jumlah")
    )

    st.divider()

    st.subheader("Distribusi Target")

    fig, ax = plt.subplots(figsize=(6,4))

    pd.Series(target).value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Data")

    st.pyplot(fig)

    st.divider()

    st.subheader("Heatmap Korelasi")

    fig2, ax2 = plt.subplots(figsize=(12,8))

    sns.heatmap(
        X.corr(),
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2)
# =====================================================
# HALAMAN MACHINE LEARNING
# =====================================================

elif menu == "Machine Learning":

    st.header("🤖 Perbandingan Model Machine Learning")

    # Import model
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        ExtraTreesClassifier
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(random_state=42),

        "K-Nearest Neighbor":
            KNeighborsClassifier(),

        "Support Vector Machine":
            SVC(probability=True),

        "Naive Bayes":
            GaussianNB(),

        "Gradient Boosting":
            GradientBoostingClassifier(random_state=42),

        "Extra Trees":
            ExtraTreesClassifier(random_state=42)

    }

    hasil = []

    best_model = None
    best_name = ""
    best_accuracy = 0

    progress = st.progress(0)

    total = len(models)

    for i, (nama, model) in enumerate(models.items()):

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        hasil.append({

            "Model": nama,
            "Accuracy": acc

        })

        if acc > best_accuracy:

            best_accuracy = acc
            best_model = model
            best_name = nama

        progress.progress((i + 1) / total)

    hasil_df = pd.DataFrame(hasil)

    hasil_df = hasil_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    st.subheader("Perbandingan Akurasi")

    st.dataframe(hasil_df)

    st.divider()

    st.subheader("Grafik Akurasi")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(
        data=hasil_df,
        x="Accuracy",
        y="Model",
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    st.success(
        f"Model Terbaik : **{best_name}**\n\n"
        f"Akurasi : **{best_accuracy*100:.2f}%**"
    )

    # ============================================
    # Confusion Matrix
    # ============================================

    pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    st.subheader("Confusion Matrix")

    fig2, ax2 = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax2
    )

    ax2.set_xlabel("Prediksi")
    ax2.set_ylabel("Aktual")

    st.pyplot(fig2)

    # ============================================
    # Classification Report
    # ============================================

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        pred,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

    # ============================================
    # ROC CURVE
    # ============================================

    if hasattr(best_model, "predict_proba"):

        proba = best_model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(
            y_test,
            proba
        )

        roc_auc = auc(fpr, tpr)

        st.subheader("ROC Curve")

        fig3, ax3 = plt.subplots(figsize=(6,5))

        ax3.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}"
        )

        ax3.plot(
            [0,1],
            [0,1],
            "--"
        )

        ax3.legend()

        st.pyplot(fig3)

    # ============================================
    # Simpan model terbaik
    # ============================================

    st.session_state["model"] = best_model
    st.session_state["model_name"] = best_name
    st.session_state["accuracy"] = best_accuracy

# ============================================
# DASHBOARD HASIL
# ============================================

st.divider()

st.header("📊 Dashboard Hasil")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model Terbaik",
    best_name
)

col2.metric(
    "Accuracy",
    f"{best_accuracy*100:.2f}%"
)

col3.metric(
    "Jumlah Data",
    len(X)
)

# ============================================
# TAMPILKAN PREDIKSI
# ============================================

hasil_prediksi = pd.DataFrame({

    "Actual": y_test,
    "Prediction": pred

})

st.subheader("Hasil Prediksi")

st.dataframe(
    hasil_prediksi.head(20)
)

# ============================================
# DOWNLOAD CSV
# ============================================

csv = hasil_prediksi.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    "📥 Download Hasil Prediksi",

    csv,

    "hasil_prediksi.csv",

    "text/csv"

)

# ============================================
# HEATMAP KORELASI
# ============================================

st.subheader("Heatmap Korelasi")

fig4, ax4 = plt.subplots(figsize=(10,8))

sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    ax=ax4
)

st.pyplot(fig4)

# ============================================
# DISTRIBUSI TARGET
# ============================================

st.subheader("Distribusi Target")

fig5, ax5 = plt.subplots(figsize=(6,4))

sns.countplot(
    x=target,
    ax=ax5
)

st.pyplot(fig5)

elif menu == "Prediksi":

    st.header("🔮 Prediksi Glioma")

    if "model" not in st.session_state:

        st.warning(
            "Silakan jalankan Machine Learning terlebih dahulu."
        )

    else:

        model = st.session_state["model"]

        st.write(
            "Masukkan nilai seluruh fitur."
        )

        input_data = {}

        for kolom in X.columns:

            input_data[kolom] = st.number_input(

                kolom,

                value=float(X[kolom].mean())

            )

        if st.button("Prediksi"):

            input_df = pd.DataFrame(
                [input_data]
            )

            input_scaled = scaler.transform(
                input_df
            )

            hasil = model.predict(
                input_scaled
            )

            st.success(
                f"Hasil Prediksi : {hasil[0]}"
            )

            if hasattr(
                model,
                "predict_proba"
            ):

                prob = model.predict_proba(
                    input_scaled
                )[0]

                prob_df = pd.DataFrame({

                    "Kelas":
                        encoder.classes_,

                    "Probabilitas":
                        prob

                })

                st.subheader(
                    "Probabilitas"
                )

                st.dataframe(
                    prob_df
                )

                fig6, ax6 = plt.subplots(
                    figsize=(6,4)
                )

                sns.barplot(

                    data=prob_df,

                    x="Kelas",

                    y="Probabilitas",

                    ax=ax6

                )

                st.pyplot(fig6)

# =====================================================
# HALAMAN TENTANG
# =====================================================

elif menu == "Tentang":

    st.title("📚 Tentang Aplikasi")

    st.markdown("""
Aplikasi ini dibuat sebagai **Responsi Praktikum Data Sains for Health**.

### Tujuan
Membangun model Machine Learning untuk melakukan klasifikasi **Glioma Grading** menggunakan dataset dari **UCI Machine Learning Repository**.

### Algoritma yang Digunakan

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbor
- Support Vector Machine
- Naive Bayes
- Gradient Boosting
- Extra Trees

### Tahapan Analisis

1. Data Collection
2. Data Cleaning
3. Encoding
4. Standardization
5. Train Test Split
6. Model Training
7. Model Evaluation
8. Prediction

### Evaluasi Model

Model dievaluasi menggunakan:

- Accuracy
- Confusion Matrix
- Classification Report
- ROC Curve

Dataset :

https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features

Framework :

- Streamlit
- Scikit-Learn
- Pandas
- Matplotlib
- Seaborn

    """)

    st.divider()

    st.success("Terima kasih telah menggunakan aplikasi ini.")

    st.divider()

    st.caption("""
Responsi Praktikum Data Sains for Health

Universitas

2026

Dibuat menggunakan ❤️ Streamlit
""")
