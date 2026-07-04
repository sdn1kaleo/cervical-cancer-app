import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tempfile
import os
import h5py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Cervical Cancer Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1 { color: #c0392b; }
    h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Cervical Cancer Risk Prediction")
st.markdown("Aplikasi prediksi risiko kanker serviks menggunakan Machine Learning.")
st.markdown("---")

# ── CNN Wrapper & Loader ───────────────────────────────────────────────────────
class KerasCNNWrapper:
    """Membungkus model Keras CNN agar punya interface predict/predict_proba
    seperti model sklearn, sehingga bisa dipakai langsung di seluruh dashboard
    tanpa mengubah kode evaluasi/prediksi yang sudah ada."""

    def __init__(self, keras_model):
        self.model = keras_model
        self.classes_ = np.array([0, 1])

    def _reshape(self, X):
        X_arr = np.asarray(X)
        return X_arr.reshape(X_arr.shape[0], X_arr.shape[1], 1)

    def predict_proba(self, X):
        prob_pos = self.model.predict(self._reshape(X), verbose=0).flatten()
        prob_pos = np.clip(prob_pos, 0.0, 1.0)
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def load_cnn_model(uploaded_file):
    """Load model CNN dari file yang diupload user.
    Mendukung 3 kemungkinan format:
      1. File .h5 / .keras asli
      2. File .pkl yang sebenarnya berformat HDF5 (rename biasa)
      3. File .pkl hasil pickle.dump(model) atau pickle.dump({'architecture', 'weights'})

    NOTE: tensorflow di-import di dalam fungsi ini (lazy import), bukan di
    top-level file, supaya app tidak wajib me-load TensorFlow (yang berat dan
    boros memori) saat user hanya memakai model sklearn biasa (Random Forest,
    Gradient Boosting, dll). Ini penting terutama di Streamlit Community Cloud
    yang resource RAM-nya terbatas.
    """
    from tensorflow.keras.models import load_model as keras_load_model, model_from_json

    suffix = os.path.splitext(uploaded_file.name)[1] or ".h5"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # Kasus 1 & 2: file berformat HDF5 (baik ekstensi .h5/.keras maupun .pkl)
        if h5py.is_hdf5(tmp_path):
            keras_model = keras_load_model(tmp_path)
            return KerasCNNWrapper(keras_model)

        # Kasus 3: file hasil pickle.dump()
        with open(tmp_path, "rb") as f:
            obj = pickle.load(f)

        # 3a. Objek yang sudah punya method predict (misal berhasil di-pickle utuh)
        if hasattr(obj, "predict"):
            return KerasCNNWrapper(obj)

        # 3b. Dict berisi arsitektur (json) + bobot
        if isinstance(obj, dict) and "architecture" in obj and "weights" in obj:
            keras_model = model_from_json(obj["architecture"])
            keras_model.set_weights(obj["weights"])
            return KerasCNNWrapper(keras_model)

        raise ValueError(
            "Format file model CNN tidak dikenali. Pastikan file disimpan dengan "
            "model.save('nama.h5') atau pickle.dump({'architecture': model.to_json(), "
            "'weights': model.get_weights()}, f)."
        )
    finally:
        os.remove(tmp_path)


# ── Load & Preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    df.replace("?", np.nan, inplace=True)
    # Drop columns with too many missing values
    drop_cols = ["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    # Convert all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def prepare_model_data(df):
    target = "Biopsy"
    X = df.drop(columns=[target, "Hinselmann", "Schiller", "Citology"], errors="ignore")
    y = df[target]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imp, y

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded = st.file_uploader("Upload Dataset CSV", type=["csv"])
    st.markdown("---")
    model_choice = st.selectbox(
        "Pilih Algoritma",
        ["Random Forest ⭐ (Rekomendasi)", "Gradient Boosting", "Logistic Regression",
         "SVM", "CNN (Deep Learning)"]
    )

    cnn_file = None
    cnn_use_scaling = False
    if "CNN" in model_choice:
        st.markdown("**📦 Upload Model CNN**")
        cnn_file = st.file_uploader(
            "File model CNN (.pkl / .h5 / .keras)",
            type=["pkl", "h5", "keras"],
            key="cnn_uploader"
        )
        cnn_use_scaling = st.checkbox(
            "Model dilatih dengan StandardScaler",
            value=False,
            help=("Kosongkan (default) jika CNN dilatih langsung dari data mentah "
                  "seperti pada skrip training awal (X_train.values). Centang jika "
                  "kamu men-scale data sebelum training CNN.")
        )

    st.markdown("---")
    test_size = st.slider("Ukuran Test Set (%)", 10, 40, 20) / 100
    st.markdown("---")
    st.markdown("**ℹ️ Info Dataset**")
    st.markdown("- 858 pasien\n- 36 fitur risiko\n- Target: Biopsy (0/1)")

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded:
    df = load_and_preprocess(uploaded)
else:
    df = load_and_preprocess("risk_factors_cervical_cancer.csv")

X, y = prepare_model_data(df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Eksplorasi Data", "🤖 Model & Evaluasi", "📈 Feature Importance", "🔍 Prediksi Manual"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pasien", len(df))
    col2.metric("Total Fitur", len(X.columns))
    col3.metric("Kasus Positif", int(y.sum()))
    col4.metric("Kasus Negatif", int((y == 0).sum()))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Distribusi Target (Biopsy)**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = y.value_counts()
        bars = ax.bar(["Negatif (0)", "Positif (1)"], counts.values,
                      color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(val), ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel("Jumlah Pasien")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    with col_b:
        st.markdown("**Distribusi Usia**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df["Age"].dropna(), bins=20, color="#3498db", edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Usia")
        ax.set_ylabel("Frekuensi")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    st.markdown("**Statistik Deskriptif**")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pelatihan & Evaluasi Model")

    @st.cache_resource
    def train_model(model_name, test_sz, _cnn_file, cnn_scaling):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_sz, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # ── Cabang khusus CNN: load model pretrained, tidak training ulang ──
        if "CNN" in model_name:
            if _cnn_file is None:
                return None, scaler, X_train, X_test, y_train, y_test, None, None, None

            model = load_cnn_model(_cnn_file)
            X_train_input = X_train_s if cnn_scaling else X_train.values
            X_test_input  = X_test_s  if cnn_scaling else X_test.values

            y_pred = model.predict(X_test_input)
            y_prob = model.predict_proba(X_test_input)[:, 1]
            # CNN sudah pretrained -> tidak di-cross-validate ulang, hanya evaluasi test set
            cv_scores = np.array([roc_auc_score(y_test, y_prob)])
            return model, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob, cv_scores

        # ── Cabang model sklearn (seperti semula) ──
        if "Random Forest" in model_name:
            model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                           class_weight="balanced", random_state=42)
        elif "Gradient Boosting" in model_name:
            model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                               max_depth=4, random_state=42)
        elif "Logistic" in model_name:
            model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        else:
            model = SVC(class_weight="balanced", probability=True, kernel="rbf", random_state=42)

        model.fit(X_train_s, y_train)
        y_pred  = model.predict(X_test_s)
        y_prob  = model.predict_proba(X_test_s)[:, 1]
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc")

        return model, scaler, X_train, X_test, y_train, y_test, y_pred, y_prob, cv_scores

    with st.spinner("Memuat / melatih model…"):
        (model, scaler, X_train, X_test, y_train, y_test,
         y_pred, y_prob, cv_scores) = train_model(model_choice, test_size, cnn_file, cnn_use_scaling)

    if model is None:
        st.warning("⚠️ Silakan upload file model CNN (.pkl / .h5 / .keras) terlebih dahulu di sidebar untuk melanjutkan.")
        st.stop()

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall",    f"{rec:.2%}")
    m4.metric("F1-Score",  f"{f1:.2%}")
    m5.metric("ROC-AUC",   f"{auc:.3f}")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["Negatif", "Positif"],
                    yticklabels=["Negatif", "Positif"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        st.pyplot(fig)

    with col_d:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)

    with st.expander("📋 Laporan Klasifikasi Lengkap"):
        report = classification_report(y_test, y_pred, target_names=["Negatif","Positif"])
        st.code(report)
        if "CNN" in model_choice:
            st.info(f"ROC-AUC pada test set: {cv_scores.mean():.3f} "
                    "(model CNN pretrained, tidak melalui cross-validation ulang).")
        else:
            st.info(f"Cross-Validation ROC-AUC (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Feature Importance
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#e74c3c" if v >= imp.quantile(0.75) else "#3498db" for v in imp.values]
        imp.plot.barh(ax=ax, color=colors)
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 15 Fitur Paling Berpengaruh")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        st.caption("🔴 Merah = fitur paling penting | 🔵 Biru = fitur pendukung")
    elif hasattr(model, "coef_"):
        coef = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        coef.plot.barh(ax=ax, color="#9b59b6")
        ax.set_xlabel("Koefisien Absolut")
        ax.set_title("Top 15 Fitur (Logistic Regression)")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
    elif "CNN" in model_choice:
        st.info(
            "Feature importance tidak tersedia secara langsung untuk model CNN. "
            "Untuk interpretasi fitur pada deep learning, pertimbangkan menggunakan "
            "teknik tambahan seperti SHAP atau permutation importance."
        )
    else:
        st.info("Feature importance tidak tersedia untuk SVM. Gunakan Random Forest atau Gradient Boosting.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Prediksi Manual
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Prediksi Risiko Individu")
    st.markdown("Masukkan data pasien untuk memprediksi risiko kanker serviks.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Usia", 10, 90, 30)
            num_partners = st.number_input("Jumlah Pasangan Seksual", 0, 30, 2)
            first_sex = st.number_input("Usia Pertama Berhubungan Seksual", 10, 50, 18)
            num_preg = st.number_input("Jumlah Kehamilan", 0, 15, 1)
        with c2:
            smokes = st.selectbox("Merokok", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            smokes_years = st.number_input("Lama Merokok (tahun)", 0.0, 50.0, 0.0)
            hormonal = st.selectbox("Kontrasepsi Hormonal", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            hormonal_years = st.number_input("Lama Kontrasepsi Hormonal (tahun)", 0.0, 30.0, 0.0)
        with c3:
            iud = st.selectbox("IUD", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            iud_years = st.number_input("Lama IUD (tahun)", 0.0, 30.0, 0.0)
            stds = st.selectbox("Riwayat STDs", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            stds_num = st.number_input("Jumlah STDs", 0, 10, 0)

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

    if submitted:
        # Build a row matching X columns; fill unknowns with 0
        row = {col: 0.0 for col in X.columns}
        row.update({
            "Age": age,
            "Number of sexual partners": num_partners,
            "First sexual intercourse": first_sex,
            "Num of pregnancies": num_preg,
            "Smokes": smokes,
            "Smokes (years)": smokes_years,
            "Hormonal Contraceptives": hormonal,
            "Hormonal Contraceptives (years)": hormonal_years,
            "IUD": iud,
            "IUD (years)": iud_years,
            "STDs": stds,
            "STDs (number)": stds_num,
        })
        input_df = pd.DataFrame([row])[X.columns]

        # Untuk CNN, ikuti pengaturan scaling yang sama seperti saat evaluasi
        if "CNN" in model_choice and not cnn_use_scaling:
            input_processed = input_df.values
        else:
            input_processed = scaler.transform(input_df)

        prob = model.predict_proba(input_processed)[0][1]
        pred = model.predict(input_processed)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **Risiko TINGGI** terdeteksi — Probabilitas: **{prob:.1%}**")
            st.markdown("Disarankan untuk segera melakukan pemeriksaan lanjutan ke dokter spesialis.")
        else:
            st.success(f"✅ **Risiko RENDAH** — Probabilitas positif: **{prob:.1%}**")
            st.markdown("Tetap lakukan pemeriksaan rutin secara berkala.")

        # Gauge bar
        fig, ax = plt.subplots(figsize=(6, 0.8))
        ax.barh(0, 1, color="#ecf0f1", height=0.5)
        ax.barh(0, prob, color="#e74c3c" if prob > 0.5 else "#2ecc71", height=0.5)
        ax.set_xlim(0, 1); ax.axis("off")
        ax.text(prob, 0, f" {prob:.1%}", va="center", fontsize=12, fontweight="bold")
        ax.set_title("Probabilitas Risiko", fontsize=11)
        st.pyplot(fig)

st.markdown("---")
st.caption("⚕️ Aplikasi ini hanya untuk keperluan edukasi/penelitian. Bukan pengganti diagnosis medis profesional.")
