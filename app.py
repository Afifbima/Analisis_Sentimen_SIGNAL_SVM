import streamlit as st
import pandas as pd
import joblib
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

st.set_page_config(
    page_title="Analisis Sentimen SIGNAL - SVM",
    layout="wide",
)

st.title("Analisis Sentimen Ulasan Aplikasi SIGNAL")
st.caption("Menggunakan metode Support Vector Machine (SVM) dengan kernel Linear dan RBF")

# ── Path helpers ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_resources():
    """Load semua model dan preprocessing resources."""
    svm_linear = joblib.load(os.path.join(BASE_DIR, "svm_linear_model.pkl"))
    svm_rbf = joblib.load(os.path.join(BASE_DIR, "svm_rbf_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    labels = joblib.load(os.path.join(BASE_DIR, "sentiment_labels.pkl"))

    stop_words = set(stopwords.words("indonesian"))

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    kamus_path = os.path.join(BASE_DIR, "kamuskatabaku.xlsx")
    try:
        kamus_df = pd.read_excel(kamus_path)
        kamus_norm = dict(zip(kamus_df["tidak_baku"], kamus_df["kata_baku"]))
    except Exception:
        kamus_norm = {}

    return {
        "svm_linear": svm_linear,
        "svm_rbf": svm_rbf,
        "vectorizer": vectorizer,
        "labels": labels,
        "stop_words": stop_words,
        "stemmer": stemmer,
        "kamus_norm": kamus_norm,
    }


# ── Preprocessing functions ──

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\d", "", text)
    return text


def case_folding(text: str) -> str:
    return text.lower() if isinstance(text, str) else ""


def normalize_text(text: str, kamus: dict) -> str:
    if not isinstance(text, str):
        return ""
    words = text.split()
    result = []
    for w in words:
        if w in kamus:
            baku = kamus[w]
            if isinstance(baku, str) and baku.isalpha():
                result.append(baku)
            else:
                result.append(w)
        else:
            result.append(w)
    return " ".join(result)


def remove_stopwords(tokens: list, stop_words: set) -> list:
    return [w for w in tokens if w not in stop_words]


def stem_tokens(tokens: list, stemmer) -> list:
    return [stemmer.stem(w) for w in tokens]


def preprocess_text(text: str, res: dict) -> str:
    """Jalankan seluruh pipeline preprocessing dan kembalikan teks final."""
    cleaned = clean_text(text)
    folded = case_folding(cleaned)
    normalized = normalize_text(folded, res["kamus_norm"])
    tokens = normalized.split() if isinstance(normalized, str) else []
    tokens_no_sw = remove_stopwords(tokens, res["stop_words"])
    stemmed = stem_tokens(tokens_no_sw, res["stemmer"])
    final_text = " ".join(stemmed)
    return final_text


def predict_sentiment(text: str, res: dict) -> tuple:
    """Prediksi sentimen dengan kedua model. Return (linear_sentiment, rbf_sentiment)."""
    final_text = preprocess_text(text, res)
    tfidf = res["vectorizer"].transform([final_text])

    # Prediksi dengan SVM Linear
    linear_pred = res["svm_linear"].predict(tfidf)[0]

    # Prediksi dengan SVM RBF
    rbf_pred = res["svm_rbf"].predict(tfidf)[0]

    return linear_pred, rbf_pred


# ── Load resources ──
try:
    resources = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info(
        "Pastikan file berikut ada di folder yang sama dengan app_simple.py:\n"
        "- svm_linear_model.pkl\n"
        "- svm_rbf_model.pkl\n"
        "- tfidf_vectorizer.pkl\n"
        "- sentiment_labels.pkl\n"
        "- kamuskatabaku.xlsx"
    )
    st.stop()

# ── Initialize session state ──
if "current_data" not in st.session_state:
    # Load data dari CSV saat pertama kali
    csv_path = os.path.join(BASE_DIR, "Hasil_Prediksi_SVM_Linear_RBF.csv")
    if os.path.exists(csv_path):
        st.session_state.current_data = pd.read_csv(csv_path)
        st.session_state.data_source = "csv"
    else:
        st.session_state.current_data = pd.DataFrame(columns=["No", "Ulasan", "Sentiment(Linear)", "Sentiment(RBF)"])
        st.session_state.data_source = "empty"

# ── Input Section ──
st.markdown("---")
st.subheader("Analisis Ulasan Baru")

col_input, col_button = st.columns([4, 1])

with col_input:
    new_text = st.text_area(
        "Masukkan ulasan baru:",
        placeholder="Contoh: Aplikasi ini sangat mudah digunakan dan prosesnya cepat...",
        height=100,
        label_visibility="collapsed",
    )

with col_button:
    st.write("")  # Spacer
    st.write("")  # Spacer
    analyze_btn = st.button("Analisis", type="primary", use_container_width=True)

# ── Reset Button ──
if st.session_state.data_source == "prediction":
    if st.button("↻ Kembali ke Data CSV", use_container_width=True):
        csv_path = os.path.join(BASE_DIR, "Hasil_Prediksi_SVM_Linear_RBF.csv")
        if os.path.exists(csv_path):
            st.session_state.current_data = pd.read_csv(csv_path)
            st.session_state.data_source = "csv"
            st.rerun()

# ── Process New Input ──
if analyze_btn and new_text.strip():
    with st.spinner("Menganalisis..."):
        linear_sentiment, rbf_sentiment = predict_sentiment(new_text, resources)

    # Buat dataframe baru dengan hasil prediksi
    new_df = pd.DataFrame({
        "No": [1],
        "Ulasan": [new_text],
        "Sentiment(Linear)": [linear_sentiment],
        "Sentiment(RBF)": [rbf_sentiment]
    })

    # Override data di session state
    st.session_state.current_data = new_df
    st.session_state.data_source = "prediction"
    st.rerun()

# ── Display Results Table ──
st.markdown("---")
st.subheader("Hasil Prediksi")

if st.session_state.data_source == "csv":
    st.caption(f"Menampilkan data dari CSV - Total: {len(st.session_state.current_data)} ulasan")
else:
    st.caption("Menampilkan hasil prediksi baru")

# Fungsi untuk mewarnai sentimen
def color_sentiment(val):
    if val == "positif":
        return "color: green; font-weight: bold;"
    elif val == "negatif":
        return "color: red; font-weight: bold;"
    return ""

# Apply styling ke dataframe
styled_df = st.session_state.current_data.style.applymap(
    color_sentiment,
    subset=["Sentiment(Linear)", "Sentiment(RBF)"]
)

# Tampilkan tabel dengan pagination
st.dataframe(
    styled_df,
    use_container_width=True,
    height=600,
    column_config={
        "No": st.column_config.NumberColumn("No", width="small"),
        "Ulasan": st.column_config.TextColumn("Ulasan", width="large"),
        "Sentiment(Linear)": st.column_config.TextColumn("Sentiment(Linear)", width="medium"),
        "Sentiment(RBF)": st.column_config.TextColumn("Sentiment(RBF)", width="medium"),
    }
)

# ── Summary Statistics (hanya untuk data CSV) ──
if st.session_state.data_source == "csv" and len(st.session_state.current_data) > 0:
    st.markdown("---")
    st.subheader("Statistik Sentimen")

    df = st.session_state.current_data

    # Statistik Linear
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Ulasan", len(df))

    linear_pos = len(df[df["Sentiment(Linear)"] == "positif"])
    linear_neg = len(df[df["Sentiment(Linear)"] == "negatif"])

    with col2:
        st.metric("Positif (Linear)", linear_pos)
    with col3:
        st.metric("Negatif (Linear)", linear_neg)

    # Statistik RBF
    col4, col5, col6 = st.columns(3)
    with col4:
        st.write("")  # Spacer

    rbf_pos = len(df[df["Sentiment(RBF)"] == "positif"])
    rbf_neg = len(df[df["Sentiment(RBF)"] == "negatif"])

    with col5:
        st.metric("Positif (RBF)", rbf_pos)
    with col6:
        st.metric("Negatif (RBF)", rbf_neg)
