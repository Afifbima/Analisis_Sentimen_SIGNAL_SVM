import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Analisis Sentimen Signal - SVM", layout="wide")
st.title("Analisis Sentimen Ulasan Aplikasi Signal")
st.markdown("Menggunakan metode **Support Vector Machine (SVM)** dengan **TF-IDF** dan **SMOTE**")

# ── Sidebar ──
st.sidebar.header("Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload CSV ulasan", type=["csv"])
kamus_file = st.sidebar.file_uploader("Upload kamus kata baku (xlsx)", type=["xlsx"])
test_size = st.sidebar.slider("Proporsi data uji", 0.05, 0.30, 0.10, 0.05)
max_features = st.sidebar.number_input("Max fitur TF-IDF", 1000, 10000, 5000, 500)
use_smote = st.sidebar.checkbox("Gunakan SMOTE", value=True)
selected_kernel = st.sidebar.selectbox("Kernel SVM", ["linear", "rbf", "sigmoid", "poly"])

# ── Helper Functions ──

def remove_URL(text):
    if isinstance(text, str):
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

def remove_html(text):
    if isinstance(text, str):
        return re.sub(r'<.*?>', '', text)
    return text

def remove_emoji(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F700-\U0001F77F"
            u"\U0001F780-\U0001F7FF"
            u"\U0001F800-\U0001F8FF"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)
    return text

def remove_symbols(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def remove_numbers(text):
    if isinstance(text, str):
        return re.sub(r'\d', '', text)
    return text

def case_folding(text):
    if isinstance(text, str):
        return text.lower()
    return text

def replace_taboo_words(text, kamus_tidak_baku):
    if isinstance(text, str):
        words = text.split()
        replaced = [kamus_tidak_baku.get(w, w) for w in words]
        return ' '.join(replaced)
    return ''

def tokenize(text):
    if isinstance(text, str):
        return text.split()
    return []

def get_sentiment_label(rating):
    try:
        r = int(rating)
    except:
        return None
    if r in [1, 2]:
        return 'negatif'
    if r in [4, 5]:
        return 'positif'
    return None


# ── Main Flow ──
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Tampilkan data mentah
    st.header("1. Data Mentah")
    st.dataframe(df.head(100), use_container_width=True)
    st.info(f"Jumlah data: **{len(df)}** baris")

    # Pastikan kolom yang dibutuhkan ada
    required_cols = ['Review Text', 'Rating']
    if not all(c in df.columns for c in required_cols):
        st.error(f"CSV harus memiliki kolom: {required_cols}")
        st.stop()

    # Pilih kolom yang relevan
    cols_to_keep = [c for c in ['Date', 'Username', 'Rating', 'Review Text'] if c in df.columns]
    df = df[cols_to_keep].copy()

    # Hapus duplikat
    before_dedup = len(df)
    df.drop_duplicates(subset='Review Text', keep='first', inplace=True)
    st.write(f"Duplikat dihapus: **{before_dedup - len(df)}** baris")

    # ── WordCloud sebelum preprocessing ──
    st.header("2. WordCloud Sebelum Preprocessing")
    all_text = ' '.join(df['Review Text'].dropna().astype(str))
    if all_text.strip():
        wc = WordCloud(width=800, height=400, background_color='black', random_state=42).generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title("WordCloud Sebelum Preprocessing")
        st.pyplot(fig_wc)

    # ── Preprocessing ──
    st.header("3. Preprocessing")

    with st.spinner("Cleaning..."):
        df['cleaning'] = df['Review Text'].apply(remove_URL)
        df['cleaning'] = df['cleaning'].apply(remove_html)
        df['cleaning'] = df['cleaning'].apply(remove_emoji)
        df['cleaning'] = df['cleaning'].apply(remove_symbols)
        df['cleaning'] = df['cleaning'].apply(remove_numbers)

    with st.spinner("Case folding..."):
        df['case_folding'] = df['cleaning'].apply(case_folding)

    with st.spinner("Normalisasi..."):
        if kamus_file is not None:
            kamus_data = pd.read_excel(kamus_file)
            kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
            df['normalisasi'] = df['case_folding'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku))
        else:
            st.warning("Kamus kata baku tidak diupload. Tahap normalisasi dilewati.")
            df['normalisasi'] = df['case_folding']

    with st.spinner("Tokenizing..."):
        df['tokenize'] = df['normalisasi'].apply(tokenize)

    with st.spinner("Stopword removal..."):
        stop_words = set(stopwords.words('indonesian'))
        df['stopword_removal'] = df['tokenize'].apply(lambda x: [w for w in x if w not in stop_words])

    with st.spinner("Stemming (memerlukan waktu)..."):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['stemming_data'] = df['stopword_removal'].apply(lambda x: ' '.join([stemmer.stem(w) for w in x]))

    st.success("Preprocessing selesai!")
    st.dataframe(df[['Review Text', 'cleaning', 'case_folding', 'normalisasi', 'stemming_data']].head(50), use_container_width=True)

    # ── WordCloud setelah preprocessing ──
    st.header("4. WordCloud Setelah Preprocessing")
    all_text_clean = ' '.join(df['stemming_data'].dropna().astype(str))
    if all_text_clean.strip():
        wc2 = WordCloud(width=800, height=400, background_color='black', random_state=42).generate(all_text_clean)
        fig_wc2, ax_wc2 = plt.subplots(figsize=(10, 5))
        ax_wc2.imshow(wc2, interpolation='bilinear')
        ax_wc2.axis('off')
        ax_wc2.set_title("WordCloud Setelah Preprocessing")
        st.pyplot(fig_wc2)

    # ── Labeling ──
    st.header("5. Labeling Sentimen")
    df['Sentiment'] = df['Rating'].apply(get_sentiment_label)
    df = df.dropna(subset=['Sentiment']).reset_index(drop=True)

    sentiment_count = df['Sentiment'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positif", sentiment_count.get('positif', 0))
    with col2:
        st.metric("Negatif", sentiment_count.get('negatif', 0))

    fig_label, ax_label = plt.subplots(figsize=(6, 4))
    sns.barplot(x=sentiment_count.index, y=sentiment_count.values, hue=sentiment_count.index, palette='pastel', legend=False, ax=ax_label)
    ax_label.set_title("Distribusi Label Sentimen")
    ax_label.set_xlabel("Kelas Sentimen")
    ax_label.set_ylabel("Jumlah")
    total = len(df)
    for i, count in enumerate(sentiment_count.values):
        pct = f'{100 * count / total:.2f}%'
        ax_label.text(i, count + 0.10, f'{count}\n{pct}', ha='center', va='bottom', fontsize=10)
    st.pyplot(fig_label)

    # ── Splitting ──
    st.header("6. Split Data")
    df = df.dropna(subset=['stemming_data'])
    X_train, X_test, y_train, y_test = train_test_split(
        df['stemming_data'], df['Sentiment'], test_size=test_size, random_state=42
    )
    st.write(f"Data latih: **{len(X_train)}** | Data uji: **{len(X_test)}**")

    fig_split, ax_split = plt.subplots(figsize=(5, 3))
    ax_split.bar(['Data Latih', 'Data Uji'], [len(X_train), len(X_test)], color=['#89B0CC', '#C5D5E4'])
    ax_split.set_ylabel("Jumlah")
    ax_split.set_title("Pembagian Data")
    for i, v in enumerate([len(X_train), len(X_test)]):
        ax_split.text(i, v + 5, str(v), ha='center', fontsize=10)
    st.pyplot(fig_split)

    # ── TF-IDF ──
    st.header("7. TF-IDF Vectorization")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    st.write(f"Dimensi TF-IDF data latih: **{X_train_vec.shape}**")
    st.write(f"Dimensi TF-IDF data uji: **{X_test_vec.shape}**")

    # ── SMOTE ──
    if use_smote:
        st.header("8. SMOTE")
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            fig_before, ax_before = plt.subplots(figsize=(5, 3))
            sns.countplot(x=y_train, ax=ax_before)
            ax_before.set_title("Distribusi Sebelum SMOTE")
            st.pyplot(fig_before)

        smote = SMOTE(random_state=42)
        X_train_final, y_train_final = smote.fit_resample(X_train_vec, y_train)

        with col_s2:
            fig_after, ax_after = plt.subplots(figsize=(5, 3))
            sns.countplot(x=y_train_final, ax=ax_after)
            ax_after.set_title("Distribusi Setelah SMOTE")
            st.pyplot(fig_after)

        st.write(f"Data latih setelah SMOTE: **{X_train_final.shape[0]}**")
    else:
        X_train_final, y_train_final = X_train_vec, y_train

    # ── SVM Classification ──
    st.header("9. Klasifikasi SVM")

    # Perbandingan semua kernel
    st.subheader("Perbandingan Semua Kernel")
    kernels = ['linear', 'rbf', 'sigmoid', 'poly']
    kernel_labels = ['Linear', 'RBF', 'Sigmoid', 'Polynomial']
    results = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

    with st.spinner("Melatih model SVM untuk semua kernel..."):
        for kernel in kernels:
            model = SVC(kernel=kernel, random_state=42)
            model.fit(X_train_final, y_train_final)
            y_pred = model.predict(X_test_vec)

            results['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
            results['Precision'].append(precision_score(y_test, y_pred, average='weighted') * 100)
            results['Recall'].append(recall_score(y_test, y_pred, average='weighted') * 100)
            results['F1-score'].append(f1_score(y_test, y_pred, average='weighted') * 100)

    # Tabel perbandingan
    results_df = pd.DataFrame(results, index=kernel_labels)
    st.dataframe(results_df.style.format("{:.2f}%"), use_container_width=True)

    # Grafik perbandingan
    metrics = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.18
    colors = ['#C5D5E4', '#89B0CC', '#4A7FA5', '#1B4F72']

    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 7))
    for i, (label, color) in enumerate(zip(kernel_labels, colors)):
        values = [results[m][i] for m in metrics]
        bars = ax_cmp.bar(x + i * width, values, width, label=label, color=color)
        for bar, val in zip(bars, values):
            ax_cmp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    ax_cmp.set_xlabel('Metric')
    ax_cmp.set_ylabel('Skor (%)')
    smote_label = "Setelah SMOTE" if use_smote else "Tanpa SMOTE"
    ax_cmp.set_title(f'Perbandingan Kernel SVM ({smote_label})')
    ax_cmp.set_xticks(x + width * 1.5)
    ax_cmp.set_xticklabels(metrics)
    ax_cmp.set_ylim(0, 105)
    ax_cmp.legend(title='Kernel SVM', loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_cmp)

    # ── Detail model terpilih ──
    st.subheader(f"Detail Hasil — Kernel: {selected_kernel.upper()}")
    with st.spinner(f"Melatih model SVM kernel {selected_kernel}..."):
        final_model = SVC(kernel=selected_kernel, random_state=42)
        final_model.fit(X_train_final, y_train_final)
        y_pred_final = final_model.predict(X_test_vec)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred_final)*100:.2f}%")
    col_m2.metric("Precision", f"{precision_score(y_test, y_pred_final, average='weighted')*100:.2f}%")
    col_m3.metric("Recall", f"{recall_score(y_test, y_pred_final, average='weighted')*100:.2f}%")
    col_m4.metric("F1-Score", f"{f1_score(y_test, y_pred_final, average='weighted')*100:.2f}%")

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred_final))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_final)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negatif', 'positif'],
                yticklabels=['negatif', 'positif'], ax=ax_cm)
    ax_cm.set_xlabel('Prediksi')
    ax_cm.set_ylabel('Aktual')
    ax_cm.set_title(f'Confusion Matrix — Kernel {selected_kernel.upper()}')
    st.pyplot(fig_cm)

    # ── WordCloud per Sentimen ──
    st.header("10. WordCloud per Sentimen")
    col_wc1, col_wc2 = st.columns(2)

    neg_text = ' '.join(df[df['Sentiment'] == 'negatif']['stemming_data'].dropna())
    pos_text = ' '.join(df[df['Sentiment'] == 'positif']['stemming_data'].dropna())

    with col_wc1:
        if neg_text.strip():
            wc_neg = WordCloud(width=800, height=400, background_color='black', random_state=42).generate(neg_text)
            fig_neg, ax_neg = plt.subplots(figsize=(8, 4))
            ax_neg.imshow(wc_neg, interpolation='bilinear')
            ax_neg.axis('off')
            ax_neg.set_title("WordCloud Sentimen Negatif")
            st.pyplot(fig_neg)

    with col_wc2:
        if pos_text.strip():
            wc_pos = WordCloud(width=800, height=400, background_color='black', random_state=42).generate(pos_text)
            fig_pos, ax_pos = plt.subplots(figsize=(8, 4))
            ax_pos.imshow(wc_pos, interpolation='bilinear')
            ax_pos.axis('off')
            ax_pos.set_title("WordCloud Sentimen Positif")
            st.pyplot(fig_pos)

    # ── Download hasil ──
    st.header("11. Download Hasil")
    csv_result = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Hasil Lengkap (CSV)", csv_result, "hasil_analisis_sentimen.csv", "text/csv")

else:
    st.info("Silakan upload file CSV ulasan melalui sidebar untuk memulai analisis.")
    st.markdown("""
    ### Format CSV yang diharapkan:
    | Date | Username | Rating | Review Text |
    |------|----------|--------|-------------|
    | 2024-01-01 | user1 | 5 | Aplikasi sangat bagus... |
    | 2024-01-02 | user2 | 1 | Tidak bisa digunakan... |

    **Kolom wajib:** `Review Text`, `Rating`
    """)
