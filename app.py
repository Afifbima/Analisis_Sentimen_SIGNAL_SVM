import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
import io
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from google_play_scraper import reviews, Sort

nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Analisis Sentimen SIGNAL - SVM", layout="wide")
st.title("Analisis Sentimen Ulasan Aplikasi SIGNAL")
st.markdown("Menggunakan metode **Support Vector Machine (SVM)** dengan **TF-IDF**")

# ── Sidebar ──
st.sidebar.header("Pengaturan")
data_source = st.sidebar.radio("Sumber Data", ["Scraping Google Play", "Upload CSV"])
jumlah_review = st.sidebar.number_input("Jumlah review scraping", 100, 10000, 10000, 100)
run_button = st.sidebar.button("▶ Jalankan Analisis", type="primary", use_container_width=True)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV ulasan", type=["csv"])

# Path kamus kata baku lokal
import os
KAMUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kamuskatabaku.xlsx")

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
        replaced = []
        for word in words:
            if word in kamus_tidak_baku:
                baku_word = kamus_tidak_baku[word]
                if isinstance(baku_word, str) and all(char.isalpha() for char in baku_word):
                    replaced.append(baku_word)
                else:
                    replaced.append(word)
            else:
                replaced.append(word)
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
if run_button:
    # ════════════════════════════════════════════════
    # 1. SCRAPING / LOAD DATA
    # ════════════════════════════════════════════════
    st.header("1. Scrapping Data")

    if data_source == "Scraping Google Play":
        with st.spinner(f"Mengambil {jumlah_review} ulasan dari Google Play..."):
            app_id = 'app.signal.id'
            try:
                result, continuation_token = reviews(
                    app_id,
                    lang='id',
                    country='id',
                    sort=Sort.NEWEST,
                    count=jumlah_review,
                )
                reviews_data = result
            except Exception as e:
                st.error(f"Error saat scraping: {e}")
                st.stop()

        if reviews_data is None or len(reviews_data) == 0:
            st.error("Tidak dapat mengambil ulasan.")
            st.stop()

        st.success(f"Berhasil mengambil **{len(reviews_data)}** ulasan")
        st.write("Contoh ulasan:")
        st.json({
            'reviewId': reviews_data[0].get('reviewId', ''),
            'userName': reviews_data[0].get('userName', ''),
            'score': reviews_data[0].get('score', ''),
            'content': reviews_data[0].get('content', ''),
            'at': str(reviews_data[0].get('at', '')),
        })

        # Konversi ke DataFrame
        rows = []
        for review in reviews_data:
            rows.append({
                'Review ID': review.get('reviewId', ''),
                'Username': review.get('userName', ''),
                'Rating': review.get('score', ''),
                'Review Text': review.get('content', ''),
                'Date': str(review.get('at', '')),
            })
        df = pd.DataFrame(rows)

        # Simpan CSV
        df.to_csv('hasil_scraper_ulasan_app_signal.csv', index=False, encoding='utf-8')
        st.info("Data berhasil diekspor ke `hasil_scraper_ulasan_app_signal.csv`")

    else:
        if uploaded_file is None:
            st.warning("Silakan upload file CSV terlebih dahulu.")
            st.stop()
        df = pd.read_csv(uploaded_file)

    # Tampilkan info data
    st.write(f"**Jumlah data:** {len(df)} baris")
    buf = io.StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())
    st.dataframe(df.head(1000), use_container_width=True)

    # Pastikan kolom yang dibutuhkan ada
    required_cols = ['Review Text', 'Rating']
    if not all(c in df.columns for c in required_cols):
        st.error(f"CSV harus memiliki kolom: {required_cols}")
        st.stop()

    # Pilih kolom yang relevan
    df = pd.DataFrame(df[['Date', 'Username', 'Rating', 'Review Text']])

    # Hapus duplikat
    before_dedup = len(df)
    df.drop_duplicates(subset='Review Text', keep='first', inplace=True)
    st.write(f"Duplikat dihapus: **{before_dedup - len(df)}** baris → sisa **{len(df)}** baris")

    st.dataframe(df.head(1000), use_container_width=True)

    # ════════════════════════════════════════════════
    # 2. WORDCLOUD SEBELUM PREPROCESSING
    # ════════════════════════════════════════════════
    st.header("2. WordCloud Sebelum Preprocessing")

    df['Review Text'] = df['Review Text'].fillna('')
    all_text = ' '.join(df['Review Text'].astype(str).tolist())

    if all_text.strip():
        sw = set(STOPWORDS)
        sw.update(['https', 'co', 'RT', '......', 'amp'])

        wc = WordCloud(stopwords=sw, background_color="black", max_words=500, width=800, height=400)
        wc.generate(all_text)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
        plt.close()

    # Frekuensi kata (top 10)
    st.subheader("Frekuensi Kata (Top 10)")
    tokens_raw = all_text.split()
    word_counts_raw = Counter(tokens_raw)
    top_words_raw = word_counts_raw.most_common(10)

    if top_words_raw:
        word_r, count_r = zip(*top_words_raw)
        colors_bar = plt.cm.tab10(range(len(word_r)))

        fig_freq, ax_freq = plt.subplots(figsize=(12, 6))
        bars = ax_freq.bar(word_r, count_r, color=colors_bar)
        ax_freq.set_xlabel("kata sering muncul", fontsize=12, fontweight='bold')
        ax_freq.set_ylabel("jumlah kata", fontsize=12, fontweight='bold')
        ax_freq.set_title("frekuensi kata", fontsize=20, fontweight='bold')
        plt.xticks(rotation=45)
        for bar, num in zip(bars, count_r):
            ax_freq.text(bar.get_x() + bar.get_width() / 2, num + 1, str(num),
                         fontsize=12, color='black', ha='center')
        st.pyplot(fig_freq)
        plt.close()

    # ════════════════════════════════════════════════
    # 3. PREPROCESSING
    # ════════════════════════════════════════════════
    st.header("3. Preprocessing")

    # Cleaning
    with st.spinner("Cleaning..."):
        df['cleaning'] = df['Review Text'].apply(remove_URL)
        df['cleaning'] = df['cleaning'].apply(remove_html)
        df['cleaning'] = df['cleaning'].apply(remove_emoji)
        df['cleaning'] = df['cleaning'].apply(remove_symbols)
        df['cleaning'] = df['cleaning'].apply(remove_numbers)
    st.success("Cleaning selesai!")

    # Case Folding
    with st.spinner("Case folding..."):
        df['case_folding'] = df['cleaning'].apply(case_folding)
    st.success("Case folding selesai!")

    # Normalisasi
    with st.spinner("Normalisasi..."):
        kamus_data = pd.read_excel(KAMUS_PATH)
        kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
        df['normalisasi'] = df['case_folding'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku))
    st.success("Normalisasi selesai!")

    # Tokenizing
    with st.spinner("Tokenizing..."):
        df['tokenize'] = df['normalisasi'].apply(tokenize)
    st.success("Tokenizing selesai!")

    # Stopword Removal
    with st.spinner("Stopword removal..."):
        stop_words = set(stopwords.words('indonesian'))
        df['stopword_removal'] = df['tokenize'].apply(lambda x: [w for w in x if w not in stop_words])
    st.success("Stopword removal selesai!")

    # Stemming
    with st.spinner("Stemming (memerlukan waktu)..."):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['stemming_data'] = df['stopword_removal'].apply(lambda x: ' '.join([stemmer.stem(w) for w in x]))
    st.success("Stemming selesai!")

    st.success("**Preprocessing selesai!**")
    st.dataframe(df[['Review Text', 'cleaning', 'case_folding', 'normalisasi', 'stemming_data']].head(5000),
                 use_container_width=True)

    # Simpan hasil preprocessing
    df.to_csv('Hasil_Preprocessing_Data.csv', encoding='utf8', index=False)
    st.info("Hasil preprocessing disimpan ke `Hasil_Preprocessing_Data.csv`")

    # ════════════════════════════════════════════════
    # 4. WORDCLOUD SETELAH PREPROCESSING
    # ════════════════════════════════════════════════
    st.header("4. WordCloud Setelah Preprocessing")

    all_text_clean = ' '.join(df['stemming_data'].astype(str).tolist())

    if all_text_clean.strip():
        sw2 = set(STOPWORDS)
        sw2.update(['https', 'co', 'RT', '.......', 'amp', 'ya'])

        wc2 = WordCloud(stopwords=sw2, background_color="black", max_words=500, width=800, height=400)
        wc2.generate(all_text_clean)

        fig_wc2, ax_wc2 = plt.subplots(figsize=(10, 5))
        ax_wc2.imshow(wc2, interpolation='bilinear')
        ax_wc2.axis('off')
        st.pyplot(fig_wc2)
        plt.close()

    # Frekuensi kata setelah preprocessing (top 10)
    st.subheader("Frekuensi Kata Setelah Preprocessing (Top 10)")
    tokens_clean = all_text_clean.split()
    word_counts_clean = Counter(tokens_clean)
    top_words_clean = word_counts_clean.most_common(10)

    if top_words_clean:
        word_c, count_c = zip(*top_words_clean)
        colors_bar2 = plt.cm.tab10(range(len(word_c)))

        fig_freq2, ax_freq2 = plt.subplots(figsize=(12, 6))
        bars2 = ax_freq2.bar(word_c, count_c, color=colors_bar2)
        ax_freq2.set_xlabel("kata sering muncul", fontsize=12, fontweight='bold')
        ax_freq2.set_ylabel("jumlah kata", fontsize=12, fontweight='bold')
        ax_freq2.set_title("frekuensi kata", fontsize=20, fontweight='bold')
        plt.xticks(rotation=45)
        for bar, num in zip(bars2, count_c):
            ax_freq2.text(bar.get_x() + bar.get_width() / 2, num + 1, str(num),
                          fontsize=12, color='black', ha='center')
        st.pyplot(fig_freq2)
        plt.close()

    # ════════════════════════════════════════════════
    # 5. LABELING
    # ════════════════════════════════════════════════
    st.header("5. Labeling Sentimen")
    st.markdown("Rating 1-2 → **Negatif** | Rating 4-5 → **Positif** | Rating 3 → Dihapus (Netral)")

    df['Sentiment'] = df['Rating'].apply(get_sentiment_label)
    df = df.dropna(subset=['Sentiment']).reset_index(drop=True)

    # Simpan hasil labeling
    df.to_csv('Hasil_Labeling_Data.csv', index=False)
    st.info(f"Pelabelan selesai. Hasil disimpan di `Hasil_Labeling_Data.csv` ({len(df)} data)")

    sentiment_count = df['Sentiment'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positif", sentiment_count.get('positif', 0))
    with col2:
        st.metric("Negatif", sentiment_count.get('negatif', 0))

    # Grafik distribusi sentimen
    fig_label, ax_label = plt.subplots(figsize=(6, 4))
    sns.set_style('whitegrid')
    sns.barplot(x=sentiment_count.index, y=sentiment_count.values, hue=sentiment_count.index,
                palette='pastel', legend=False, ax=ax_label)
    ax_label.set_title("Jumlah Label Positif dan Negatif", fontsize=14, pad=20)
    ax_label.set_xlabel("Kelas Sentimen", fontsize=12)
    ax_label.set_ylabel("Jumlah Tweet", fontsize=12)
    total = len(df)
    for i, count in enumerate(sentiment_count.values):
        pct = f'{100 * count / total:.2f}%'
        ax_label.text(i, count + 0.10, f'{count}\n{pct}', ha='center', va='bottom', fontsize=10)
    st.pyplot(fig_label)
    plt.close()

    # ════════════════════════════════════════════════
    # 6. SPLITTING DATA
    # ════════════════════════════════════════════════
    st.header("6. Spliting Data")

    df = df.dropna()
    st.write(f"Total data setelah drop NaN: **{len(df)}**")

    buf2 = io.StringIO()
    df.info(buf=buf2)
    st.text(buf2.getvalue())

    X_train, X_test, y_train, y_test = train_test_split(
        df['stemming_data'], df['Sentiment'], test_size=0.1, random_state=42
    )

    # Simpan train dan test
    train_set = pd.DataFrame({'text': X_train, 'sentiment': y_train})
    train_set.to_csv('train_data.csv', index=False)
    test_set = pd.DataFrame({'text': X_test, 'sentiment': y_test})
    test_set.to_csv('test_data.csv', index=False)

    st.write(f"Jumlah Data Latih: **{len(X_train)}**")
    st.write(f"Jumlah Data Uji: **{len(X_test)}**")

    # Grafik split data
    train_size = len(X_train)
    test_size = len(X_test)

    fig_split, ax_split = plt.subplots(figsize=(8, 6))
    bars_split = ax_split.bar(['Data Latih', 'Data Uji'], [train_size, test_size], color=['blue', 'orange'])
    for bar in bars_split:
        height = bar.get_height()
        ax_split.text(bar.get_x() + bar.get_width() / 2, height + 0.7,
                      f'{height} ({(height / (train_size + test_size) * 100):.2f}%)',
                      ha='center', va='bottom')
    ax_split.set_title('Jumlah Data Latih dan Data Uji')
    ax_split.set_xlabel('Jenis Data')
    ax_split.set_ylabel('Jumlah Data')
    st.pyplot(fig_split)
    plt.close()

    # ════════════════════════════════════════════════
    # 7. TF-IDF
    # ════════════════════════════════════════════════
    st.header("7. TF-IDF Vectorization")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    st.write(f"Dimensi TF-IDF data latih: **{X_train_vec.shape}**")
    st.write(f"Dimensi TF-IDF data uji: **{X_test_vec.shape}**")

    # Sebagian kecil matriks
    st.subheader("Sebagian kecil Matriks Vektorisasi untuk Data Latih")
    st.text(str(X_train_vec[:5, :].toarray()))

    # Top 5 TF-IDF terms
    st.subheader("Top 5 Terms Berdasarkan TF-IDF")
    tfidf_train_df = pd.DataFrame(X_train_vec.toarray(), columns=vectorizer.get_feature_names_out())
    term_ranks = tfidf_train_df.mean(axis=0).reset_index()
    term_ranks.columns = ["term", "rank"]
    top_terms = term_ranks.sort_values(by="rank", ascending=False).head(5).reset_index(drop=True)
    top_terms["rank"] = top_terms["rank"].round(6)
    st.dataframe(top_terms, use_container_width=True)

    # ════════════════════════════════════════════════
    # 8. SVM - PERBANDINGAN KERNEL
    # ════════════════════════════════════════════════
    st.header("8. SVM - Perbandingan Kernel")

    kernels = ['linear', 'rbf', 'poly']
    kernel_labels = ['Linear', 'RBF', 'Polynomial']
    results = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

    with st.spinner("Melatih model SVM untuk semua kernel..."):
        for kernel in kernels:
            model = SVC(kernel=kernel, random_state=42)
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)

            results['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
            results['Precision'].append(precision_score(y_test, y_pred, average='weighted') * 100)
            results['Recall'].append(recall_score(y_test, y_pred, average='weighted') * 100)
            results['F1-score'].append(f1_score(y_test, y_pred, average='weighted') * 100)

            st.write(f"**Kernel: {kernel}**")
            st.write(f"  Accuracy : {results['Accuracy'][-1]:.2f}%")
            st.write(f"  Precision: {results['Precision'][-1]:.2f}%")
            st.write(f"  Recall   : {results['Recall'][-1]:.2f}%")
            st.write(f"  F1-score : {results['F1-score'][-1]:.2f}%")

    # Grafik perbandingan kernel
    metrics = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.18
    colors_kernel = ['#C5D5E4', '#89B0CC', '#1B4F72']

    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 7))
    for i, (label, color) in enumerate(zip(kernel_labels, colors_kernel)):
        values = [results[m][i] for m in metrics]
        bars_k = ax_cmp.bar(x + i * width, values, width, label=label, color=color)
        for bar, val in zip(bars_k, values):
            ax_cmp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    ax_cmp.set_xlabel('Metric')
    ax_cmp.set_ylabel('Skor (%)')
    ax_cmp.set_title('Perbandingan Kernel SVM Berdasarkan Metrik Evaluasi')
    ax_cmp.set_xticks(x + width * 1.0)
    ax_cmp.set_xticklabels(metrics)
    ax_cmp.set_ylim(0, 105)
    ax_cmp.legend(title='Kernel SVM', loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_cmp)
    plt.close()

    # ════════════════════════════════════════════════
    # 9. CONFUSION MATRIX - LINEAR & RBF
    # ════════════════════════════════════════════════
    st.header("9. Confusion Matrix — Linear & RBF")

    with st.spinner("Melatih SVM Linear dan RBF..."):
        svm_linear = SVC(kernel="linear", random_state=42)
        svm_linear.fit(X_train_vec, y_train)
        y_pred_linear = svm_linear.predict(X_test_vec)

        svm_rbf = SVC(kernel="rbf", random_state=42)
        svm_rbf.fit(X_train_vec, y_train)
        y_pred_rbf = svm_rbf.predict(X_test_vec)

    cm_linear = confusion_matrix(y_test, y_pred_linear)
    cm_rbf = confusion_matrix(y_test, y_pred_rbf)

    fig_cm, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
    axes[0].set_title("SVM Kernel Linear", pad=15)
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
    axes[1].set_title("SVM Kernel RBF", pad=15)
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")

    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close()

    # Classification Reports
    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.subheader("Classification Report — Linear")
        st.text(classification_report(y_test, y_pred_linear))
    with col_rep2:
        st.subheader("Classification Report — RBF")
        st.text(classification_report(y_test, y_pred_rbf))

    # ════════════════════════════════════════════════
    # 10. PERBANDINGAN LINEAR vs RBF
    # ════════════════════════════════════════════════
    st.header("10. Perbandingan Kernel SVM Terbaik (Linear vs RBF)")

    acc_linear = accuracy_score(y_test, y_pred_linear) * 100
    prec_linear = precision_score(y_test, y_pred_linear, average='weighted') * 100
    rec_linear = recall_score(y_test, y_pred_linear, average='weighted') * 100
    f1_linear = f1_score(y_test, y_pred_linear, average='weighted') * 100

    acc_rbf = accuracy_score(y_test, y_pred_rbf) * 100
    prec_rbf = precision_score(y_test, y_pred_rbf, average='weighted') * 100
    rec_rbf = recall_score(y_test, y_pred_rbf, average='weighted') * 100
    f1_rbf = f1_score(y_test, y_pred_rbf, average='weighted') * 100

    linear_scores = [acc_linear, prec_linear, rec_linear, f1_linear]
    rbf_scores = [acc_rbf, prec_rbf, rec_rbf, f1_rbf]

    # Tabel perbandingan
    compare_df = pd.DataFrame({
        'Metric': metrics,
        'Linear': [f'{v:.2f}%' for v in linear_scores],
        'RBF': [f'{v:.2f}%' for v in rbf_scores],
    })
    st.dataframe(compare_df, use_container_width=True)

    x2 = np.arange(len(metrics))
    width2 = 0.25
    colors_lr = ['#C5D5E4', '#1B4F72']

    fig_lr, ax_lr = plt.subplots(figsize=(12, 7))

    bars1 = ax_lr.bar(x2 - width2 / 2, linear_scores, width2, label='Linear', color=colors_lr[0])
    bars2 = ax_lr.bar(x2 + width2 / 2, rbf_scores, width2, label='RBF', color=colors_lr[1])

    for bar, val in zip(bars1, linear_scores):
        ax_lr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, rbf_scores):
        ax_lr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    ax_lr.set_xlabel('Metric')
    ax_lr.set_ylabel('Skor (%)')
    ax_lr.set_title('Perbandingan Kernel SVM Terbaik (Linear vs RBF)')
    ax_lr.set_xticks(x2)
    ax_lr.set_xticklabels(metrics)
    ax_lr.set_ylim(0, 105)
    ax_lr.legend(title='Kernel SVM', loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_lr)
    plt.close()

    # ════════════════════════════════════════════════
    # 11. WORDCLOUD PER SENTIMEN
    # ════════════════════════════════════════════════
    st.header("11. WordCloud per Sentimen")

    neg_text = ' '.join(df[df['Sentiment'] == 'negatif']['stemming_data'].dropna().astype(str))
    pos_text = ' '.join(df[df['Sentiment'] == 'positif']['stemming_data'].dropna().astype(str))

    col_wc1, col_wc2 = st.columns(2)

    with col_wc1:
        if neg_text.strip():
            wc_neg = WordCloud(width=800, height=400, random_state=42, max_font_size=100,
                               background_color='black').generate(neg_text)
            fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
            ax_neg.imshow(wc_neg, interpolation='bilinear')
            ax_neg.axis('off')
            ax_neg.set_title("WordCloud Sentimen Negatif")
            st.pyplot(fig_neg)
            plt.close()

    with col_wc2:
        if pos_text.strip():
            wc_pos = WordCloud(width=800, height=400, random_state=42, max_font_size=100,
                               background_color='black').generate(pos_text)
            fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
            ax_pos.imshow(wc_pos, interpolation='bilinear')
            ax_pos.axis('off')
            ax_pos.set_title("WordCloud Sentimen Positif")
            st.pyplot(fig_pos)
            plt.close()

    # ════════════════════════════════════════════════
    # 12. DOWNLOAD HASIL
    # ════════════════════════════════════════════════
    st.header("12. Download Hasil")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        csv_preprocess = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil Preprocessing", csv_preprocess,
                           "Hasil_Preprocessing_Data.csv", "text/csv")

    with col_dl2:
        csv_train = train_set.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data Latih", csv_train,
                           "train_data.csv", "text/csv")

    with col_dl3:
        csv_test = test_set.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data Uji", csv_test,
                           "test_data.csv", "text/csv")

    st.success("Analisis selesai!")

else:
    st.info("Klik tombol **▶ Jalankan Analisis** di sidebar untuk memulai.")
    st.markdown("""
    ### Cara Penggunaan:
    1. Pilih sumber data: **Scraping Google Play** (otomatis) atau **Upload CSV**
    2. (Opsional) Upload **kamus kata baku** (.xlsx) untuk normalisasi
    3. Klik **▶ Jalankan Analisis**

    ### Format CSV (jika upload manual):
    | Date | Username | Rating | Review Text |
    |------|----------|--------|-------------|
    | 2024-01-01 | user1 | 5 | Aplikasi sangat bagus... |
    | 2024-01-02 | user2 | 1 | Tidak bisa digunakan... |

    **Kolom wajib:** `Review Text`, `Rating`
    """)
