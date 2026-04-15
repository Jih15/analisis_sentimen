import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Prediksi | SentimenTA", page_icon="🔮", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "df_dataset":   None,
    "preprocessor": None,
    "all_models":   None,
    "hasil_eval":   None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
section[data-testid="stSidebar"] { background-color: #0F1B2D; }
section[data-testid="stSidebar"] * { color: #94A3B8 !important; }
.page-title { font-size:1.8rem; font-weight:800; color:#0F1B2D; letter-spacing:-.02em; }
.pred-card {
    border-radius: 12px; padding: 14px 16px; margin: 6px 0;
    display: flex; align-items: center; gap: 12px;
    border: 1px solid #E2E8F0;
}
.pred-positif { background: #F0FDF4; border-color: #86EFAC; }
.pred-negatif { background: #FFF1F2; border-color: #FDA4AF; }
.pred-netral  { background: #FFFBEB; border-color: #FDE68A; }
.pred-label   { font-weight: 700; font-size: 1rem; }
.pred-conf    { font-size: 0.82rem; color: #64748B; margin-top: 2px; }
.section-title { font-size: 1.05rem; font-weight: 700; color: #1E293B;
    border-left: 4px solid #2563EB; padding-left: 10px; margin: 18px 0 10px 0; }
.info-box {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
    padding:14px 18px; margin-bottom:12px;
}
.info-box p { margin:0; color:#1E40AF; font-size:0.88rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="page-title">🔮 Prediksi Sentimen Baru</p>', unsafe_allow_html=True)
st.caption("Masukkan respons kuesioner baru untuk mendapatkan prediksi sentimen dari setiap aspek.")
st.divider()

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
TEXT_COLUMNS = [
    'prep_fitur_deteksi', 'prep_penjelasan_deteksi', 'prep_fitur_cv',
    'prep_konten_edukasi', 'prep_kelebihan', 'prep_kekurangan', 'prep_saran',
]
LIKERT_COLUMNS = [
    'Berdasarkan Pengalam anda menggunakan aplikasi Dan melihat Gambar di atas, Seberapa mudah proses memasukkan data lowongan kerja untuk dideteksi? ',
    'Seberapa akurat Anda merasakan hasil deteksi yang diberikan oleh aplikasi? ',
    '  Seberapa bermanfaat fitur "Pembuat CV" bagi Anda?   ',
    'Seberapa informatif artikel dan tips yang ada di fitur "Konten Edukasi"? ',
    '  Secara keseluruhan, seberapa puas Anda dengan aplikasi ini?  ',
]
ALL_LABELS = [
    'label_kemudahan_deteksi', 'label_akurasi_deteksi', 'label_fitur_deteksi',
    'label_penjelasan_deteksi', 'label_manfaat_cv', 'label_fitur_cv',
    'label_informatif_edukasi', 'label_konten_edukasi', 'label_kepuasan_keseluruhan',
    'label_kelebihan_aplikasi', 'label_kekurangan_aplikasi', 'label_saran_kritik',
]
LABEL_DISPLAY = {
    "label_kemudahan_deteksi":   "Kemudahan Deteksi",
    "label_akurasi_deteksi":     "Akurasi Deteksi",
    "label_fitur_deteksi":       "Fitur Deteksi",
    "label_penjelasan_deteksi":  "Penjelasan Deteksi",
    "label_manfaat_cv":          "Manfaat Pembuat CV",
    "label_fitur_cv":            "Fitur Pembuat CV",
    "label_informatif_edukasi":  "Informatif Konten Edukasi",
    "label_konten_edukasi":      "Konten Edukasi",
    "label_kepuasan_keseluruhan":"Kepuasan Keseluruhan",
    "label_kelebihan_aplikasi":  "Kelebihan Aplikasi",
    "label_kekurangan_aplikasi": "Kekurangan Aplikasi",
    "label_saran_kritik":        "Saran & Kritik",
}

TEXT_LABELS = {
    'prep_fitur_deteksi':       "Pengalaman fitur deteksi lowongan",
    'prep_penjelasan_deteksi':  "Pendapat tentang penjelasan hasil deteksi",
    'prep_fitur_cv':            "Pendapat tentang fitur Pembuat CV",
    'prep_konten_edukasi':      "Komentar tentang Konten Edukasi",
    'prep_kelebihan':           "Kelebihan aplikasi yang disukai",
    'prep_kekurangan':          "Kekurangan / hal yang tidak disukai",
    'prep_saran':               "Saran atau kritik untuk pengembangan",
}

LIKERT_SHORT = [
    "Kemudahan input deteksi (1–4)",
    "Akurasi hasil deteksi (1–4)",
    "Manfaat Pembuat CV (1–4)",
    "Informatif Konten Edukasi (1–4)",
    "Kepuasan keseluruhan (1–4)",
]

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
prep   = st.session_state.preprocessor
models = st.session_state.all_models

if prep is None or models is None:
    st.warning("⚠️ Model belum dimuat. Silakan ke halaman **Model** terlebih dahulu.")
    st.stop()

# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------
import re

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "dalam", "tidak", "juga", "sudah", "saya", "kamu",
    "kami", "mereka", "akan", "bisa", "ada", "lebih", "seperti", "dapat",
    "oleh", "karena", "sehingga", "namun", "tetapi", "atau", "jika", "maka",
    "sangat", "telah", "belum", "masih", "hanya", "saja", "pun", "bukan",
    "agar", "supaya", "ketika", "sebelum", "sesudah", "setelah", "antara",
    "atas", "bawah", "lain", "semua", "setiap", "beberapa", "satu", "dua",
    "tiga", "ia", "nya", "mu", "ku", "jadi", "hal", "cara", "kita", "ya",
    "ga", "gak", "tak", "nggak", "si", "pak", "bu", "bang", "kak", "mas",
    "mbak", "lah", "pun", "kok", "deh", "sih", "dong", "nih", "tuh",
    "udah", "udh", "sdh", "banget", "bgt", "jg", "yg", "dgn", "utk",
    "krn", "tp", "tapi", "dr", "pd", "sy", "km", "mrk", "klo", "kalo",
    "kalau", "emang", "memang", "gimana", "kenapa", "dimana", "kapan",
    "iya", "mau", "msh", "blm", "jgn", "jangan", "saat", "waktu",
    "sekarang", "nanti", "tdk", "g", "u", "d", "y", "n",
}

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    text = " ".join(w for w in words if w not in STOPWORDS_ID and len(w) > 1)
    return text

def transform_single(row_dict, prep):
    tfidf_vectorizers = prep["tfidf_vectorizers"]
    scaler            = prep["scaler"]
    text_cols         = prep["text_columns"]
    likert_cols       = prep["likert_columns"]

    tfidf_parts = []
    for col in text_cols:
        val = row_dict.get(col, "")
        tfidf_parts.append(tfidf_vectorizers[col].transform([val]))

    likert_data = np.zeros((1, len(likert_cols)))
    for i, col in enumerate(likert_cols):
        likert_data[0, i] = float(row_dict.get(col, 0))

    likert_scaled = csr_matrix(scaler.transform(likert_data))
    return hstack(tfidf_parts + [likert_scaled])

# ---------------------------------------------------------------------------
# Form input
# ---------------------------------------------------------------------------
st.markdown("""<div class="info-box">
    <p>Isi kolom teks dengan jawaban dari responden (akan dipreproses otomatis).
    Isi nilai Likert sesuai skala 1–4. Kosongkan jika tidak ada jawaban.</p>
</div>""", unsafe_allow_html=True)

with st.form("form_prediksi"):
    st.markdown('<p class="section-title">📝 Jawaban Teks Bebas</p>', unsafe_allow_html=True)
    text_inputs = {}
    col1, col2 = st.columns(2)
    text_col_list = list(TEXT_LABELS.items())
    for i, (col_key, col_label) in enumerate(text_col_list):
        target = col1 if i % 2 == 0 else col2
        with target:
            text_inputs[col_key] = st.text_area(
                col_label,
                placeholder=f"Isi jawaban untuk {col_label.lower()}...",
                height=80,
                key=f"inp_{col_key}",
            )

    st.markdown('<p class="section-title">📊 Nilai Likert (Skala 1–4)</p>', unsafe_allow_html=True)
    likert_inputs = {}
    lk_cols = st.columns(5)
    for i, (col_key, short_label) in enumerate(zip(LIKERT_COLUMNS, LIKERT_SHORT)):
        with lk_cols[i]:
            likert_inputs[col_key] = st.selectbox(
                short_label, options=[1, 2, 3, 4], index=1, key=f"lk_{i}"
            )

    st.markdown('<p class="section-title">⚙️ Pilih Model</p>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Gunakan model:", options=["SVM", "Regresi Logistik", "Keduanya"],
        horizontal=True, index=2,
    )

    submitted = st.form_submit_button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Jalankan prediksi
# ---------------------------------------------------------------------------
if submitted:
    # Preproses teks
    row_dict = {}
    for col_key, raw_text in text_inputs.items():
        row_dict[col_key] = preprocess_text(raw_text) if raw_text.strip() else ""

    for col_key, val in likert_inputs.items():
        row_dict[col_key] = float(val)

    X = transform_single(row_dict, prep)

    st.divider()
    st.subheader("🔮 Hasil Prediksi")

    label_icons = {"positif": "🟢", "negatif": "🔴", "netral": "🟡"}
    pred_rows = []

    for lbl in ALL_LABELS:
        model_info = models.get(lbl)
        if model_info is None:
            continue

        le       = model_info["label_encoder"]
        svm_mdl  = model_info["svm"]
        lr_mdl   = model_info["lr"]
        display  = LABEL_DISPLAY.get(lbl, lbl)

        results = {}
        if model_choice in ["SVM", "Keduanya"]:
            pred_enc = svm_mdl.predict(X)[0]
            pred_lbl = le.inverse_transform([pred_enc])[0]
            proba = svm_mdl.predict_proba(X)[0]
            conf  = proba.max()
            results["SVM"] = (pred_lbl, conf)

        if model_choice in ["Regresi Logistik", "Keduanya"]:
            pred_enc = lr_mdl.predict(X)[0]
            pred_lbl = le.inverse_transform([pred_enc])[0]
            proba = lr_mdl.predict_proba(X)[0]
            conf  = proba.max()
            results["LR"] = (pred_lbl, conf)

        pred_rows.append({"label_key": lbl, "display": display, "results": results})

    # Tampilkan dalam grid
    n_cols = 3
    for row_start in range(0, len(pred_rows), n_cols):
        cols = st.columns(n_cols)
        for c_idx, pred in enumerate(pred_rows[row_start:row_start + n_cols]):
            with cols[c_idx]:
                st.markdown(f"**{pred['display']}**")
                for model_nm, (lbl_val, conf) in pred["results"].items():
                    lbl_lower = str(lbl_val).lower()
                    icon = label_icons.get(lbl_lower, "⚪")
                    css_cls = f"pred-{lbl_lower}" if lbl_lower in ["positif", "negatif", "netral"] else ""
                    st.markdown(
                        f'<div class="pred-card {css_cls}">'
                        f'  <div>'
                        f'    <div class="pred-label">{icon} {model_nm}: {str(lbl_val).capitalize()}</div>'
                        f'    <div class="pred-conf">Confidence: {conf:.1%}</div>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Tabel ringkasan
    st.divider()
    st.markdown("**📋 Ringkasan Prediksi**")
    summary_data = []
    for pred in pred_rows:
        row = {"Aspek": pred["display"]}
        for nm, (lbl_val, conf) in pred["results"].items():
            row[f"{nm} Prediksi"]  = str(lbl_val).capitalize()
            row[f"{nm} Confidence"] = f"{conf:.1%}"
        summary_data.append(row)

    st.dataframe(pd.DataFrame(summary_data).set_index("Aspek"), use_container_width=True)

    # Export
    csv_pred = pd.DataFrame(summary_data).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Hasil Prediksi (CSV)",
        data=csv_pred,
        file_name="hasil_prediksi_baru.csv",
        mime="text/csv",
    )