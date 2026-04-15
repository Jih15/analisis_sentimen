import streamlit as st
import pandas as pd
import pickle
import io

st.set_page_config(page_title="Model | SentimenTA", page_icon="🤖", layout="wide")

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
.model-card {
    background: #F0FDF4; border: 1px solid #86EFAC;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 10px;
}
.model-card h4 { margin: 0 0 6px 0; color: #166534; }
.model-card p  { margin: 0; color: #15803D; font-size: 0.88rem; line-height: 1.7; }
.warn-card {
    background: #FFF7ED; border: 1px solid #FED7AA;
    border-radius: 12px; padding: 18px 20px;
}
.warn-card h4 { margin: 0 0 6px 0; color: #C2410C; }
.warn-card p  { margin: 0; color: #9A3412; font-size: 0.88rem; line-height: 1.6; }
.info-box {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
    padding:16px 18px; margin-bottom:12px;
}
.info-box h4 { margin:0 0 6px 0; color:#1D4ED8; }
.info-box p  { margin:0; color:#1E40AF; font-size:0.88rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="page-title">🤖 Upload Model</p>', unsafe_allow_html=True)
st.caption("Upload dua file hasil training: preprocessor.pkl dan models_all_labels.pkl")
st.divider()

LABEL_DISPLAY = {
    "label_kemudahan_deteksi":   "Kemudahan Deteksi",
    "label_akurasi_deteksi":     "Akurasi Deteksi",
    "label_fitur_deteksi":       "Fitur Deteksi",
    "label_penjelasan_deteksi":  "Penjelasan Deteksi",
    "label_manfaat_cv":          "Manfaat CV",
    "label_fitur_cv":            "Fitur CV",
    "label_informatif_edukasi":  "Informatif Edukasi",
    "label_konten_edukasi":      "Konten Edukasi",
    "label_kepuasan_keseluruhan":"Kepuasan Keseluruhan",
    "label_kelebihan_aplikasi":  "Kelebihan Aplikasi",
    "label_kekurangan_aplikasi": "Kekurangan Aplikasi",
    "label_saran_kritik":        "Saran & Kritik",
}

st.markdown("""<div class="info-box">
    <h4>Cara Mendapatkan File Model</h4>
    <p>Jalankan script <code>train_ml.py</code> pada dataset Anda. Script akan menghasilkan:<br>
    &bull; <code>models/preprocessor.pkl</code> — TF-IDF vectorizer (7 kolom) + StandardScaler Likert<br>
    &bull; <code>models/models_all_labels.pkl</code> — semua model SVM &amp; LR untuk 12 label</p>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload kedua file
# ---------------------------------------------------------------------------
col_prep, col_model = st.columns(2)

with col_prep:
    st.markdown("### 1️⃣ preprocessor.pkl")
    f_prep = st.file_uploader("Upload preprocessor.pkl", type=["pkl"], key="up_prep")
    if f_prep:
        try:
            bundle = pickle.load(io.BytesIO(f_prep.read()))
            required = {"tfidf_vectorizers", "scaler", "text_columns", "likert_columns"}
            if not required.issubset(bundle.keys()):
                st.error(f"❌ Key tidak lengkap. Ditemukan: {set(bundle.keys())}")
            else:
                st.session_state.preprocessor = bundle
                n_tfidf = len(bundle["tfidf_vectorizers"])
                n_likert = len(bundle["likert_columns"])
                st.markdown(f"""<div class="model-card">
                    <h4>✅ Preprocessor Aktif</h4>
                    <p>📝 TF-IDF vectorizer: <b>{n_tfidf} kolom</b><br>
                    📊 Likert scaler: <b>{n_likert} kolom</b></p>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat: {e}")

with col_model:
    st.markdown("### 2️⃣ models_all_labels.pkl")
    f_model = st.file_uploader("Upload models_all_labels.pkl", type=["pkl"], key="up_model")
    if f_model:
        try:
            bundle = pickle.load(io.BytesIO(f_model.read()))
            if not isinstance(bundle, dict):
                st.error("❌ Format tidak valid. Diharapkan dict {label: {...}}")
            else:
                st.session_state.all_models = bundle
                st.session_state.hasil_eval  = None
                n_lbl = len(bundle)
                st.markdown(f"""<div class="model-card">
                    <h4>✅ Model Aktif</h4>
                    <p>🏷️ Jumlah label: <b>{n_lbl}</b><br>
                    📋 Label: <b>{', '.join(LABEL_DISPLAY.get(k, k) for k in list(bundle.keys())[:4])}...</b></p>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat: {e}")

# ---------------------------------------------------------------------------
# Status model aktif
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📌 Status Model Saat Ini")

prep    = st.session_state.preprocessor
models  = st.session_state.all_models

if prep is None and models is None:
    st.info("Belum ada model yang dimuat. Upload kedua file di atas.")
else:
    c1, c2 = st.columns(2)

    with c1:
        if prep is not None:
            st.success("✅ Preprocessor berhasil dimuat")
            with st.expander("🔍 Detail Preprocessor"):
                st.markdown("**Kolom Teks (TF-IDF):**")
                for col in prep["text_columns"]:
                    vec = prep["tfidf_vectorizers"].get(col)
                    n_feat = len(vec.vocabulary_) if vec else "?"
                    st.markdown(f"- `{col}` → **{n_feat}** fitur")
                st.markdown("**Kolom Likert (Scaler):**")
                for col in prep["likert_columns"]:
                    st.markdown(f"- `{col.strip()[:60]}...`")
        else:
            st.markdown('<div class="warn-card"><h4>⚠️ Preprocessor belum dimuat</h4>'
                        '<p>Upload preprocessor.pkl di atas.</p></div>', unsafe_allow_html=True)

    with c2:
        if models is not None:
            st.success(f"✅ {len(models)} model berhasil dimuat")
            with st.expander("🔍 Akurasi per Label"):
                rows = []
                for lbl, info in models.items():
                    rows.append({
                        "Label": LABEL_DISPLAY.get(lbl, lbl),
                        "SVM": f"{info['accuracy']['svm']:.2%}",
                        "LR":  f"{info['accuracy']['lr']:.2%}",
                        "Terbaik": "SVM" if info['accuracy']['svm'] >= info['accuracy']['lr'] else "LR",
                    })
                st.dataframe(pd.DataFrame(rows).set_index("Label"), use_container_width=True)
        else:
            st.markdown('<div class="warn-card"><h4>⚠️ Model belum dimuat</h4>'
                        '<p>Upload models_all_labels.pkl di atas.</p></div>', unsafe_allow_html=True)

    if prep is not None or models is not None:
        st.divider()
        if st.button("🗑️ Hapus Semua Model", type="secondary"):
            st.session_state.preprocessor = None
            st.session_state.all_models   = None
            st.session_state.hasil_eval   = None
            st.rerun()