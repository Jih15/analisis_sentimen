import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="SentimenTA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "df_dataset":    None,   # DataFrame dataset lengkap (dari upload xlsx)
    "preprocessor":  None,   # dict {tfidf_vectorizers, scaler, text_columns, likert_columns}
    "all_models":    None,   # dict {label: {svm, lr, label_encoder, classes, accuracy}}
    "hasil_eval":    None,   # hasil evaluasi per label
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

section[data-testid="stSidebar"] { background-color: #0F1B2D; }
section[data-testid="stSidebar"] * { color: #94A3B8 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }

.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #0F1B2D; line-height: 1.15;
    letter-spacing: -0.03em;
}
.hero-sub {
    color: #64748B; font-size: 1.05rem; margin-top: 6px; line-height: 1.6;
}
.accent { color: #2563EB; }

.kpi-card {
    background: #fff; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 22px 20px; text-align: center; transition: box-shadow .2s;
}
.kpi-card:hover { box-shadow: 0 4px 20px rgba(37,99,235,.10); }
.kpi-card .kpi-val {
    font-size: 2.2rem; font-weight: 800; color: #0F1B2D; font-family: 'JetBrains Mono', monospace;
}
.kpi-card .kpi-lbl { font-size: 0.82rem; color: #94A3B8; margin-top: 4px; font-weight: 500; }
.kpi-card.blue  .kpi-val { color: #2563EB; }
.kpi-card.green .kpi-val { color: #059669; }
.kpi-card.orange .kpi-val { color: #D97706; }
.kpi-card.purple .kpi-val { color: #7C3AED; }

.step-wrap {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB;
    border-radius: 10px; padding: 18px 16px; height: 100%;
}
.step-wrap h4 { margin: 0 0 8px 0; color: #1E293B; font-size: 0.95rem; }
.step-wrap p  { margin: 0; color: #64748B; font-size: 0.85rem; line-height: 1.6; }

.label-chip {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin: 2px;
    background: #EFF6FF; color: #1D4ED8;
}
.chip-green { background: #D1FAE5; color: #065F46; }
.chip-yellow { background: #FEF3C7; color: #92400E; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown(
        '<p class="hero-title">🔍 <span class="accent">SentimenTA</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Analisis sentimen multi-label pengguna terhadap '
        '<b>Aplikasi Pendeteksi Lowongan Kerja Palsu</b> — SVM vs Regresi Logistik</p>',
        unsafe_allow_html=True,
    )
st.divider()

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
ds     = st.session_state.df_dataset
models = st.session_state.all_models
hasil  = st.session_state.hasil_eval

n_responden = len(ds) if ds is not None else 0
n_labels    = len(models) if models else 0
n_labels_done = len(hasil) if hasil else 0

if hasil:
    avg_svm = sum(h["svm_acc"] for h in hasil.values()) / len(hasil)
    avg_lr  = sum(h["lr_acc"]  for h in hasil.values()) / len(hasil)
    best_acc_str = f"{max(avg_svm, avg_lr):.1%}"
    best_model   = "SVM" if avg_svm >= avg_lr else "LR"
else:
    best_acc_str = "—"
    best_model   = "—"

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "blue",   str(n_responden),    "📋 Responden Dataset"),
    (c2, "green",  str(n_labels),       "🏷️ Label Dilatih"),
    (c3, "orange", best_model,          "🏆 Model Terbaik"),
    (c4, "purple", best_acc_str,        "🎯 Akurasi Rata-rata"),
]
for col, cls, val, lbl in cards:
    with col:
        st.markdown(
            f'<div class="kpi-card {cls}">'
            f'  <div class="kpi-val">{val}</div>'
            f'  <div class="kpi-lbl">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Panduan penggunaan
# ---------------------------------------------------------------------------
st.subheader("📋 Panduan Penggunaan")
s1, s2, s3, s4 = st.columns(4)
steps = [
    ("① Upload Dataset", "Upload file <code>dataset_preprocessed.xlsx</code> di halaman <b>Dataset</b>. File harus memiliki kolom teks, Likert, dan label sentimen."),
    ("② Upload Model", "Upload <code>preprocessor.pkl</code> dan <code>models_all_labels.pkl</code> di halaman <b>Model</b> (hasil dari training script)."),
    ("③ Lihat Hasil", "Halaman <b>Hasil</b> menampilkan perbandingan akurasi SVM vs LR untuk setiap label sentimen."),
    ("④ Prediksi Baru", "Masukkan respons baru di halaman <b>Prediksi</b> untuk mendapatkan prediksi sentimen dari model."),
]
for col, (title, desc) in zip([s1, s2, s3, s4], steps):
    with col:
        st.markdown(
            f'<div class="step-wrap"><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Label info
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🏷️ Label Sentimen yang Dianalisis")
st.caption("Setiap aspek aplikasi dianalisis secara terpisah dengan model SVM & Regresi Logistik.")

label_meta = {
    "label_kemudahan_deteksi":   "Kemudahan Deteksi",
    "label_akurasi_deteksi":     "Akurasi Deteksi",
    "label_fitur_deteksi":       "Fitur Deteksi",
    "label_penjelasan_deteksi":  "Penjelasan Deteksi",
    "label_manfaat_cv":          "Manfaat Pembuat CV",
    "label_fitur_cv":            "Fitur Pembuat CV",
    "label_informatif_edukasi":  "Informatif Edukasi",
    "label_konten_edukasi":      "Konten Edukasi",
    "label_kepuasan_keseluruhan":"Kepuasan Keseluruhan",
    "label_kelebihan_aplikasi":  "Kelebihan Aplikasi",
    "label_kekurangan_aplikasi": "Kekurangan Aplikasi",
    "label_saran_kritik":        "Saran & Kritik",
}

chips_html = ""
for key, display in label_meta.items():
    # check if trained
    if models and key in models:
        acc = models[key]["accuracy"]
        best = max(acc["svm"], acc["lr"])
        extra = f' <span style="font-size:.7rem;color:#94A3B8">({best:.0%})</span>'
        cls = "chip-green"
    else:
        extra = ""
        cls = ""
    chips_html += f'<span class="label-chip {cls}">{display}{extra}</span>'

st.markdown(chips_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dataset preview kalau sudah ada
# ---------------------------------------------------------------------------
if ds is not None:
    st.divider()
    st.subheader("📊 Distribusi Dataset")

    label_cols_present = [c for c in label_meta.keys() if c in ds.columns]
    if label_cols_present:
        # Hitung distribusi sentimen per label
        dist_rows = []
        for col in label_cols_present:
            vc = ds[col].value_counts()
            for sentiment, count in vc.items():
                dist_rows.append({
                    "Label": label_meta.get(col, col),
                    "Sentimen": str(sentiment).capitalize(),
                    "Jumlah": count,
                })
        df_dist = pd.DataFrame(dist_rows)

        color_map = {"Positif": "#059669", "Negatif": "#DC2626", "Netral": "#D97706"}
        fig = px.bar(
            df_dist, x="Label", y="Jumlah", color="Sentimen",
            barmode="stack", color_discrete_map=color_map,
            height=380,
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=60),
            xaxis=dict(tickangle=-30),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kolom label tidak ditemukan di dataset. Pastikan nama kolom sesuai.")