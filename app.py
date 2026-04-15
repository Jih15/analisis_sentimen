import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# Page config — harus dipanggil pertama sebelum apapun
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentimenTA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "df_latih" not in st.session_state:
    st.session_state.df_latih = []
if "df_uji" not in st.session_state:
    st.session_state.df_uji = []
if "hasil" not in st.session_state:
    st.session_state.hasil = None

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E3A5F;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #CBD5E1 !important;
    }

    /* Card containers */
    .summary-card {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border-radius: 12px;
        padding: 24px 20px;
        color: white;
        text-align: center;
        margin-bottom: 8px;
    }
    .summary-card .value {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .summary-card .label {
        font-size: 0.88rem;
        opacity: 0.85;
        margin-top: 4px;
    }

    .card-green  { background: linear-gradient(135deg, #059669, #047857) !important; }
    .card-orange { background: linear-gradient(135deg, #D97706, #B45309) !important; }
    .card-purple { background: linear-gradient(135deg, #7C3AED, #6D28D9) !important; }

    /* Step cards */
    .step-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 18px 16px;
        height: 100%;
    }
    .step-card h4 { margin: 0 0 8px 0; color: #1E293B; }
    .step-card p  { margin: 0; color: #64748B; font-size: 0.9rem; line-height: 1.5; }

    /* Page title */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1E293B;
    }
    .page-subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-top: -8px;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ok  { background:#D1FAE5; color:#065F46; }
    .badge-warn { background:#FEF3C7; color:#92400E; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="page-title">🔍 SentimenTA</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Analisis Sentimen Pengguna terhadap Aplikasi Pendeteksi '
    "Lowongan Kerja Palsu &nbsp;|&nbsp; SVM vs Regresi Logistik</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
n_latih   = len(st.session_state.df_latih)
n_uji     = len(st.session_state.df_uji)
has_hasil = st.session_state.hasil is not None

if has_hasil:
    svm = st.session_state.hasil["svm"]
    lr  = st.session_state.hasil["lr"]
    best     = svm if svm["accuracy"] >= lr["accuracy"] else lr
    best_acc = f"{best['accuracy']:.2%}"
    best_nm  = "SVM" if svm["accuracy"] >= lr["accuracy"] else "LR"
else:
    best_acc = "—"
    best_nm  = "—"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="summary-card">'
        f'  <div class="value">{n_latih}</div>'
        f'  <div class="label">📝 Data Latih</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="summary-card card-green">'
        f'  <div class="value">{n_uji}</div>'
        f'  <div class="label">🧪 Data Uji</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="summary-card card-orange">'
        f'  <div class="value">{best_nm}</div>'
        f'  <div class="label">🏆 Model Terbaik</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="summary-card card-purple">'
        f'  <div class="value">{best_acc}</div>'
        f'  <div class="label">🎯 Akurasi Tertinggi</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Workflow guide
# ---------------------------------------------------------------------------
st.subheader("📋 Panduan Penggunaan")
s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        """<div class="step-card">
        <h4>① Input Data Latih</h4>
        <p>Upload file CSV atau tambahkan data secara manual.
        Setiap data harus memiliki teks umpan balik dan label sentimen
        (Positif, Negatif, atau Netral).</p>
        </div>""",
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        """<div class="step-card">
        <h4>② Input Data Uji</h4>
        <p>Upload atau masukkan data uji beserta label yang benar.
        Data ini digunakan untuk mengukur performa model setelah
        proses pelatihan selesai.</p>
        </div>""",
        unsafe_allow_html=True,
    )
with s3:
    st.markdown(
        """<div class="step-card">
        <h4>③ Lihat Hasil Evaluasi</h4>
        <p>Latih kedua model (SVM & Regresi Logistik) dan bandingkan
        performa melalui Accuracy, Precision, Recall, F1-Score,
        serta Confusion Matrix interaktif.</p>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Label distribution chart (tampil kalau data latih sudah ada)
# ---------------------------------------------------------------------------
if n_latih > 0:
    st.divider()
    st.subheader("📊 Distribusi Label Data Latih")

    df = pd.DataFrame(st.session_state.df_latih)
    label_counts = df["label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Jumlah"]

    color_map = {"Positif": "#059669", "Negatif": "#DC2626", "Netral": "#D97706"}

    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        fig = px.pie(
            label_counts,
            values="Jumlah",
            names="Label",
            color="Label",
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig.update_layout(
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.15),
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Rincian per label:**")
        total = label_counts["Jumlah"].sum()
        for _, row in label_counts.iterrows():
            pct = row["Jumlah"] / total * 100
            badge_cls = (
                "badge-ok" if row["Label"] == "Positif"
                else "badge-warn" if row["Label"] == "Netral"
                else "badge"
            )
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:10px 0;border-bottom:1px solid #E2E8F0">'
                f'  <span style="font-weight:500">{row["Label"]}</span>'
                f'  <span><b>{row["Jumlah"]}</b> data &nbsp;'
                f'  <span style="color:#94A3B8;font-size:0.85rem">({pct:.1f}%)</span></span>'
                f"</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Quick results preview (kalau model sudah dilatih)
# ---------------------------------------------------------------------------
if has_hasil:
    st.divider()
    st.subheader("📈 Ringkasan Hasil Terakhir")

    metrics = ["accuracy", "precision", "recall", "f1"]
    labels_display = ["Accuracy", "Precision", "Recall", "F1-Score"]

    rows = []
    for key, res in [("SVM", svm), ("Regresi Logistik", lr)]:
        rows.append({
            "Model": key,
            **{lbl: f"{res[m]:.4f}" for m, lbl in zip(metrics, labels_display)},
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Model"),
        use_container_width=True,
    )
    st.caption("Buka halaman **Hasil** untuk melihat Confusion Matrix dan analisis lengkap.")
