import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_batch
from utils.model import run_pipeline, save_model, load_model, evaluate_loaded

st.set_page_config(page_title="Hasil | SentimenTA", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("df_latih",      []),
    ("df_uji",        []),
    ("hasil",         None),
    ("loaded_bundle", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] { background-color: #1E3A5F; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
.page-title { font-size:1.8rem; font-weight:700; color:#1E293B; }
.model-header { font-size:1.1rem; font-weight:600; color:#1E293B; margin-bottom:12px; }
.winner-badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    background:#D1FAE5; color:#065F46; font-size:0.82rem; font-weight:600;
}
.info-box {
    background:#EFF6FF; border:1px solid #BFDBFE;
    border-radius:10px; padding:18px 20px; margin-bottom:8px;
}
.info-box h4 { margin:0 0 6px 0; color:#1D4ED8; }
.info-box p  { margin:0; color:#1E40AF; font-size:0.9rem; line-height:1.6; }
.loaded-card {
    background:#F0FDF4; border:1px solid #86EFAC;
    border-radius:10px; padding:16px 20px; margin-bottom:12px;
}
.loaded-card h4 { margin:0 0 6px 0; color:#166534; }
.loaded-card p  { margin:0; color:#15803D; font-size:0.88rem; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="page-title">📊 Hasil Evaluasi Model</p>', unsafe_allow_html=True)
st.caption("Latih model baru, atau load model yang sudah ada untuk langsung melihat hasil.")
st.divider()

n_latih = len(st.session_state.df_latih)
n_uji   = len(st.session_state.df_uji)

# ============================================================================
# SECTION A — LOAD MODEL
# ============================================================================
with st.expander(
    "📂 Load Model yang Sudah Ada (.pkl)",
    expanded=(st.session_state.loaded_bundle is None and st.session_state.hasil is None),
):
    st.markdown("""<div class="info-box">
        <h4>Punya model yang sudah dilatih sebelumnya?</h4>
        <p>Upload file <code>.pkl</code> hasil export dari sesi sebelumnya.
        Model akan langsung dimuat dan siap dievaluasi pada data uji baru.
        Kamu tidak perlu melatih ulang dari awal.</p>
    </div>""", unsafe_allow_html=True)

    uploaded_model = st.file_uploader(
        "Upload file model (.pkl)", type=["pkl"], key="upload_model_file"
    )

    if uploaded_model:
        try:
            bundle = load_model(uploaded_model.read())
            st.session_state.loaded_bundle = bundle
            st.session_state.hasil = None
            st.success("✅ Model berhasil dimuat!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")

    if st.session_state.loaded_bundle is not None:
        bundle = st.session_state.loaded_bundle
        meta   = bundle.get("metadata", {})
        labels = bundle.get("label_names", [])
        st.markdown(f"""<div class="loaded-card">
            <h4>✅ Model Aktif (dari file)</h4>
            <p>
            📅 Disimpan pada: <b>{meta.get('saved_at', '—')}</b><br>
            📝 Dilatih dengan: <b>{meta.get('n_train', '?')} data latih</b><br>
            🏷️ Label: <b>{', '.join(labels)}</b><br>
            🎯 Akurasi SVM: <b>{meta.get('svm_accuracy', 0):.2%}</b>
            &nbsp;|&nbsp; LR: <b>{meta.get('lr_accuracy', 0):.2%}</b>
            </p>
        </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Hapus Model yang Di-load", type="secondary"):
            st.session_state.loaded_bundle = None
            st.session_state.hasil = None
            st.rerun()

# ============================================================================
# SECTION B — LATIH MODEL BARU
# ============================================================================
st.subheader("🚀 Latih Model Baru")

if n_latih < 10:
    st.warning(
        f"⚠️ Data latih minimal 10 baris. Saat ini: **{n_latih}** baris. "
        "Silakan isi di halaman **Data Latih**."
    )
elif n_uji < 1:
    st.warning("⚠️ Data uji belum diisi. Silakan isi di halaman **Data Uji**.")
else:
    with st.expander("⚙️ Opsi Preprocessing & Training", expanded=False):
        use_stemming = st.checkbox("Gunakan Stemming (PySastrawi)", value=True)
        max_features = st.slider("Jumlah fitur TF-IDF maksimum", 100, 10000, 5000, 500)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Latih & Evaluasi", type="primary", use_container_width=True)
    with col_info:
        st.markdown(
            f"Melatih dengan **{n_latih} data latih**, "
            f"evaluasi pada **{n_uji} data uji**."
        )

    if run_btn:
        with st.spinner("⏳ Memproses..."):
            df_lt = pd.DataFrame(st.session_state.df_latih)
            df_uj = pd.DataFrame(st.session_state.df_uji)

            X_train = preprocess_batch(df_lt["teks"].tolist(), use_stemming)
            y_train = df_lt["label"].tolist()
            X_test  = preprocess_batch(df_uj["teks"].tolist(), use_stemming)
            y_test  = df_uj["label"].tolist()
            label_names = sorted(set(y_train) | set(y_test))

            try:
                hasil = run_pipeline(X_train, y_train, X_test, y_test, label_names)
                hasil["label_names"]      = label_names
                hasil["loaded_from_file"] = False
                st.session_state.hasil         = hasil
                st.session_state.loaded_bundle = None
                st.success("✅ Model berhasil dilatih!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

# ============================================================================
# SECTION C — EVALUASI MODEL DARI FILE pada data uji baru
# ============================================================================
if st.session_state.loaded_bundle is not None and st.session_state.hasil is None:
    st.subheader("🧪 Evaluasi Model (dari file) pada Data Uji")

    if n_uji < 1:
        st.info("ℹ️ Isi data uji di halaman **Data Uji** untuk mulai evaluasi.")
    else:
        with st.expander("⚙️ Opsi", expanded=False):
            use_stemming_load = st.checkbox("Gunakan Stemming", value=True, key="stem_load")

        col_ev, col_inf = st.columns([1, 3])
        with col_ev:
            eval_btn = st.button("▶️ Evaluasi Sekarang", type="primary", use_container_width=True)
        with col_inf:
            st.markdown(f"Evaluasi pada **{n_uji} data uji**.")

        if eval_btn:
            with st.spinner("⏳ Mengevaluasi..."):
                df_uj = pd.DataFrame(st.session_state.df_uji)
                try:
                    hasil = evaluate_loaded(
                        st.session_state.loaded_bundle,
                        df_uj["teks"].tolist(),
                        df_uj["label"].tolist(),
                        use_stemming=use_stemming_load,
                    )
                    st.session_state.hasil = hasil
                    st.success("✅ Evaluasi selesai!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

# ============================================================================
# VIEW-ONLY placeholder — belum ada model sama sekali
# ============================================================================
if st.session_state.hasil is None and st.session_state.loaded_bundle is None:
    st.divider()
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#94A3B8;">
        <div style="font-size:4rem;">📊</div>
        <h3 style="color:#CBD5E1; margin:16px 0 8px 0;">Belum ada hasil untuk ditampilkan</h3>
        <p style="max-width:420px; margin:0 auto; line-height:1.6;">
            Latih model baru menggunakan data latih &amp; uji yang sudah diisi,
            atau load model <code>.pkl</code> dari sesi sebelumnya
            menggunakan panel di atas.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if st.session_state.hasil is None:
    st.stop()

# ============================================================================
# SECTION D — TAMPILKAN HASIL
# ============================================================================
hasil       = st.session_state.hasil
svm         = hasil["svm"]
lr          = hasil["lr"]
label_names = hasil.get("label_names", svm["labels"])
svm_better  = svm["accuracy"] >= lr["accuracy"]

if hasil.get("loaded_from_file"):
    meta = hasil.get("metadata", {})
    st.info(
        f"ℹ️ Menggunakan **model dari file** "
        f"(disimpan: {meta.get('saved_at', '—')}, "
        f"dilatih dengan {meta.get('n_train', '?')} data)."
    )

st.divider()

# ── D1: Metrik ──────────────────────────────────────────────────────────────
st.subheader("📈 Perbandingan Metrik Evaluasi")

metrics_list = [
    ("accuracy",  "Accuracy",  "Proporsi prediksi benar dari keseluruhan data uji."),
    ("precision", "Precision", "Dari semua prediksi positif, berapa yang benar-benar positif."),
    ("recall",    "Recall",    "Dari semua data positif aktual, berapa yang berhasil diprediksi."),
    ("f1",        "F1-Score",  "Rata-rata harmonik antara Precision dan Recall."),
]

for m_key, m_label, m_desc in metrics_list:
    svm_val = svm[m_key]
    lr_val  = lr[m_key]
    better  = "SVM" if svm_val >= lr_val else "LR"

    st.markdown(
        f"**{m_label}** &nbsp;"
        f"<span style='color:#94A3B8;font-size:0.85rem'>{m_desc}</span>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.metric(
            f"{'🏆 ' if better == 'SVM' else ''}SVM",
            f"{svm_val:.4f}",
            delta=f"+{svm_val - lr_val:.4f}" if svm_val > lr_val else None,
        )
    with c2:
        st.metric(
            f"{'🏆 ' if better == 'LR' else ''}Regresi Logistik",
            f"{lr_val:.4f}",
            delta=f"+{lr_val - svm_val:.4f}" if lr_val > svm_val else None,
        )
    with c3:
        st.markdown(
            f'<div style="padding:8px 0">'
            f'<span class="winner-badge">✓ {better} unggul</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='margin:6px 0;border-color:#F1F5F9'>", unsafe_allow_html=True)

# ── D2: Tabel ───────────────────────────────────────────────────────────────
st.subheader("📋 Tabel Ringkasan")
st.dataframe(
    pd.DataFrame({
        "Model":      ["SVM", "Regresi Logistik"],
        "Accuracy":   [f"{svm['accuracy']:.4f}",  f"{lr['accuracy']:.4f}"],
        "Precision":  [f"{svm['precision']:.4f}", f"{lr['precision']:.4f}"],
        "Recall":     [f"{svm['recall']:.4f}",    f"{lr['recall']:.4f}"],
        "F1-Score":   [f"{svm['f1']:.4f}",        f"{lr['f1']:.4f}"],
        "Kesimpulan": ["✅ Terbaik" if svm_better else "—",
                       "—" if svm_better else "✅ Terbaik"],
    }).set_index("Model"),
    use_container_width=True,
)

# ── D3: Bar chart ───────────────────────────────────────────────────────────
st.subheader("📊 Grafik Perbandingan")
chart_df = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"] * 2,
    "Nilai":  [svm["accuracy"], svm["precision"], svm["recall"], svm["f1"],
               lr["accuracy"],  lr["precision"],  lr["recall"],  lr["f1"]],
    "Model":  ["SVM"] * 4 + ["Regresi Logistik"] * 4,
})
fig_bar = px.bar(
    chart_df, x="Metrik", y="Nilai", color="Model", barmode="group",
    color_discrete_map={"SVM": "#2563EB", "Regresi Logistik": "#059669"},
    range_y=[0, 1.1], text_auto=".3f",
)
fig_bar.update_layout(
    height=380, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=20, b=20),
    yaxis=dict(gridcolor="#F1F5F9"),
)
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# ── D4: Confusion Matrix ────────────────────────────────────────────────────
def plot_cm(cm, labels, title, color):
    total = cm.sum(axis=1, keepdims=True)
    pct   = np.where(total > 0, cm / total * 100, 0)
    ann   = [
        dict(
            x=labels[j], y=labels[i],
            text=f"<b>{cm[i,j]}</b><br><span style='font-size:11px'>{pct[i,j]:.1f}%</span>",
            showarrow=False,
            font=dict(color="white" if cm[i,j] > cm.max() / 2 else "#1E293B", size=13),
        )
        for i in range(len(labels)) for j in range(len(labels))
    ]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=color, showscale=False))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Prediksi", side="bottom"),
        yaxis=dict(title="Aktual", autorange="reversed"),
        annotations=ann, height=320,
        margin=dict(t=50, b=50, l=60, r=20),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig

st.subheader("🔢 Confusion Matrix")
cm1, cm2 = st.columns(2)
with cm1:
    badge = '<span class="winner-badge">🏆 Terbaik</span>' if svm_better else ""
    st.markdown(f'<p class="model-header">SVM &nbsp;{badge}</p>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_cm(svm["cm"], label_names, "Confusion Matrix — SVM", "Blues"),
        use_container_width=True,
    )
with cm2:
    badge = '<span class="winner-badge">🏆 Terbaik</span>' if not svm_better else ""
    st.markdown(f'<p class="model-header">Regresi Logistik &nbsp;{badge}</p>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_cm(lr["cm"], label_names, "Confusion Matrix — Regresi Logistik", "Greens"),
        use_container_width=True,
    )
st.caption("Baris = label **aktual**, Kolom = label **prediksi**. Diagonal = prediksi benar.")

# ── D5: Detail prediksi ─────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Detail Prediksi Data Uji")

best_res  = svm if svm_better else lr
best_name = "SVM" if svm_better else "Regresi Logistik"

df_uji    = pd.DataFrame(st.session_state.df_uji)
df_result = df_uji.copy()
df_result["Prediksi"] = best_res["y_pred"]
df_result["Benar?"]   = (df_result["label"] == df_result["Prediksi"]).map(
    {True: "✅", False: "❌"}
)
df_result.columns = ["Teks", "Label Aktual", "Prediksi", "Benar?"]

st.caption(f"Prediksi menggunakan model terbaik: **{best_name}**")
st.dataframe(df_result, use_container_width=True, height=min(400, 50 + len(df_result) * 35))
st.download_button(
    "⬇️ Export Hasil Prediksi (CSV)",
    df_result.to_csv(index=False).encode("utf-8"),
    "hasil_prediksi.csv", "text/csv",
)

# ── D6: Kesimpulan ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📝 Kesimpulan")

best_acc = max(svm["accuracy"], lr["accuracy"])
loser    = lr if svm_better else svm
loser_nm = "Regresi Logistik" if svm_better else "SVM"
best_nm  = "SVM" if svm_better else "Regresi Logistik"

st.success(
    f"Berdasarkan evaluasi pada **{hasil.get('n_test', n_uji)} data uji**, "
    f"model **{best_nm}** menunjukkan performa terbaik dengan "
    f"akurasi **{best_acc:.2%}**, F1-Score **{best_res['f1']:.4f}**, "
    f"Precision **{best_res['precision']:.4f}**, dan Recall **{best_res['recall']:.4f}**. "
    f"Unggul dibandingkan {loser_nm} (akurasi: {loser['accuracy']:.2%})."
)

# ============================================================================
# SECTION E — SIMPAN MODEL (hanya kalau model baru dilatih, bukan dari file)
# ============================================================================
if not hasil.get("loaded_from_file", False):
    st.divider()
    st.subheader("💾 Simpan Model")
    st.markdown(
        "Export model yang baru dilatih ke file `.pkl` agar bisa di-load "
        "kembali di sesi berikutnya **tanpa perlu melatih ulang**."
    )
    col_save, col_info = st.columns([1, 3])
    with col_save:
        st.download_button(
            label="⬇️ Download Model (.pkl)",
            data=save_model(hasil, label_names),
            file_name="sentimen_model.pkl",
            mime="application/octet-stream",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.markdown(
            f"File berisi: TF-IDF vectorizer, model SVM, model Regresi Logistik, "
            f"label `{label_names}`, dan metadata training (tanggal, jumlah data, akurasi)."
        )
