import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

st.set_page_config(page_title="Hasil | SentimenTA", page_icon="📊", layout="wide")

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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
section[data-testid="stSidebar"] { background-color: #0F1B2D; }
section[data-testid="stSidebar"] * { color: #94A3B8 !important; }
.page-title { font-size:1.8rem; font-weight:800; color:#0F1B2D; letter-spacing:-.02em; }
.winner-badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    background:#D1FAE5; color:#065F46; font-size:0.78rem; font-weight:600;
}
.label-header {
    font-size:1.05rem; font-weight:700; color:#1E293B;
    border-left:4px solid #2563EB; padding-left:10px; margin:16px 0 8px 0;
}
.metric-row {
    background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;
    padding:14px 16px; margin-bottom:6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="page-title">📊 Hasil Evaluasi Model</p>', unsafe_allow_html=True)
st.caption("Perbandingan performa SVM vs Regresi Logistik untuk setiap label sentimen.")
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
    "label_manfaat_cv":          "Manfaat CV",
    "label_fitur_cv":            "Fitur CV",
    "label_informatif_edukasi":  "Informatif Edukasi",
    "label_konten_edukasi":      "Konten Edukasi",
    "label_kepuasan_keseluruhan":"Kepuasan Keseluruhan",
    "label_kelebihan_aplikasi":  "Kelebihan Aplikasi",
    "label_kekurangan_aplikasi": "Kekurangan Aplikasi",
    "label_saran_kritik":        "Saran & Kritik",
}

# ---------------------------------------------------------------------------
# Guard: cek prerequisite
# ---------------------------------------------------------------------------
df      = st.session_state.df_dataset
prep    = st.session_state.preprocessor
models  = st.session_state.all_models

if df is None:
    st.warning("⚠️ Dataset belum diupload. Silakan ke halaman **Dataset** terlebih dahulu.")
    st.stop()
if prep is None or models is None:
    st.warning("⚠️ Model belum dimuat. Silakan ke halaman **Model** terlebih dahulu.")
    st.stop()

# ---------------------------------------------------------------------------
# Fungsi transform
# ---------------------------------------------------------------------------
def transform_features(df_input, prep):
    tfidf_vectorizers = prep["tfidf_vectorizers"]
    scaler            = prep["scaler"]
    text_cols         = prep["text_columns"]
    likert_cols       = prep["likert_columns"]

    tfidf_parts = []
    for col in text_cols:
        vals = df_input[col].fillna('').astype(str).values if col in df_input.columns else [''] * len(df_input)
        tfidf_parts.append(tfidf_vectorizers[col].transform(vals))

    likert_data = np.zeros((len(df_input), len(likert_cols)))
    for i, col in enumerate(likert_cols):
        if col in df_input.columns:
            likert_data[:, i] = pd.to_numeric(df_input[col], errors='coerce').fillna(0).values

    likert_scaled = csr_matrix(scaler.transform(likert_data))
    return hstack(tfidf_parts + [likert_scaled])

# ---------------------------------------------------------------------------
# Tombol evaluasi
# ---------------------------------------------------------------------------
if st.session_state.hasil_eval is None:
    st.info("ℹ️ Klik tombol di bawah untuk menjalankan evaluasi model pada dataset yang sudah diupload.")
    if st.button("▶️ Jalankan Evaluasi Semua Label", type="primary", use_container_width=False):
        hasil_dict = {}
        labels_present = [l for l in ALL_LABELS if l in df.columns]

        progress = st.progress(0, text="⏳ Mengevaluasi...")
        for i, lbl in enumerate(labels_present):
            subset = df[TEXT_COLUMNS + LIKERT_COLUMNS + [lbl]].copy()
            # fill missing
            for col in TEXT_COLUMNS:
                if col not in subset.columns:
                    subset[col] = ''
                subset[col] = subset[col].fillna('').astype(str)
            for col in LIKERT_COLUMNS:
                if col not in subset.columns:
                    subset[col] = 0
                subset[col] = pd.to_numeric(subset[col], errors='coerce').fillna(0)

            subset = subset.dropna(subset=[lbl])
            if len(subset) < 3:
                continue

            y_true = subset[lbl].astype(str).str.lower().values
            X = transform_features(subset, prep)

            model_info = models.get(lbl)
            if model_info is None:
                continue

            le       = model_info["label_encoder"]
            svm_mdl  = model_info["svm"]
            lr_mdl   = model_info["lr"]
            classes  = model_info["classes"]

            y_enc = []
            valid_idx = []
            for j, val in enumerate(y_true):
                if val in le.classes_:
                    y_enc.append(le.transform([val])[0])
                    valid_idx.append(j)

            if len(valid_idx) < 3:
                continue

            X_valid = X[valid_idx]
            y_valid = np.array(y_enc)

            svm_pred = svm_mdl.predict(X_valid)
            lr_pred  = lr_mdl.predict(X_valid)

            def safe_metrics(y_true, y_pred, labels):
                acc  = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
                rec  = recall_score(y_true, y_pred,    average="weighted", labels=labels, zero_division=0)
                f1   = f1_score(y_true, y_pred,        average="weighted", labels=labels, zero_division=0)
                cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(le.classes_))))
                return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm, "y_pred": y_pred}

            label_ints = list(range(len(le.classes_)))
            svm_res = safe_metrics(y_valid, svm_pred, label_ints)
            lr_res  = safe_metrics(y_valid, lr_pred,  label_ints)

            svm_res["svm_acc"] = svm_res["acc"]
            lr_res["lr_acc"]   = lr_res["acc"]

            hasil_dict[lbl] = {
                "svm": svm_res,
                "lr":  lr_res,
                "svm_acc":  svm_res["acc"],
                "lr_acc":   lr_res["acc"],
                "classes":  classes,
                "le":       le,
                "n":        len(valid_idx),
                "y_true":   y_valid,
            }
            progress.progress((i + 1) / len(labels_present), text=f"⏳ Memproses {LABEL_DISPLAY.get(lbl, lbl)}...")

        progress.empty()
        st.session_state.hasil_eval = hasil_dict
        st.success(f"✅ Evaluasi selesai untuk **{len(hasil_dict)}** label!")
        st.rerun()
    st.stop()

hasil = st.session_state.hasil_eval

# ---------------------------------------------------------------------------
# Tombol reset evaluasi
# ---------------------------------------------------------------------------
col_h, col_r = st.columns([4, 1])
with col_r:
    if st.button("🔄 Evaluasi Ulang", type="secondary"):
        st.session_state.hasil_eval = None
        st.rerun()

# ============================================================================
# D1: Ringkasan akurasi semua label
# ============================================================================
st.subheader("📈 Ringkasan Akurasi Semua Label")

summary_rows = []
for lbl, h in hasil.items():
    best = "SVM" if h["svm_acc"] >= h["lr_acc"] else "LR"
    summary_rows.append({
        "Label": LABEL_DISPLAY.get(lbl, lbl),
        "SVM Acc": h["svm_acc"],
        "LR Acc":  h["lr_acc"],
        "Terbaik": best,
        "Δ":       abs(h["svm_acc"] - h["lr_acc"]),
        "N Data":  h["n"],
    })

df_summary = pd.DataFrame(summary_rows)
avg_svm = df_summary["SVM Acc"].mean()
avg_lr  = df_summary["LR Acc"].mean()

# KPI baris atas
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Rata-rata Akurasi SVM", f"{avg_svm:.2%}")
with k2:
    st.metric("Rata-rata Akurasi LR", f"{avg_lr:.2%}")
with k3:
    svm_wins = sum(1 for h in hasil.values() if h["svm_acc"] >= h["lr_acc"])
    st.metric("Label Unggul SVM", f"{svm_wins}/{len(hasil)}")
with k4:
    lr_wins = len(hasil) - svm_wins
    st.metric("Label Unggul LR", f"{lr_wins}/{len(hasil)}")

# Tabel ringkasan
st.dataframe(
    df_summary.style.format({
        "SVM Acc": "{:.2%}", "LR Acc": "{:.2%}", "Δ": "{:.2%}"
    }).background_gradient(subset=["SVM Acc", "LR Acc"], cmap="Greens"),
    use_container_width=True,
    height=420,
)

# ============================================================================
# D2: Bar chart perbandingan akurasi
# ============================================================================
st.subheader("📊 Grafik Akurasi per Label")
chart_rows = []
for lbl, h in hasil.items():
    name = LABEL_DISPLAY.get(lbl, lbl)
    chart_rows.append({"Label": name, "Akurasi": h["svm_acc"], "Model": "SVM"})
    chart_rows.append({"Label": name, "Akurasi": h["lr_acc"],  "Model": "Regresi Logistik"})

fig_bar = px.bar(
    pd.DataFrame(chart_rows), x="Label", y="Akurasi", color="Model",
    barmode="group", text_auto=".1%",
    color_discrete_map={"SVM": "#2563EB", "Regresi Logistik": "#059669"},
    range_y=[0, 1.15], height=420,
)
fig_bar.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickangle=-30), legend=dict(orientation="h", y=1.05),
    margin=dict(t=20, b=80),
)
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================================
# D3: Detail per label (expander)
# ============================================================================
st.divider()
st.subheader("🔍 Detail Evaluasi per Label")
st.caption("Klik label untuk melihat detail metrik dan confusion matrix.")

def plot_cm(cm, classes, title, colorscale):
    total = cm.sum(axis=1, keepdims=True)
    pct   = np.where(total > 0, cm / total * 100, 0)
    ann = [
        dict(
            x=classes[j], y=classes[i],
            text=f"<b>{cm[i,j]}</b><br><span style='font-size:10px'>{pct[i,j]:.0f}%</span>",
            showarrow=False,
            font=dict(color="white" if cm[i,j] > (cm.max() / 2) else "#1E293B", size=12),
        )
        for i in range(len(classes)) for j in range(len(classes))
    ]
    fig = go.Figure(go.Heatmap(z=cm, x=classes, y=classes, colorscale=colorscale, showscale=False))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Prediksi"),
        yaxis=dict(title="Aktual", autorange="reversed"),
        annotations=ann, height=280,
        margin=dict(t=40, b=40, l=50, r=10),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig

for lbl, h in hasil.items():
    display_name = LABEL_DISPLAY.get(lbl, lbl)
    svm_h  = h["svm"]
    lr_h   = h["lr"]
    best   = "SVM" if h["svm_acc"] >= h["lr_acc"] else "LR"
    classes = h["classes"]

    with st.expander(f"{'🏆 ' if best == 'SVM' else ''}**{display_name}**  —  SVM: {h['svm_acc']:.1%}  |  LR: {h['lr_acc']:.1%}  ({'SVM unggul' if best == 'SVM' else 'LR unggul'})"):
        m1, m2, m3, m4 = st.columns(4)
        for col, key, label in [
            (m1, "acc",  "Accuracy"),
            (m2, "prec", "Precision"),
            (m3, "rec",  "Recall"),
            (m4, "f1",   "F1-Score"),
        ]:
            with col:
                delta_val = svm_h[key] - lr_h[key]
                st.metric(
                    f"SVM {label}", f"{svm_h[key]:.4f}",
                    delta=f"{delta_val:+.4f}",
                )
                st.metric(
                    f"LR {label}", f"{lr_h[key]:.4f}",
                )

        # Confusion matrices
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.plotly_chart(
                plot_cm(svm_h["cm"], classes, f"Confusion Matrix — SVM", "Blues"),
                use_container_width=True,
            )
        with cm_col2:
            st.plotly_chart(
                plot_cm(lr_h["cm"], classes, f"Confusion Matrix — LR", "Greens"),
                use_container_width=True,
            )
        st.caption(f"Dievaluasi pada **{h['n']} data** | Kelas: {', '.join(classes)}")

# ============================================================================
# D4: Kesimpulan global
# ============================================================================
st.divider()
st.subheader("📝 Kesimpulan")

best_global = "SVM" if avg_svm >= avg_lr else "Regresi Logistik"
best_acc_g  = max(avg_svm, avg_lr)
st.success(
    f"Secara rata-rata dari **{len(hasil)} label** yang dievaluasi, "
    f"model **{best_global}** memberikan akurasi tertinggi sebesar **{best_acc_g:.2%}**. "
    f"SVM unggul pada **{svm_wins}** label, sementara Regresi Logistik unggul pada **{lr_wins}** label."
)

# Export ringkasan
csv_sum = df_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Export Ringkasan Akurasi (CSV)",
    data=csv_sum,
    file_name="ringkasan_akurasi.csv",
    mime="text/csv",
)