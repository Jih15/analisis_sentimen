# import streamlit as st
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Dataset | SentimenTA", page_icon="📋", layout="wide")

# # ---------------------------------------------------------------------------
# # Session state
# # ---------------------------------------------------------------------------
# defaults = {
#     "df_dataset":   None,
#     "preprocessor": None,
#     "all_models":   None,
#     "hasil_eval":   None,
# }
# for k, v in defaults.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ---------------------------------------------------------------------------
# # CSS
# # ---------------------------------------------------------------------------
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
# html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
# section[data-testid="stSidebar"] { background-color: #0F1B2D; }
# section[data-testid="stSidebar"] * { color: #94A3B8 !important; }
# .page-title { font-size:1.8rem; font-weight:800; color:#0F1B2D; letter-spacing:-.02em; }
# .info-box {
#     background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
#     padding:16px 18px; margin-bottom:12px;
# }
# .info-box h4 { margin:0 0 6px 0; color:#1D4ED8; }
# .info-box p  { margin:0; color:#1E40AF; font-size:0.88rem; line-height:1.6; }
# .col-badge {
#     display:inline-block; padding:2px 10px; border-radius:12px;
#     font-size:0.75rem; font-weight:600; margin:2px;
# }
# .badge-text { background:#EDE9FE; color:#5B21B6; }
# .badge-num  { background:#FEF3C7; color:#92400E; }
# .badge-label{ background:#D1FAE5; color:#065F46; }
# </style>
# """, unsafe_allow_html=True)

# st.markdown('<p class="page-title">📋 Dataset</p>', unsafe_allow_html=True)
# st.caption("Upload file Excel dataset kuesioner yang sudah dipreproses untuk melihat distribusi data.")
# st.divider()

# # ---------------------------------------------------------------------------
# # Kolom yang diharapkan
# # ---------------------------------------------------------------------------
# TEXT_COLUMNS = [
#     'prep_fitur_deteksi', 'prep_penjelasan_deteksi', 'prep_fitur_cv',
#     'prep_konten_edukasi', 'prep_kelebihan', 'prep_kekurangan', 'prep_saran',
# ]
# LIKERT_COLUMNS = [
#     'Berdasarkan Pengalam anda menggunakan aplikasi Dan melihat Gambar di atas, Seberapa mudah proses memasukkan data lowongan kerja untuk dideteksi? ',
#     'Seberapa akurat Anda merasakan hasil deteksi yang diberikan oleh aplikasi? ',
#     '  Seberapa bermanfaat fitur "Pembuat CV" bagi Anda?   ',
#     'Seberapa informatif artikel dan tips yang ada di fitur "Konten Edukasi"? ',
#     '  Secara keseluruhan, seberapa puas Anda dengan aplikasi ini?  ',
# ]
# ALL_LABELS = [
#     'label_kemudahan_deteksi', 'label_akurasi_deteksi', 'label_fitur_deteksi',
#     'label_penjelasan_deteksi', 'label_manfaat_cv', 'label_fitur_cv',
#     'label_informatif_edukasi', 'label_konten_edukasi', 'label_kepuasan_keseluruhan',
#     'label_kelebihan_aplikasi', 'label_kekurangan_aplikasi', 'label_saran_kritik',
# ]
# LABEL_DISPLAY = {
#     "label_kemudahan_deteksi":   "Kemudahan Deteksi",
#     "label_akurasi_deteksi":     "Akurasi Deteksi",
#     "label_fitur_deteksi":       "Fitur Deteksi",
#     "label_penjelasan_deteksi":  "Penjelasan Deteksi",
#     "label_manfaat_cv":          "Manfaat CV",
#     "label_fitur_cv":            "Fitur CV",
#     "label_informatif_edukasi":  "Informatif Edukasi",
#     "label_konten_edukasi":      "Konten Edukasi",
#     "label_kepuasan_keseluruhan":"Kepuasan Keseluruhan",
#     "label_kelebihan_aplikasi":  "Kelebihan Aplikasi",
#     "label_kekurangan_aplikasi": "Kekurangan Aplikasi",
#     "label_saran_kritik":        "Saran & Kritik",
# }

# # ---------------------------------------------------------------------------
# # Upload
# # ---------------------------------------------------------------------------
# st.markdown("""<div class="info-box">
#     <h4>Format File yang Diterima</h4>
#     <p>File <code>.xlsx</code> hasil preprocessing dengan kolom berikut:<br>
#     • <b>7 kolom teks</b> (prep_*) — hasil preprocessing teks bebas<br>
#     • <b>5 kolom Likert</b> — nilai skala 1–4<br>
#     • <b>12 kolom label</b> (label_*) — nilai: positif / negatif / netral</p>
# </div>""", unsafe_allow_html=True)

# # Info kolom yang diharapkan
# with st.expander("📑 Lihat daftar kolom yang diharapkan"):
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.markdown("**Kolom Teks (prep_*)**")
#         for col in TEXT_COLUMNS:
#             st.markdown(f'<span class="col-badge badge-text">{col}</span>', unsafe_allow_html=True)
#     with c2:
#         st.markdown("**Kolom Likert**")
#         for col in LIKERT_COLUMNS:
#             short = col.strip()[:40] + "..."
#             st.markdown(f'<span class="col-badge badge-num">{short}</span>', unsafe_allow_html=True)
#     with c3:
#         st.markdown("**Kolom Label**")
#         for col in ALL_LABELS:
#             st.markdown(f'<span class="col-badge badge-label">{col}</span>', unsafe_allow_html=True)

# uploaded = st.file_uploader("Upload file dataset Excel (.xlsx)", type=["xlsx"], key="upload_dataset")

# if uploaded:
#     try:
#         df = pd.read_excel(uploaded)
#         df.columns = [str(c) for c in df.columns]  # pastikan string

#         st.success(f"✅ File berhasil dibaca: **{df.shape[0]} baris**, **{df.shape[1]} kolom**")

#         # Cek kolom
#         missing_text   = [c for c in TEXT_COLUMNS   if c not in df.columns]
#         missing_likert = [c for c in LIKERT_COLUMNS  if c not in df.columns]
#         missing_labels = [c for c in ALL_LABELS      if c not in df.columns]

#         if missing_text or missing_likert or missing_labels:
#             st.warning("⚠️ Beberapa kolom tidak ditemukan:")
#             if missing_text:
#                 st.markdown(f"**Kolom teks tidak ada:** {', '.join(missing_text)}")
#             if missing_likert:
#                 st.markdown(f"**Kolom Likert tidak ada:** {len(missing_likert)} kolom")
#             if missing_labels:
#                 st.markdown(f"**Kolom label tidak ada:** {', '.join(missing_labels)}")

#         if st.button("✅ Simpan Dataset", type="primary"):
#             st.session_state.df_dataset  = df
#             st.session_state.hasil_eval  = None
#             st.success("✅ Dataset berhasil disimpan ke sesi!")
#             st.rerun()

#     except Exception as e:
#         st.error(f"❌ Gagal membaca file: {e}")

# # ---------------------------------------------------------------------------
# # Preview dataset yang sudah ada
# # ---------------------------------------------------------------------------
# if st.session_state.df_dataset is not None:
#     df = st.session_state.df_dataset
#     st.divider()

#     head_col, reset_col = st.columns([4, 1])
#     with head_col:
#         st.subheader(f"📊 Dataset Aktif  ({len(df)} responden)")
#     with reset_col:
#         if st.button("🗑️ Hapus Dataset", type="secondary"):
#             st.session_state.df_dataset = None
#             st.session_state.hasil_eval = None
#             st.rerun()

#     # Tab: preview, statistik, distribusi label
#     t1, t2, t3 = st.tabs(["👁️ Preview Data", "📈 Statistik Likert", "🏷️ Distribusi Label"])

#     with t1:
#         cols_show = [c for c in TEXT_COLUMNS + ALL_LABELS if c in df.columns]
#         st.dataframe(df[cols_show].head(20), use_container_width=True, height=320)

#     with t2:
#         likert_present = [c for c in LIKERT_COLUMNS if c in df.columns]
#         if likert_present:
#             stats = df[likert_present].describe().T.round(2)
#             stats.index = [c.strip()[:50] for c in stats.index]
#             st.dataframe(stats, use_container_width=True)
#         else:
#             st.info("Kolom Likert tidak ditemukan.")

#     with t3:
#         labels_present = [c for c in ALL_LABELS if c in df.columns]
#         if labels_present:
#             import plotly.express as px
#             color_map = {"positif": "#059669", "negatif": "#DC2626", "netral": "#D97706",
#                          "Positif": "#059669", "Negatif": "#DC2626", "Netral": "#D97706"}
#             for lbl in labels_present:
#                 vc = df[lbl].value_counts().reset_index()
#                 vc.columns = ["Sentimen", "Jumlah"]
#                 display_name = LABEL_DISPLAY.get(lbl, lbl)

#                 c_bar, c_nums = st.columns([3, 1])
#                 with c_bar:
#                     fig = px.bar(
#                         vc, x="Sentimen", y="Jumlah", color="Sentimen",
#                         color_discrete_map=color_map,
#                         title=display_name, height=200,
#                         text="Jumlah",
#                     )
#                     fig.update_layout(
#                         showlegend=False, margin=dict(t=40, b=10),
#                         plot_bgcolor="white", paper_bgcolor="white",
#                     )
#                     fig.update_traces(textposition="outside")
#                     st.plotly_chart(fig, use_container_width=True)
#                 with c_nums:
#                     st.markdown(f"**{display_name}**")
#                     for _, row in vc.iterrows():
#                         pct = row["Jumlah"] / len(df) * 100
#                         st.markdown(
#                             f"<div style='display:flex;justify-content:space-between;"
#                             f"border-bottom:1px solid #F1F5F9;padding:6px 0'>"
#                             f"<span>{row['Sentimen']}</span>"
#                             f"<span><b>{row['Jumlah']}</b> <span style='color:#94A3B8;font-size:.8rem'>({pct:.0f}%)</span></span>"
#                             f"</div>",
#                             unsafe_allow_html=True,
#                         )
#                 st.markdown("---")
#         else:
#             st.info("Kolom label tidak ditemukan di dataset.")

#     # Download
#     st.divider()
#     csv_out = df.to_csv(index=False).encode("utf-8")
#     st.download_button("⬇️ Export Dataset (CSV)", csv_out, "dataset.csv", "text/csv")