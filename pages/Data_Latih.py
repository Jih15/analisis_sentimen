import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Data Latih | SentimenTA", page_icon="📝", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "df_latih" not in st.session_state:
    st.session_state.df_latih = []
if "df_uji" not in st.session_state:
    st.session_state.df_uji = []
if "hasil" not in st.session_state:
    st.session_state.hasil = None

# ---------------------------------------------------------------------------
# Custom CSS (shared)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { background-color: #1E3A5F; }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    .page-title   { font-size:1.8rem; font-weight:700; color:#1E293B; }
    .section-card {
        background:#F8FAFC; border:1px solid #E2E8F0;
        border-radius:10px; padding:20px; margin-bottom:12px;
    }
    .label-pos { color:#059669; font-weight:600; }
    .label-neg { color:#DC2626; font-weight:600; }
    .label-net { color:#D97706; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="page-title">📝 Data Latih</p>', unsafe_allow_html=True)
st.caption(
    "Masukkan data umpan balik pengguna beserta label sentimen yang akan "
    "digunakan untuk melatih model machine learning."
)
st.divider()

# ---------------------------------------------------------------------------
# Tabs: Upload CSV  |  Input Manual
# ---------------------------------------------------------------------------
tab_upload, tab_manual = st.tabs(["📂 Upload CSV", "✍️ Input Manual"])

# ── Tab 1: Upload CSV ───────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        """<div class="section-card">
        <b>Format CSV yang diterima:</b><br>
        File harus memiliki dua kolom: <code>teks</code> (isi feedback) dan
        <code>label</code> (Positif / Negatif / Netral).
        </div>""",
        unsafe_allow_html=True,
    )

    # Download template
    template_df = pd.DataFrame(
        {
            "teks": [
                "Aplikasi ini sangat membantu saya menghindari lowongan palsu",
                "Tampilannya membingungkan dan sulit digunakan",
                "Fiturnya cukup lumayan tapi perlu dikembangkan lagi",
            ],
            "label": ["Positif", "Negatif", "Netral"],
        }
    )
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Template CSV",
        data=csv_template,
        file_name="template_data_latih.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload file CSV data latih", type=["csv"], key="upload_latih"
    )

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.columns = [c.strip().lower() for c in df_up.columns]

            if "teks" not in df_up.columns or "label" not in df_up.columns:
                st.error("❌ Kolom 'teks' dan 'label' harus ada di file CSV.")
            else:
                # Validasi label
                valid_labels = {"Positif", "Negatif", "Netral"}
                df_up["label"] = df_up["label"].str.strip().str.capitalize()
                invalid = df_up[~df_up["label"].isin(valid_labels)]

                if not invalid.empty:
                    st.warning(
                        f"⚠️ {len(invalid)} baris memiliki label tidak valid dan akan dilewati. "
                        f"Label yang diterima: Positif, Negatif, Netral."
                    )
                    df_up = df_up[df_up["label"].isin(valid_labels)]

                df_up = df_up[["teks", "label"]].dropna()

                c1, c2 = st.columns([2, 1])
                with c1:
                    st.dataframe(df_up, use_container_width=True, height=250)
                with c2:
                    st.metric("Total baris valid", len(df_up))
                    lc = df_up["label"].value_counts()
                    for lbl, cnt in lc.items():
                        st.metric(lbl, cnt)

                if st.button("✅ Simpan ke Data Latih", type="primary", key="save_csv_latih"):
                    new_rows = df_up.to_dict("records")
                    st.session_state.df_latih.extend(new_rows)
                    # Reset hasil karena data berubah
                    st.session_state.hasil = None
                    st.success(f"✅ {len(new_rows)} data berhasil ditambahkan!")
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

# ── Tab 2: Input Manual ─────────────────────────────────────────────────────
with tab_manual:
    with st.form("form_manual_latih", clear_on_submit=True):
        teks_input = st.text_area(
            "Teks umpan balik pengguna",
            placeholder="Contoh: Aplikasi ini sangat membantu saya...",
            height=100,
        )
        label_input = st.selectbox(
            "Label sentimen",
            options=["Positif", "Negatif", "Netral"],
        )
        submitted = st.form_submit_button("➕ Tambah Data", type="primary")

    if submitted:
        if teks_input.strip():
            st.session_state.df_latih.append(
                {"teks": teks_input.strip(), "label": label_input}
            )
            st.session_state.hasil = None
            st.success("✅ Data berhasil ditambahkan!")
            st.rerun()
        else:
            st.warning("⚠️ Teks tidak boleh kosong.")

# ---------------------------------------------------------------------------
# Tabel data latih saat ini
# ---------------------------------------------------------------------------
st.divider()
n = len(st.session_state.df_latih)
header_col, reset_col = st.columns([3, 1])
with header_col:
    st.subheader(f"📋 Data Latih Saat Ini  ({n} data)")
with reset_col:
    if n > 0:
        if st.button("🗑️ Reset Semua", type="secondary"):
            st.session_state.df_latih = []
            st.session_state.hasil = None
            st.rerun()

if n == 0:
    st.info("Belum ada data latih. Silakan upload CSV atau input manual di atas.")
else:
    df_display = pd.DataFrame(st.session_state.df_latih)

    # Color-coded label column
    def color_label(val):
        colors = {
            "Positif": "color:#059669;font-weight:600",
            "Negatif": "color:#DC2626;font-weight:600",
            "Netral":  "color:#D97706;font-weight:600",
        }
        return colors.get(val, "")

    # Show table
    st.dataframe(
        df_display,
        use_container_width=True,
        height=min(400, 50 + n * 35),
        column_config={
            "teks":  st.column_config.TextColumn("Teks Umpan Balik", width="large"),
            "label": st.column_config.TextColumn("Label", width="small"),
        },
    )

    # Label summary
    lbl_counts = df_display["label"].value_counts()
    cols = st.columns(len(lbl_counts))
    color_map = {"Positif": "🟢", "Negatif": "🔴", "Netral": "🟡"}
    for col, (lbl, cnt) in zip(cols, lbl_counts.items()):
        with col:
            st.metric(f"{color_map.get(lbl, '')} {lbl}", cnt)

    # Download data latih
    csv_out = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Data Latih (CSV)",
        data=csv_out,
        file_name="data_latih.csv",
        mime="text/csv",
    )

    # Delete individual row
    with st.expander("🗑️ Hapus data tertentu"):
        idx_del = st.number_input(
            "Nomor baris yang ingin dihapus (mulai dari 1)",
            min_value=1, max_value=n, step=1, value=1,
        )
        if st.button("Hapus baris ini"):
            st.session_state.df_latih.pop(int(idx_del) - 1)
            st.session_state.hasil = None
            st.success("✅ Baris berhasil dihapus.")
            st.rerun()
