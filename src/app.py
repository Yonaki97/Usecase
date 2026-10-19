import streamlit as st
import tempfile
import os

from pipeline import run_assessment

# ===========================================s=================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Asesmen Kecukupan Juknis",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📄 Asesmen Kecukupan Juknis")

st.write(
    "Upload dokumen Juknis untuk dilakukan "
    "asesmen kecukupan berdasarkan regulasi."
)


# ============================================================
# UPLOAD FILE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload dokumen Juknis",
    type=["pdf"]
)

enable_llm_judge = st.checkbox(
    "Jalankan Qwen sebagai LLM Judge (evaluasi sekunder)",
    value=False,
    help=(
        "Qwen memeriksa apakah output assessment didukung evidence yang sama. "
        "Ini tidak mengubah assessment score dan dijalankan lokal melalui Ollama."
    )
)


# ============================================================
# ASSESSMENT
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Dokumen dipilih: {uploaded_file.name}"
    )

    if st.button(
        "Mulai Assessment",
        type="primary"
    ):

        # Progress bar
        progress_bar = st.progress(0)

        # Status text
        status_text = st.empty()


        # ====================================================
        # PROGRESS CALLBACK
        # ====================================================

        def update_progress(
            value,
            message
        ):

            progress_bar.progress(
                int(value * 100)
            )

            status_text.info(
                message
            )


        # ====================================================
        # SAVE TEMPORARY PDF
        # ====================================================

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_pdf_path = temp_file.name


        # ====================================================
        # RUN ASSESSMENT
        # ====================================================

        try:

            with st.spinner(
                "Sistem sedang melakukan assessment..."
            ):

                result = run_assessment(

                    pdf_path=
                        temp_pdf_path,

                    file_name=
                        uploaded_file.name,

                    progress_callback=
                        update_progress,

                    enable_llm_judge=
                        enable_llm_judge
                )


            # ====================================================
            # FINISH
            # ====================================================

            progress_bar.progress(100)

            status_text.success(
                "Assessment selesai!"
            )


            # ====================================================
            # RESULT
            # ====================================================

            st.divider()

            st.header(
                "Hasil Assessment"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Assessment Score",
                    f"{result['percentage']:.2f}%"
                )


            with col2:

                st.metric(
                    "Classification",
                    result["classification"]
                )


            with col3:

                st.metric(
                    "Dokumen",
                    result["document_name"]
                )


            # ====================================================
            # DETAIL SUBCATEGORY
            # ====================================================

            st.divider()

            st.subheader(
                "Detail Assessment"
            )


            assessment_results = (
                result["assessment_results"]
            )


            for kode, item in (
                assessment_results.items()
            ):

                with st.expander(
                    f"{kode} — "
                    f"{item.get('nama', '')}"
                ):

                    st.write(
                        "### Status"
                    )

                    st.write(
                        item.get(
                            "status",
                            "-"
                        )
                    )


                    st.write(
                        "### Assessment Level"
                    )

                    st.write(
                        item.get(
                            "assessment_level",
                            "-"
                        )
                    )


                    st.write(
                        "### Alasan"
                    )

                    st.write(
                        item.get(
                            "alasan",
                            "-"
                        )
                    )


                    st.write(
                        "### Evidence Juknis"
                    )

                    evidence_juknis = (
                        item.get(
                            "evidence_juknis",
                            []
                        )
                    )

                    for evidence in (
                        evidence_juknis
                    ):

                        st.info(
                            evidence.get(
                                "text",
                                ""
                            )
                        )


                    st.write(
                        "### Evidence Regulasi"
                    )

                    evidence_regulasi = (
                        item.get(
                            "evidence_regulasi",
                            []
                        )
                    )

                    for evidence in (
                        evidence_regulasi
                    ):

                        st.warning(
                            evidence.get(
                                "text",
                                ""
                            )
                        )

            judge_results = result.get("llm_judge_results")
            if judge_results:
                st.divider()
                st.subheader("Evaluasi Qwen sebagai LLM Judge")
                st.caption(
                    "Evaluasi sekunder: tidak mengubah skor assessment Qwen."
                )
                for kode, judgment in judge_results.items():
                    with st.expander(f"Judge — {kode}"):
                        st.json(judgment)


        except Exception as e:

            st.error(
                f"Terjadi kesalahan: {e}"
            )


        finally:

            # Hapus file temporary
            if os.path.exists(
                temp_pdf_path
            ):

                os.remove(
                    temp_pdf_path
                )
