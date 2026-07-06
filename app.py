import streamlit as st
import numpy as np

from model_utils import (
    load_model_cached, run_inference, preprocess_volumes, DEFAULT_MODEL_PATH,
    group_dicom_series, detect_modality, convert_bytes_list_to_nifti,
)
from visualization import show_slice_comparison, show_diagnostic_summary, load_seg_volume, show_3d_view

st.set_page_config(
    page_title="Análise de Tumor - RM Cerebral",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Análise de Tumor em Ressonância Magnética Cerebral")
st.markdown(
    "Modelo **3D U-Net** treinado com perda combinada "
    "**Entropia Cruzada Categórica e Soft Dice Loss** ulitilizando imagens BraTS2020"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configurações")
    model_path = st.text_input("Caminho do modelo (.onnx)", value=DEFAULT_MODEL_PATH)
    st.markdown("---")
    st.markdown(
        "**Classes detectadas:**\n"
        "- ⬛ Sem tumor\n"
        "- 🟡 Tumor sem realce\n"
        "- 🟢 Edema\n"
        "- 🔴 Tumor com realce"
    )
    st.info("O modelo usa **2 modalidades**: FLAIR e T1CE.")

st.markdown(
    "Para usar e testar você pode usar os arquivos que deixo disponíveis aqui: "
    "[Exemplos](https://github.com/ErikaRochadeAraujo/mri-tumor-app/tree/main/Imagens%20de%20IRM)"
)

# ── Seletor de formato ────────────────────────────────────────────────────────
st.subheader("Formato das imagens")
fmt = st.radio(
    "Selecione o formato dos arquivos de entrada:",
    ["NIfTI (.nii / .nii.gz)", "DICOM (.dcm)"],
    horizontal=True,
)

flair_bytes = None
t1ce_bytes  = None
seg_file    = None

# ── Upload NIfTI ──────────────────────────────────────────────────────────────
if fmt == "NIfTI (.nii / .nii.gz)":
    st.subheader("Enviar imagens NIfTI")
    col1, col2, col3 = st.columns(3)
    with col1:
        flair_file = st.file_uploader("FLAIR (.nii)", type=["nii", "gz"])
    with col2:
        t1ce_file = st.file_uploader("T1CE (.nii)", type=["nii", "gz"])
    with col3:
        seg_file = st.file_uploader("Segmentação / Rótulo verdade (.nii) — opcional", type=["nii", "gz"])

    if flair_file:
        flair_bytes = flair_file.read()
    if t1ce_file:
        t1ce_bytes = t1ce_file.read()

# ── Upload DICOM ───────────────────────────────────────────────────────────────
else:
    st.subheader("Enviar arquivos DICOM")
    st.info("Envie **todos os arquivos .dcm do exame** de uma vez. O app detectará automaticamente as sequências.")

    all_dcm = st.file_uploader(
        "Selecione todos os arquivos DICOM (.dcm)",
        type=["dcm"],
        accept_multiple_files=True,
    )

    if all_dcm:
        with st.spinner("Lendo metadados das séries..."):
            series = group_dicom_series(all_dcm)

        n_series = len(series)
        if n_series == 0:
            st.error("Nenhuma série DICOM válida encontrada.")
        else:
            st.success(f"{n_series} série(s) detectada(s).")

            # Detecta modality automaticamente
            uids = list(series.keys())
            auto = {uid: detect_modality(series[uid]["description"]) for uid in uids}

            def default_index(modality):
                for i, uid in enumerate(uids):
                    if auto[uid] == modality:
                        return i
                return 0

            def series_label(uid):
                info = series[uid]
                mod  = auto[uid]
                tag  = f" [detectado: {mod}]" if mod != "Não identificado" else ""
                return f"{info['description']} ({len(info['bytes_list'])} arquivos){tag}"

            st.markdown("#### Confirme qual série corresponde a cada modalidade:")
            col1, col2 = st.columns(2)
            with col1:
                flair_uid = st.selectbox(
                    "Série FLAIR",
                    options=uids,
                    format_func=series_label,
                    index=default_index("FLAIR"),
                )
            with col2:
                t1ce_uid = st.selectbox(
                    "Série T1CE",
                    options=uids,
                    format_func=series_label,
                    index=default_index("T1CE"),
                )

            if flair_uid == t1ce_uid:
                st.warning("FLAIR e T1CE estão apontando para a mesma série. Verifique a seleção.")
            elif st.button("Converter DICOM → NIfTI", width="stretch"):
                with st.spinner("Convertendo FLAIR..."):
                    try:
                        flair_bytes = convert_bytes_list_to_nifti(series[flair_uid]["bytes_list"])
                        st.session_state["flair_bytes"] = flair_bytes
                    except Exception as e:
                        st.error(f"Erro ao converter FLAIR: {e}")
                        st.stop()
                with st.spinner("Convertendo T1CE..."):
                    try:
                        t1ce_bytes = convert_bytes_list_to_nifti(series[t1ce_uid]["bytes_list"])
                        st.session_state["t1ce_bytes"] = t1ce_bytes
                    except Exception as e:
                        st.error(f"Erro ao converter T1CE: {e}")
                        st.stop()
                st.success("Conversão concluída!")

    # Recupera bytes convertidos se já existirem na sessão
    if flair_bytes is None:
        flair_bytes = st.session_state.get("flair_bytes")
    if t1ce_bytes is None:
        t1ce_bytes = st.session_state.get("t1ce_bytes")

# ── Verificação e botão de análise ────────────────────────────────────────────
st.markdown("---")
ready = flair_bytes and t1ce_bytes and model_path
if not ready:
    missing = []
    if not model_path:  missing.append("caminho do modelo")
    if not flair_bytes: missing.append("FLAIR")
    if not t1ce_bytes:  missing.append("T1CE")
    st.info(f"Aguardando: {', '.join(missing)}")

if ready and st.button("Analisar", type="primary", width="stretch"):
    with st.spinner("Carregando modelo e pré-processando imagens..."):
        try:
            model = load_model_cached(model_path)
            X = preprocess_volumes(flair_bytes, t1ce_bytes)
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            st.stop()

    with st.spinner("Executando inferência..."):
        try:
            prediction = run_inference(model, X)
        except Exception as e:
            st.error(f"Erro na inferência: {e}")
            st.stop()

    st.session_state["X"] = X
    st.session_state["prediction"] = prediction
    st.session_state["seg"] = load_seg_volume(seg_file.read()) if seg_file else None

# ── Resultados ────────────────────────────────────────────────────────────────
if "prediction" in st.session_state:
    X          = st.session_state["X"]
    prediction = st.session_state["prediction"]
    seg_vol    = st.session_state.get("seg")
    flair_vol  = X[:, :, :, 0]

    st.success("Análise concluída!")
    st.subheader("Visualização")

    slice_idx = st.slider(
        "Selecione o slice axial",
        min_value=0, max_value=X.shape[0] - 1,
        value=X.shape[0] // 2,
    )
    show_slice_comparison(flair_vol, prediction, slice_idx, seg_vol)

    show_diagnostic_summary(prediction, flair_vol)

    st.markdown("---")
    st.subheader("Visualização 3D interativa")
    show_3d = st.checkbox("Gerar visualização 3D (nuvem de pontos do cérebro e do tumor)")
    if show_3d:
        density_options = {
            "Leve (mais rápido)": 15,
            "Média": 8,
            "Densa": 5,
            "Muito densa (mais lento)": 3,
        }
        density_label = st.select_slider(
            "Densidade da nuvem de pontos",
            options=list(density_options.keys()),
            value="Média",
            help="Mais densa = mais pontos e mais detalhe, porém mais lento para girar/ampliar.",
        )
        downsample = density_options[density_label]
        st.caption("Arraste para girar, use o scroll para zoom. Clique na legenda para ligar/desligar cada classe.")
        with st.spinner("Montando nuvem de pontos 3D..."):
            show_3d_view(flair_vol, prediction, seg_vol, mri_downsample=downsample)
