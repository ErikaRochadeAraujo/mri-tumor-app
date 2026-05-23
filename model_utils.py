import io
import tempfile
import os
import numpy as np
import nibabel as nib
import cv2
import streamlit as st
import tf_keras as keras

VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE = 128

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_x80_dcs65copy42.h5")


@st.cache_resource(show_spinner="Carregando modelo...")
def load_model_cached(model_path: str):
    # compile=False evita precisar das funções de perda/métricas customizadas
    return keras.models.load_model(model_path, compile=False)


def _load_nifti_from_bytes(nii_bytes: bytes) -> np.ndarray:
    """Carrega NIfTI de bytes, detectando automaticamente se está comprimido (.nii.gz)."""
    import gzip
    if nii_bytes[:2] == b'\x1f\x8b':   # magic bytes do gzip
        nii_bytes = gzip.decompress(nii_bytes)
    img = nib.Nifti1Image.from_bytes(nii_bytes)
    return np.asarray(img.dataobj).copy()


_FLAIR_KEYWORDS = ["flair", "t2_flair", "t2-flair", "t2flair"]
_T1CE_KEYWORDS  = ["t1ce", "t1+c", "t1c", "t1_gd", "t1gd", "t1w_ce",
                   "t1wce", "gd", "contrast", "ce", "post", "realce"]


def group_dicom_series(dicom_files: list) -> dict:
    """Lê metadados de cada arquivo e agrupa por série.

    Retorna {series_uid: {'description': str, 'bytes_list': [bytes, ...]}}
    """
    import pydicom

    series: dict = {}
    for f in dicom_files:
        data = f.read()
        try:
            ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
            uid  = str(getattr(ds, "SeriesInstanceUID", f"uid_{len(series)}"))
            desc = (str(getattr(ds, "SeriesDescription", "")).strip()
                    or str(getattr(ds, "ProtocolName", "")).strip()
                    or "Série desconhecida")
            if uid not in series:
                series[uid] = {"description": desc, "bytes_list": []}
            series[uid]["bytes_list"].append(data)
        except Exception:
            pass
    return series


def detect_modality(description: str) -> str:
    """Detecta 'FLAIR', 'T1CE' ou 'Não identificado' pelo nome da série."""
    d = description.lower()
    if any(k in d for k in _FLAIR_KEYWORDS):
        return "FLAIR"
    if any(k in d for k in _T1CE_KEYWORDS):
        return "T1CE"
    return "Não identificado"


def convert_bytes_list_to_nifti(bytes_list: list) -> bytes:
    """Converte lista de bytes DICOM em bytes NIfTI."""
    import dicom2nifti

    with tempfile.TemporaryDirectory() as dicom_dir:
        for i, data in enumerate(bytes_list):
            with open(os.path.join(dicom_dir, f"slice_{i:04d}.dcm"), "wb") as out:
                out.write(data)
        with tempfile.TemporaryDirectory() as nifti_dir:
            dicom2nifti.convert_directory(dicom_dir, nifti_dir,
                                          compression=False, reorient=True)
            nifti_files = [
                fn for fn in os.listdir(nifti_dir)
                if fn.endswith(".nii") or fn.endswith(".nii.gz")
            ]
            if not nifti_files:
                raise ValueError("Nenhum NIfTI gerado — verifique se os arquivos são DICOM válidos.")
            with open(os.path.join(nifti_dir, nifti_files[0]), "rb") as f:
                return f.read()


def preprocess_volumes(flair_bytes: bytes, t1ce_bytes: bytes) -> np.ndarray:
    """
    Retorna array (n, IMG_SIZE, IMG_SIZE, 2) pronto para inferência.
    Canal 0 = FLAIR, Canal 1 = T1CE.
    Adapta-se ao número real de slices: BraTS (155) ou DICOM clínico (qualquer tamanho).
    """
    flair = _load_nifti_from_bytes(flair_bytes)
    t1ce  = _load_nifti_from_bytes(t1ce_bytes)

    n_slices = min(flair.shape[2], t1ce.shape[2])
    # Mantém proporção do BraTS: começa em ~14% do volume
    start = max(0, int(n_slices * (VOLUME_START_AT / 155)))
    end   = min(n_slices, start + VOLUME_SLICES)
    n     = end - start

    X = np.zeros((n, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)
    for j in range(n):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + start], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(t1ce[:, :, j + start],  (IMG_SIZE, IMG_SIZE))

    X = X / np.max(X)
    return X
