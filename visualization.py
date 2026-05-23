import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st

SEGMENT_CLASSES = {
    0: "Sem tumor",
    1: "Tumor sem realce",
    2: "Edema",
    3: "Tumor com realce",
}

# Cores únicas por classe — usadas em overlays E gráficos
CLASS_COLORS_HEX = {
    0: "#6c757d",  # cinza
    1: "#ffc107",  # amarelo
    2: "#28a745",  # verde
    3: "#dc3545",  # vermelho
}

def _hex_to_rgba(hex_color: str, prob_map: np.ndarray) -> np.ndarray:
    """Converte cor hex + mapa de probabilidade em overlay RGBA."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    h, w = prob_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[:, :, 0] = r
    overlay[:, :, 1] = g
    overlay[:, :, 2] = b
    overlay[:, :, 3] = np.clip(prob_map * 0.7, 0, 1)
    return overlay


def _make_argmax_overlay(class_map: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Colore cada pixel pela sua classe dominante; transparente onde classe == 0."""
    h, w = class_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for cls_id in [1, 2, 3]:
        hex_c = CLASS_COLORS_HEX[cls_id]
        r = int(hex_c[1:3], 16) / 255
        g = int(hex_c[3:5], 16) / 255
        b = int(hex_c[5:7], 16) / 255
        mask = class_map == cls_id
        overlay[mask, 0] = r
        overlay[mask, 1] = g
        overlay[mask, 2] = b
        overlay[mask, 3] = alpha
    return overlay

DISPLAY_SIZE = 1024
DPI = 250
VOLUME_START_AT = 22


def _upscale(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img.astype(np.float32), (DISPLAY_SIZE, DISPLAY_SIZE),
                      interpolation=cv2.INTER_LANCZOS4)


def load_seg_volume(seg_bytes: bytes) -> np.ndarray:
    """Carrega o volume de segmentação e mapeia valor 4 → 3 (padrão BraTS)."""
    import gzip
    if seg_bytes[:2] == b'\x1f\x8b':
        seg_bytes = gzip.decompress(seg_bytes)
    img = nib.Nifti1Image.from_bytes(seg_bytes)
    vol = np.asarray(img.dataobj).copy()
    vol[vol == 4] = 3   # BraTS usa 4 para tumor com realce
    return vol          # shape original (240, 240, 155)


def _get_seg_slice(seg_vol: np.ndarray, slice_idx: int) -> np.ndarray:
    """Extrai e redimensiona o slice do rótulo verdade."""
    raw = seg_vol[:, :, slice_idx + VOLUME_START_AT]
    return cv2.resize(raw.astype(np.float32), (DISPLAY_SIZE, DISPLAY_SIZE),
                      interpolation=cv2.INTER_NEAREST)


def show_slice_comparison(flair_vol: np.ndarray, pred: np.ndarray,
                          slice_idx: int, seg_vol: np.ndarray | None = None):
    orig      = _upscale(flair_vol[slice_idx])
    core      = _upscale(pred[slice_idx, :, :, 1])
    edema     = _upscale(pred[slice_idx, :, :, 2])
    enhancing = _upscale(pred[slice_idx, :, :, 3])

    has_gt = seg_vol is not None
    n_cols = 3
    n_rows = 2 if has_gt else 1

    fig, axarr = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows), dpi=DPI)

    # Garante sempre array 2D de eixos
    if n_rows == 1:
        axarr = axarr[np.newaxis, :]

    # Fundo FLAIR em todos os painéis
    for i in range(n_rows):
        for j in range(n_cols):
            axarr[i, j].imshow(orig, cmap="gray", interpolation="lanczos")
            axarr[i, j].axis("off")

    # ── 1ª linha: previsões por classe ────────────────────────────────────────
    axarr[0, 0].imshow(_hex_to_rgba(CLASS_COLORS_HEX[1], core), interpolation="nearest")
    axarr[0, 0].set_title(f"{SEGMENT_CLASSES[1]} previsto")

    axarr[0, 1].imshow(_hex_to_rgba(CLASS_COLORS_HEX[2], edema), interpolation="nearest")
    axarr[0, 1].set_title(f"{SEGMENT_CLASSES[2]} previsto")

    axarr[0, 2].imshow(_hex_to_rgba(CLASS_COLORS_HEX[3], enhancing), interpolation="nearest")
    axarr[0, 2].set_title(f"{SEGMENT_CLASSES[3]} previsto")

    # ── 2ª linha: FLAIR + rótulo verdade (se disponível) ─────────────────────
    if has_gt:
        axarr[1, 0].set_title("Imagem Original FLAIR")

        # Ground truth: argmax por pixel com as mesmas cores das previsões
        gt_slice = _get_seg_slice(seg_vol, slice_idx).astype(int)
        axarr[1, 1].imshow(_make_argmax_overlay(gt_slice), interpolation="nearest")
        axarr[1, 1].set_title("Rótulo verdade (Ground Truth)")

        # Todas as classes: classe dominante entre 1-3, alpha = probabilidade dessa classe
        # (mesmo comportamento dos painéis individuais — bordas idênticas)
        probs_tumor = pred[slice_idx, :, :, 1:4]           # (128, 128, 3)
        dominant_cls = np.argmax(probs_tumor, axis=-1) + 1  # 1, 2 ou 3
        dominant_prob = np.max(probs_tumor, axis=-1)        # (128, 128)

        dominant_cls_up = cv2.resize(dominant_cls.astype(np.float32),
                                     (DISPLAY_SIZE, DISPLAY_SIZE),
                                     interpolation=cv2.INTER_NEAREST).astype(int)
        dominant_prob_up = _upscale(dominant_prob)

        combined = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 4), dtype=np.float32)
        for cls_id in [1, 2, 3]:
            hex_c = CLASS_COLORS_HEX[cls_id]
            r = int(hex_c[1:3], 16) / 255
            g = int(hex_c[3:5], 16) / 255
            b = int(hex_c[5:7], 16) / 255
            m = dominant_cls_up == cls_id
            combined[m, 0] = r
            combined[m, 1] = g
            combined[m, 2] = b
            combined[m, 3] = np.clip(dominant_prob_up[m] * 0.7, 0, 1)

        axarr[1, 2].imshow(combined, interpolation="nearest")
        axarr[1, 2].set_title("Todas as classes previstas")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)



def show_diagnostic_summary(pred: np.ndarray, flair_vol: np.ndarray):
    mask = np.argmax(pred, axis=-1)          # (n_slices, 128, 128)

    # Usa apenas voxels com sinal cerebral (exclui fundo preto)
    brain_mask = flair_vol > 0.01            # mesma shape de mask
    total = int(np.sum(brain_mask))
    if total == 0:
        total = mask.size                    # fallback se tudo for zero

    mask_flat  = mask[brain_mask].flatten()
    tumor_voxels = int(np.sum(mask_flat > 0))
    pct_tumor = 100 * tumor_voxels / total

    st.markdown("---")
    st.subheader("Resumo Diagnóstico")

    if tumor_voxels == 0:
        st.success("Nenhuma região tumoral detectada no volume analisado.")
        return

    st.warning(f"Regiões tumorais detectadas em **{pct_tumor:.2f}%** do volume cerebral.")
    for cls_id in [3, 1, 2]:
        n = int(np.sum(mask_flat == cls_id))
        if n > 0:
            st.markdown(
                f"<span style='color:{CLASS_COLORS_HEX[cls_id]}; font-weight:bold'>■</span> "
                f"**{SEGMENT_CLASSES[cls_id]}**: {n:,} voxels ({100*n/total:.2f}%)",
                unsafe_allow_html=True,
            )
