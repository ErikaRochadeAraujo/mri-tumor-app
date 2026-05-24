# 🧠 Análise de Tumor em Ressonância Magnética Cerebral

Aplicação web para segmentação automática de tumores cerebrais em imagens de Ressonância Magnética (RM), utilizando um modelo de deep learning **3D U-Net** treinado com o dataset **BraTS2020**.

🔗 **[Acesse o app](https://mri-tumor-brain.streamlit.app/)**

---

## Sobre o modelo

O modelo é uma **3D U-Net** treinada com função de perda combinada: **Entropia Cruzada Categórica + Soft Dice Loss**. Ele recebe duas modalidades de RM como entrada e produz mapas de probabilidade para cada classe tumoral.

**Modalidades de entrada:**
- **FLAIR** — destaca edema e infiltração tumoral
- **T1CE** — realce por contraste, evidencia tumor ativo

**Classes detectadas:**

| Cor | Classe | Descrição |
|-----|--------|-----------|
| ⬛ Cinza | Sem tumor | Tecido cerebral saudável |
| 🟡 Amarelo | Tumor sem realce | Núcleo necrótico / não realçado |
| 🟢 Verde | Edema | Zona peritumoral |
| 🔴 Vermelho | Tumor com realce | Tumor ativo com realce por contraste |

---

## Como usar

1. Acesse o app em [mri-tumor-brain.streamlit.app](https://mri-tumor-brain.streamlit.app/)
2. Selecione o formato dos arquivos: **NIfTI** (`.nii` / `.nii.gz`) ou **DICOM** (`.dcm`)
3. Envie os arquivos **FLAIR** e **T1CE**
4. Opcionalmente, envie o arquivo de **segmentação** (rótulo verdade) para comparação
5. Clique em **Analisar** e aguarde o resultado

### Arquivos de exemplo

Arquivos de exemplo para teste estão disponíveis na pasta [Imagens de IRM](https://github.com/ErikaRochadeAraujo/mri-tumor-app/tree/main/Imagens%20de%20IRM):

- **BraTS20_Training_001, 006, 009** — casos NIfTI do dataset BraTS2020 (FLAIR, T1CE, segmentação)
- **dcm / dcm2** — séries DICOM de exemplo (192 arquivos cada)
- **Nifti** — volumes UPENN-GBM (FLAIR e T1GD)

---

## Tecnologias

- [Streamlit](https://streamlit.io/) — interface web
- [ONNX Runtime](https://onnxruntime.ai/) — inferência do modelo
- [nibabel](https://nipy.org/nibabel/) — leitura de arquivos NIfTI
- [pydicom](https://pydicom.github.io/) + [dicom2nifti](https://github.com/icometrix/dicom2nifti) — suporte a DICOM
- [OpenCV](https://opencv.org/) + [Matplotlib](https://matplotlib.org/) — visualização

---

## Dataset

Treinado com o dataset público [BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) (Brain Tumor Segmentation Challenge 2020).
