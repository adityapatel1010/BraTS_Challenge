# BraTS 2024 Brain Tumor Segmentation

This project implements and compares **deep learning models** for automated brain tumor segmentation on the **BraTS-2024 dataset**.  
We explore **CNN-based** (nnUNet, SegResNet) and **Transformer-based** (SwinUNETR) architectures to segment different tumor sub-regions from MRI scans.  

---

## Models Implemented
- **nnUNet**
  - Self-configuring UNet-based architecture.
  - Supports **2D and 3D convolutions**.
  - Features: deep supervision, ensembling, extensive data augmentation.
  - Hyperparameter details stored in `nnUNet_hyperparameters`.

- **SegResNet**
  - Residual network tailored for medical image segmentation.
  - Efficient in capturing hierarchical spatial features.
  - Implemented in `SegResNet_Train.ipynb`.

- **SwinUNETR**
  - Transformer-based model using shifted windows for global context.
  - Strong performance on volumetric medical imaging.
  - Implemented in `swinunetr_train.py`.

---

## Tumor Sub-Regions Segmented
The models were trained to predict the following **labels**:
- **WT** – Whole Tumor  
- **TC** – Tumor Core  
- **ET** – Enhancing Tumor  
- **RC** – Resection Cavity  

---

## Repository Contents
- **Training Notebooks & Scripts**
  - `nnUNet_train.ipynb` – Training pipeline for nnUNet.
  - `SegResNet_Train.ipynb` – Training pipeline for SegResNet.
  - `swinunetr_train.py` – Training pipeline for SwinUNETR.
- **Results & Visualization**
  - `nnUNet_results.ipynb` – Qualitative and quantitative results.
  - `Validation/` – Fold-wise loss/accuracy curves (`progress_fold0.png` → `progress_fold4.png`).
- **Configurations**
  - `nnUNet_hyperparameters/` – Hyperparameter details for nnUNet.
  - `dataset.json`, `dataset_fingerprint.json`, `plans.json` – Dataset and preprocessing configs.

---

## Results & Visualizations
- Training and validation tracked across **5 folds** with accuracy and loss curves.
- Results visualized in notebooks (`nnUNet_results.ipynb`) and progress plots:
  - `progress_fold0.png` → `progress_fold4.png`.
---

## Objectives & Insights
- Train and evaluate **multiple segmentation models** on BraTS-2024.
- Compare CNN vs Transformer-based approaches.
- Use **cross-validation** to ensure robust evaluation.
- Segment clinically relevant tumor regions (**WT, TC, ET, RC**) for potential use in diagnosis and treatment planning.

---

## About This Project
Brain tumor segmentation on BraTS-2024 dataset using **nnUNet, SegResNet, and SwinUNETR**. Models trained and validated on MRI scans to segment **WT, TC, ET, and RC** with fold-wise performance tracking and visualizations.

---

## Topics
`Deep Learning` `Medical Imaging` `Brain Tumor Segmentation` `MRI` `BraTS-2024` `nnUNet` `SegResNet` `SwinUNETR` `Computer Vision` `PyTorch`

