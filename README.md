# Classical Computer Vision for Retail Monitoring
## A Dual-Mode Dashboard for Tracking & Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

This repository contains a professional computer vision project developed for a master's level course. The application demonstrates the use of classical CV methodologies (SIFT and HOG+SVM) for real-time object tracking and retail shelf monitoring.

> [!TIP]
> **Live Demo Ready**: This repository includes a `demo_data/` folder (~10MB) allowing the app to run immediately on **Streamlit Cloud** without downloading the full 625MB dataset.

### 🎯 Key Features
- **Dual-Mode Interface**: 
  - **OTB Tracking**: Benchmarking SIFT matching against the Object Tracking Benchmark (OTB).
  - **Retail Detection**: Multi-product detection in cluttered shelf environments using the Grozi-120 dataset.
- **Robust Methodology**:
  - **SIFT Matching**: Rotation and scale invariant local features with RANSAC homography.
  - **HOG+SVM**: Structural shape-based classification for textureless objects.
  - **Temporal Smoothing**: Alpha-beta filtering for stable bounding boxes.
  - **Occlusion Recovery**: Heuristic-based prediction during short-term object loss.
- **Interactive Metrics**: Real-time FPS monitoring and product visibility logs.

---

### 🚀 Getting Started

#### 1. Prerequisites
- Python 3.8+
- [Optional] Virtual environment recommended.

#### 2. Installation
```bash
git clone https://github.com/aashrithdsba/CV_Project.git
cd CV_Project
pip install -r requirements.txt
```

#### 3. Data Setup
There are two ways to run this project:

- **Option A: Demo Mode (Instant)**: The repository includes a `demo_data/` folder with a subset of the OTB and Grozi datasets. The app will automatically detect this and run in Demo Mode (perfect for Streamlit Cloud).
- **Option B: Full Dataset (Local)**: To access the full 625MB dataset, download the files from the sources below and place them in a `data/` directory:
    - **OTB Dataset**: [OTB-100 Mirror](https://github.com/prosti221/OTB-dataset) (Extract sequences into `data/OTB-dataset/sequences/`)
    - **Grozi-120 Dataset**: [Official UCSD Link](http://grozi.calit2.net/grozi.html)
        - Extract `inVitro` into `data/inVitro/inVitro/`
        - Extract `inSitu` into `data/inSitu/inSitu/`
        - Extract `videos` into `data/videos/video/`

#### 4. Running the App
```bash
streamlit run app.py
```

---

### 📂 Project Structure
- `app.py`: Main Streamlit dashboard and UI logic.
- `cv_pipeline.py`: SIFT-based single-object tracking for OTB sequences.
- `grozi_pipeline.py`: Multi-target SIFT detection for Grozi-120 retail footage.
- `train_hog_svm.py`: Training script for shape-based product classification.
- `data/`: Local copies of OTB and Grozi-120 datasets.
- `models/`: Pre-trained SVM XML models for product identification.

---

### 📊 Technical Report
A comprehensive academic report is included in the root directory:
- [PDF/LaTeX Version](CV_Master_Project_Submission.tex)
- [Markdown Summary](CV_Master_Project_Submission.md)

---

### 🛠 Methodologies Used
- **SIFT**: Scale-Invariant Feature Transform (Lowe's Ratio Test = 0.75).
- **HOG**: Histogram of Oriented Gradients (9 bins, 8x8 cells).
- **SVM**: Linear Support Vector Machine ($C=2.67$).
- **RANSAC**: Robust homography estimation (threshold = 5.0).
