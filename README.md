# Deep Learning Footprint Classifier  
**Kaggle competition project — binary image classification (predicting “SEX” from footprint images)**

> This repository is intentionally lightweight: the full implementation lives in a single notebook that you can open and run via a **Colab badge**.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Run in Colab](#run-in-colab)
  - [Run on Kaggle](#run-on-kaggle)
- [Problem definition](#problem-definition)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Baseline model](#baseline-model)
- [State-of-the-art comparison](#state-of-the-art-comparison)
- [10 systematic experiments](#10-systematic-experiments)
- [Explainability with Grad-CAM](#explainability-with-grad-cam)
- [Final model selection and Kaggle submission file generation (submission.csv)](#final-model-selection-and-kaggle-submission-file-generation-submissioncsv)
- [Why I made these design choices](#why-i-made-these-design-choices)
- [Tech stack](#tech-stack)
- [Limitations and how I’d improve it next](#limitations-and-how-id-improve-it-next)
- [Author and context](#author-and-context)

---

## Overview

This project demonstrates an **end-to-end deep learning workflow** for an image classification task: predicting the competition label **“SEX”** (class 0 vs class 1) from **footprint images**. It includes data checks, preprocessing, model training, controlled experiments, and explainability (Grad-CAM).

---

## Getting Started

### Prerequisites

- A **Google account** (to open the notebook in Colab), or a Kaggle account (to run on Kaggle).
- Optional but recommended: enable a **GPU runtime** (GPU = faster training).

### Run in Colab

1. Click the Colab badge   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raf945/Deep_learning_Footprint_classifer/blob/main/Deep_learning_footprint_classifier.ipynb).
2. In Colab: **Runtime → Change runtime type → GPU**.
3. Run the notebook cells from top to bottom.

### Run on Kaggle

The notebook also works in Kaggle Notebooks (the file paths are compatible with typical Kaggle dataset layouts).  
Upload/import the notebook, attach the dataset, and run all cells.

---

## Problem definition

This project trains a **deep learning image classifier** that predicts **sex (class 0 vs class 1)** from **footprint images**.

- **Input:** a grayscale footprint image  
- **Output:** a prediction of class **0** or **1** (the “SEX” label used in the competition)

### Aim of the project (why I built it)

- Build a complete, recruiter-readable deep learning pipeline (not just a model).
- Compare a **simple custom model** against **industry-standard pre-trained models**.
- Run **controlled experiments** to improve results in a measurable way.
- Use **Grad-CAM** so the model’s decisions can be explained visually.

### Why this matters in the real world

A practical use case is **forensic support**: footprints can contain subtle shape/pressure patterns that correlate with demographic traits. This model is a **proof-of-concept** showing how an automated system can learn those patterns from labeled examples.

---

## Exploratory Data Analysis (EDA)

**EDA** means “Exploratory Data Analysis”: basic checks to understand the dataset before training.

The dataset contains:

- **1,573 labeled footprint images** (training data)
  - Class **0:** 845 images  
  - Class **1:** 728 images  
- **1,055 unlabeled images** (test set used for Kaggle submissions)

### Class balance (why it matters)

There are more examples of class 0 than class 1. This is called **class imbalance** and it can make a model “play it safe” by predicting the majority class too often. Because of that, I track metrics beyond accuracy (explained below).

---

## Preprocessing

Preprocessing means making the raw images consistent so the model can learn effectively.

### Steps used

1. **Convert to grayscale**  
   - These images are effectively single-channel; grayscale reduces complexity.
2. **Resize every image to 80 × 120**  
   - Standardizes the input and makes training faster.  
   - Trade-off: higher resolution can capture more detail but costs more memory/time.
3. **Normalize pixel values**  
   - Rescales pixel intensity values so training is more stable.

### Data augmentation (controlled “fake variety”)

Because the dataset is relatively small, the notebook uses **data augmentation**, meaning it slightly modifies training images so the model learns robust patterns instead of memorizing.

Augmentations used across experiments include:
- **Random horizontal flip** (mirrors an image sometimes)
- **Brightness/contrast jitter** (simulates different lighting conditions)
- **Random erasing** (covers small rectangles to force the model to use broader cues)
- **Small rotations** (simulates minor capture angle differences)

---

## Baseline model

### What a CNN is (simple explanation)

A **Convolutional Neural Network (CNN)** is a type of AI model designed for images. It learns small visual patterns (edges, curves, textures) and combines them into higher-level signals that help classification.

### Baseline CNN architecture

The baseline is intentionally compact (a good engineering choice for a small dataset):
- 2 convolution layers (pattern detectors)
- max pooling (downsamples to keep the strongest signals)
- fully connected layers (final decision-making part)

### Training setup (high level)

- **Loss function:** Cross-Entropy Loss  
  - “Loss” is how wrong the model is; training tries to minimize it.
- **Optimizer:** Adam  
  - A standard method for updating model weights efficiently.
- **Batch size:** 32  
  - Images processed per update step.
- **Epochs:** up to 50  
  - One epoch = one pass through the training set.
- **Weight initialization:** Kaiming initialization  
  - Helps stable training at the start.

---

## State-of-the-art comparison

When people say **state of the art (SOTA)** in image classification, they usually mean architectures that perform strongly across many benchmarks and are common in industry:

- **ResNet family:** reliable CNNs with “skip connections” that help training deep networks
- **EfficientNet family:** strong accuracy-per-compute tradeoff (useful when efficiency matters)
- **Vision Transformers (ViT/Swin):** newer architectures that treat image patches like tokens
- **ConvNeXt:** modern CNN design inspired by Transformers, very competitive performance

### What I tested in this project

I compared 3 well-known pre-trained CNNs (trained on huge general datasets) and adapted them for this grayscale footprint task:

- ResNet-18  
- AlexNet  
- EfficientNet-B0  

This technique is called **transfer learning**:
- Start from a model already trained on millions of images.
- Replace and train the final layer(s) for your specific task (“fine-tuning”).

### Why a small custom model can still win here

This dataset is:
- small (~1.5k labeled images),
- grayscale,
- resized to a small resolution (80×120),
- domain-specific (footprints ≠ everyday photos)

So a compact CNN with the right regularization can compete surprisingly well.

---

## 10 systematic experiments

I ran controlled experiments where I changed **one idea at a time**, measured impact, and kept what worked. This mirrors professional ML work: **hypothesis → experiment → measurement → decision**.

### Metrics (explained for non-technical readers)

- **Accuracy:** “How often did the model get it right overall?”
- **Precision (per class):** “When the model predicts class X, how often is it correct?”
- **Recall (per class):** “Out of all true class X examples, how many did it catch?”
- **F1 score:** A single balanced score combining precision + recall (useful with mild class imbalance)

### Baseline reference

- Baseline CNN  
  - Validation Accuracy: **0.760**
  - Validation F1: **0.737**

> **Validation** means testing on a held-out subset the model did not train on (to estimate real-world performance).

---

### Experiment 1 — Add dropout

**Dropout** randomly “turns off” parts of the model during training to reduce **overfitting** (memorizing training images).

- Validation Accuracy: **0.760**
- Validation F1: **0.715**

---

### Experiment 2 — Early stopping

**Early stopping** keeps the best checkpoint and prevents training too long after overfitting starts.

- Validation Accuracy: **0.770**
- Validation F1: **0.721**

---

### Experiment 3 — Batch normalization

**Batch normalization** stabilizes internal values during training and can improve consistency.

- Validation Accuracy: **0.702**
- Validation F1: **0.629**  
**Result:** Not helpful in this setup (likely due to architecture + small dataset + small resolution).

---

### Experiment 4 — Early stopping + dropout (best)

Combines two strong anti-overfitting techniques.

- **Validation Accuracy: 0.829 (best)**
- **Validation F1: 0.816 (best)**  
✅ Selected as the final model.

---

### Experiment 5 — Stronger augmentation with random erasing

**Random erasing** masks small patches so the model learns broader footprint cues.

- Validation Accuracy: **0.760**
- Validation F1: **0.701**

---

### Experiment 6 — Random erasing + dropout

Combines augmentation + regularization.

- Validation Accuracy: **0.803**
- Validation F1: **0.775**

---

### Experiment 7 — Random erasing + batch normalization

- Validation Accuracy: **0.779**
- Validation F1: **0.740**

---

### Experiment 8 — Random erasing + batch normalization + early stopping

- Validation Accuracy: **0.775**
- Validation F1: **0.766**

---

### Experiment 9 — Small rotations + batch normalization + early stopping

Rotations (±10°) improve robustness to minor capture/scan angle differences.

- Validation Accuracy: **0.822**
- Validation F1: **0.806**

---

### Experiment 10 — Shrink fully connected layers + early stopping

Reducing layer sizes lowers the number of parameters and can reduce overfitting.

- Validation Accuracy: **0.777**
- Validation F1: **0.748**

---

## Explainability with Grad-CAM

### What Grad-CAM is (plain English)

**Grad-CAM** produces a heatmap overlay showing **where the model “looked”** when making a prediction. This helps confirm the model uses the **footprint region** (not background noise).

### What I did

- Selected some of the worst-performing validation examples (highest loss)
- Generated Grad-CAM maps from the last convolution layer
- Checked whether the attention regions were reasonable

### Why it’s useful

It shows the project includes not only performance tuning, but also **interpretability**—an important real-world expectation in many AI roles.

---

## Final model selection and Kaggle submission file generation (submission.csv)

### Final selected model

**Baseline CNN + Early Stopping + Dropout** (Experiment 4)

- Best Validation Accuracy: **0.829**
- Validation F1: **0.816**
- Kaggle private leaderboard score reported in the notebook: **0.809**
- Total Kaggle submissions: **24**

### Submission file

The notebook generates a `submission.csv` file with:

- `filename`
- predicted `SEX`

This is the file format Kaggle expects for uploading predictions.

---

## Why I made these design choices

This project is structured to demonstrate industry-standard ML engineering thinking:

- Start with a **small baseline** to understand the problem quickly
- Standardize inputs (resize/normalize) to reduce noise and variability
- Use a clean train/validation split to avoid misleading results
- Run controlled experiments so improvements are measurable
- Use regularization (dropout/early stopping) to reduce overfitting on small datasets
- Use Grad-CAM to sanity-check what the model learned

---

## Tech stack

- PyTorch for model training and inference  
- Torchvision for transforms and pre-trained models  
- NumPy / pandas for data handling  
- Matplotlib for training curves and visualizations  

---

## Limitations and how I’d improve it next

If I were taking this beyond the competition/coursework setting:

- Train at a higher resolution (if compute allows) to capture more fine detail
- Use cross-validation (multiple validation splits) for more reliable performance estimates
- Try a stronger transfer learning setup (unfreeze more layers gradually)
- Add more footprint-specific augmentation (e.g., slight scaling, mild elastic transforms)
- Calibrate output probabilities for better decision thresholds

---

## Author and context

Coursework notebook authored as part of a deep learning assessment at Bournemouth University.
