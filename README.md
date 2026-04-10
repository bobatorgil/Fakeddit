# Multimodal Fake News Detection (Fakeddit)

## 📌 Overview
This project focuses on detecting fake news using multimodal data from the :contentReference[oaicite:0]{index=0}.  
The goal is to analyze and model relationships between textual and visual information for improved fake news classification.

This repository provides a baseline implementation using image features, with extensibility toward full multimodal fusion.

---

## 📂 Dataset

We use the Fakeddit dataset, which contains:

- Text data (news titles)
- Image data (linked via URLs)
- Labels:
  - `2_way_label` (Fake / Real)
  - `3_way_label`, `6_way_label` (not used in this project)

### Data Structure
data/
├── multimodal_train.tsv
├── images/


### Key Columns

- `id`: unique identifier
- `clean_title`: processed text
- `image_url`: image link
- `hasImage`: image availability
- `2_way_label`: target label (binary)

---

## ⚙️ Data Processing

- Remove unused columns (`6_way_label`, `3_way_label`, `title`)
- Handle missing values by replacing with empty strings
- Download images from URLs
- Filter out invalid or corrupted images
- Stratified sampling for balanced subset training

---

## 🖼️ Image Processing

All images are preprocessed using:

- Resize → 256
- Center Crop → 224
- Normalize (ImageNet statistics)

Invalid images are automatically skipped during dataset loading.

---

## 🧠 Model

We implement an image-based baseline using:

- Backbone: ResNet50 (pretrained on ImageNet)
- Modification:
  - Final fully connected layer replaced for binary classification

---

## ⚙️ Training

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 3

Training is performed on GPU if available.

---

## 📊 Evaluation

Evaluation metrics:

- Accuracy
- F1 Score

Predictions are obtained using argmax over output logits.

---

## 📈 Results

Accuracy: 0.9134
F1 Score: 0.9216


---

## 🧩 Features

- Image-based fake news detection baseline
- Automatic image downloading and validation
- Robust handling of missing and corrupted data
- Transfer learning with ResNet50

---

## 🚀 How to Run

### Install dependencies
```bash
pip install torch torchvision transformers pandas scikit-learn pillow 
```
### Run training
```bash
python fakeddit_resnet.py
```