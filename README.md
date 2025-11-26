# 🛡️ Intelligent Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Model](https://img.shields.io/badge/Model-IndoBERT-green)
![Domain](https://img.shields.io/badge/Domain-NLP-purple)

> **A robust Indonesian SMS/Chat fraud detection system utilizing IndoBERT, Explainable AI (LIME), and Text Similarity to detect scams with 98.67% accuracy.**

---

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Dataset Overview](#-dataset-overview)
- [Methodology & Architecture](#-methodology--architecture)
- [Performance Evaluation](#-performance-evaluation)
- [Installation & Usage](#-installation--usage)
- [Authors](#-authors)

---

## 🚀 About the Project
Digital fraud through short messaging services (SMS/WhatsApp) is a growing threat in Indonesia. This project, developed for the **Natural Language Processing (IF4072)** Final Project at **Institut Teknologi Bandung (ITB)**, aims to automate the filtering of malicious messages.

The system classifies messages into three categories:
1.  **Scam:** Malicious messages (e.g., fake lottery, money transfer requests, phishing links)[cite: 15].
2.  **Not Scam:** Legitimate transactional messages (e.g., OTPs, bank notifications, package delivery)[cite: 16].
3.  **Neutral:** Casual daily conversations without transactional context[cite: 17].

Unlike traditional filters, this system integrates **Explainable AI (XAI)** to provide transparency on *why* a message is flagged as a scam[cite: 21].

---

## ✨ Key Features
1.  **IndoBERT Classification:** Utilizes the `indobenchmark/indobert-base-p1` pre-trained model, fine-tuned to understand the nuances of the Indonesian language[cite: 39].
2.  **Slang Normalization:** Features a custom dictionary to convert Indonesian slang (e.g., "trf" $\rightarrow$ "transfer", "sy" $\rightarrow$ "saya") into formal language, significantly improving model comprehension [cite: 177-180].
3.  **Explainable AI (LIME):** Highlights suspicious keywords (e.g., "hadiah", "klik", "menang") to help users understand the model's reasoning [cite: 250-251].
4.  **Text Similarity:** Compares incoming messages against a database of known fraud patterns to detect recurring scams[cite: 253].

---

## 📊 Dataset Overview
The dataset was constructed from public repositories, social media reports, and synthetic data augmentation via LLM [cite: 25-26].

* **Total Samples:** 1,503 rows[cite: 28].
* **Class Distribution:** Balanced[cite: 68].
    * Neutral: 520
    * Not Scam: 495
    * Scam: 488
* **Labeling Process:** Manual annotation by 3 annotators with Cross-Validation Quality Assurance [cite: 29-30].

---

## ⚙️ Methodology & Architecture
The system follows a standard NLP pipeline:

1.  **Preprocessing:**
    * **Cleaning:** URL removal, phone number redaction (privacy), symbol removal [cite: 173-176].
    * **Case Folding:** Lowercasing.
    * **Normalization:** Mapping slang/abbreviations to standard Indonesian[cite: 244].
2.  **Tokenization:** IndoBERT Tokenizer (Max length: 128) [cite: 181-182].
3.  **Modeling:** Transfer Learning (Fine-tuning IndoBERT)[cite: 184].
    * *Epochs:* 5
    * *Batch Size:* 16
    * *Learning Rate:* 2e-5
    * *Optimizer:* AdamW
4.  **Inference:** Outputs the predicted label and a confidence score[cite: 40].

---

## 📈 Performance Evaluation
The model was evaluated on a held-out Test Set (15% split) and achieved state-of-the-art results:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.67%** [cite: 207] |
| **F1-Score (Macro)** | **98.66%** [cite: 209] |
| **Recall (Scam)** | **0.99** [cite: 248] |

> **Insight:** The high recall (0.99) on the Scam class indicates the system is highly sensitive and minimizes False Negatives, ensuring almost no fraud messages slip through[cite: 248].

### Demo Inference Examples
* *Input:* "Selamat! Anda menang hadiah 10 juta. Klik link ini."
    * **Prediction:** SCAM (99.8%) [cite: 256]
* *Input:* "Shopee: Paket Anda sudah sampai di hub terdekat."
    * **Prediction:** NOT SCAM (99.9%) [cite: 257]
* *Input:* "Gue lagi cari laptop buat kuliah nih."
    * **Prediction:** NEUTRAL (99.9%) [cite: 258]

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch
* Transformers

### 1. Clone Repository
```bash
git clone [https://github.com/auraleaas/ScamDetection.git](https://github.com/auraleaas/ScamDetection.git)
cd ScamDetection