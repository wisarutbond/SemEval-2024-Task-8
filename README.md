# Detecting Machine-Generated Texts Using RoBERTa and Linguistic Features

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)

This repository contains the code, data analysis, and final report for our system designed to distinguish between human-written and machine-generated texts. 

**Note:** This research was developed as a final project for the Natural Language Processing Systems course at Chulalongkorn University based on the SemEval-2024 Task 8 dataset.

## 📄 Read the Full Report
**[Read the Final Technical Report (PDF) here](./2209678_NLP_SYS_Final_Project_Report.pdf)**

## 🧠 Project Overview
Distinguishing between human-written and machine-generated texts has become increasingly vital. This project tackles the monolingual (English) binary classification task to identify the origins of a given paragraph. 

We explored the effectiveness of fine-tuning a `roberta-base` classifier and proposed two feature-augmentation strategies to enrich the model's `[CLS]` embeddings with linguistic features:
1. **Handcrafted Syntactic Features (spaCy):** Extracted features like dependency tree depths, POS bigrams, and syntactic structures.
2. **LLM-Derived Stylistic Attributes (Gemini 2.0 Flash API):** Extracted deep stylistic patterns like clause complexity, passive voice usage, and sentence length variance.

## 🚀 Key Results
Our comparative analysis demonstrated that explicit, handcrafted syntactic cues provided more robust signals for detection than the LLM-derived stylistic features.

| Model | Macro F1 Score | Accuracy |
| :--- | :---: | :---: |
| RoBERTa (Baseline) | 0.7619 | 0.7680 |
| **RoBERTa + spaCy Features** | **0.8138** | **0.8147** |
| RoBERTa + Gemini 2.0 Features | 0.8001 | 0.8012 |

*Augmenting the baseline with spaCy features improved the overall Macro F1 score by over 5%.*

## 🛠️ Tech Stack & Architecture
* **Framework:** PyTorch
* **Base Model:** RoBERTa (`roberta-base`)
* **Linguistic Parsing:** spaCy (`en_core_web_md`)
* **LLM API Integration:** Google Gemini 2.0 Flash API
* **Data Preprocessing:** scikit-learn (`StandardScaler`, `OneHotEncoder`)

## ⚙️ Methodology Highlight
Instead of relying solely on the transformer's self-attention mechanisms, we concatenated normalized linguistic feature vectors directly to RoBERTa's `[CLS]` token embedding before passing it through the final classification head. This hybrid architecture allows the model to leverage both deep contextual semantics and highly interpretable linguistic signals.

---
*Developed by Chanachon Wongchaya and Wisarut Tangtemjit*
