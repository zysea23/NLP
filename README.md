# Natural Language Processing

A collection of projects exploring Natural Language Processing.

---

## Named Entity Recognition

### 📂 **[Traditional to Transformer-based NER Models](https://github.com/zysea23/NLP/tree/main/NER(CRF%2C%20BiLSTM%2C%20BiLSTM%2BCRF))**

#### Summary

In this project, I built and optimized multiple models for **Named Entity Recognition (NER)**, focusing on extracting structured information from unstructured text.

#### Highlights
- Designed and compared **CRF**, **BiLSTM**, **BiLSTM-CRF**, **BERT**, and **DistilBERT** models for entity extraction.
- Tuned hyperparameters and optimized model performance for different architectures.
- Evaluated models using **entity-level F1-score** to accurately capture extraction quality.
- Extended experiments to the **BC5CDR biomedical dataset** to assess domain adaptability.
- Conducted **error analysis** to understand common misclassification patterns and model limitations.
- Reflected on findings and proposed improvements, including **advanced embeddings**, **data augmentation**, and **architecture enhancements**.

#### Results
- **Transformer-based models** (BERT, DistilBERT) significantly outperformed traditional methods, achieving ~89% F1-score.
- **DistilBERT** maintained strong performance across both general and domain-specific datasets.
- Insights from evaluation informed strategies for improving information extraction from diverse text sources.

---

### 📂 **[Transformer Model Comparison for NER](https://github.com/zysea23/NLP/tree/main/NER(Transformer-based%20models))**

#### Summary

This project focused on **evaluating Transformer-based models** to extract insights from unstructured text, using **NER** as the target task on the CoNLL-2003 dataset.

#### Highlights
- Benchmarked multiple Transformer models: **BERT**, **DistilBERT**, **ALBERT**, **DeBERTa**, and **T5**.
- Conducted **efficiency analysis**, comparing model accuracy, runtime, and parameter sizes.
- Explored the **text-to-text approach** with T5, reframing NER as a sequence generation task.
- Performed **entity-level performance breakdown** and deep-dive **misclassification analysis**.
- Visualized findings with confusion matrices and performance charts for clearer insights.

#### Results
- **DeBERTa** achieved the highest F1-score (92%), excelling at entity extraction.
- **DistilBERT** offered an excellent balance of efficiency and performance.
- The project provided comparative insights into Transformer models' capabilities for **information extraction** tasks.
- Identified improvement areas such as handling underrepresented entity classes (**MISC**) and recommended solutions like **class balancing** and **contrastive learning**.

---

