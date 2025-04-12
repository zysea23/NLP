# **Named Entity Recognition (NER) Report**

---
## **1. Literature Review: Traditional and Deep Learning-Based NER Approaches**

Named Entity Recognition (NER) has evolved significantly over the years, transitioning from **rule-based methods** to **statistical learning** and finally to **deep learning-based approaches**. Below, we discuss these methods and their impact on NER performance, with appropriate citations.

### **1.1 Rule-Based NER Approaches**
Early NER systems relied on manually crafted rules and dictionaries, using pattern matching and contextual heuristics to identify entities. While effective in specific domains, these approaches required extensive manual effort, lacked adaptability, and struggled with unseen data.

### **1.2 Statistical Learning-Based NER Approaches**
Machine learning introduced more adaptable approaches to NER, including:

- **Hidden Markov Models (HMM)**: HMMs model sequences probabilistically, treating entity labels as hidden states. However, they assume that observations are conditionally independent, which is a strong limitation for language tasks [1].
- **Maximum Entropy Models (ME)**: ME models use feature-based probability distributions to improve flexibility, though they do not effectively model dependencies between labels in sequential data [2].
- **Conditional Random Fields (CRF)**: CRFs, proposed by Lafferty et al. (2001), improve upon HMMs by capturing dependencies between neighboring labels, making them a popular choice for sequence labeling tasks [3].

Despite their effectiveness, statistical models often require extensive feature engineering and struggle with handling complex contextual relationships.

### **1.3 RNN-Based NER Approaches**
Recurrent Neural Networks (RNNs) improved upon statistical models by learning distributed representations of text sequences. Two key advancements emerged:

- **Long Short-Term Memory (LSTM)**: LSTM networks mitigate the vanishing gradient problem, allowing models to capture long-range dependencies [4].
- **Bidirectional LSTM (BiLSTM)**: BiLSTM extends LSTM by incorporating past and future context, enhancing performance in sequence labeling tasks like NER [5].

One of the most effective hybrid approaches is **BiLSTM-CRF**, which combines BiLSTM’s ability to learn contextual embeddings with CRF’s structured prediction, yielding significant performance improvements over traditional models [6].

### **1.4 Transformer-Based NER Approaches**
While RNNs advanced sequence modeling, they suffered from **sequential processing inefficiencies**, making them difficult to scale. **Transformers**, introduced by Vaswani et al. (2017), addressed these issues by introducing **self-attention mechanisms**, allowing models to process entire sequences in parallel [7]. Key Transformer-based NER models include:

- **BERT (Bidirectional Encoder Representations from Transformers)**: BERT is a **pretrained Transformer-based model** that captures contextual embeddings more effectively than BiLSTMs. Fine-tuning BERT on NER datasets has achieved **state-of-the-art results** [8].
- **DistilBERT**: A **lighter version of BERT**, DistilBERT retains 97% of BERT’s performance while reducing computational cost, making it an efficient alternative [9].
- **Domain-Specific Models (e.g., BioBERT, SciBERT)**: These models specialize in specific fields (e.g., biomedical text processing) and outperform general-purpose models like BERT on domain-specific tasks [10].

These advancements have significantly improved NER performance, enabling more accurate and efficient entity recognition across various domains.

---

### **References**
1. Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition." *Proceedings of the IEEE*, 77(2), 257-286. [https://doi.org/10.1109/5.18626](https://doi.org/10.1109/5.18626)
2. Berger, A. L., Della Pietra, S. A., & Della Pietra, V. J. (1996). "A maximum entropy approach to natural language processing." *Computational Linguistics*, 22(1), 39-71. [https://www.aclweb.org/anthology/J96-1002/](https://www.aclweb.org/anthology/J96-1002/)
3. Lafferty, J., McCallum, A., & Pereira, F. (2001). "Conditional random fields: Probabilistic models for segmenting and labeling sequence data."  *Proceedings of the 18th International Conference on Machine Learning (ICML)*, 282-289. [https://repository.upenn.edu/cis_papers/159/](https://repository.upenn.edu/cis_papers/159/)  
4. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780. [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
5. Graves, A., & Schmidhuber, J. (2005). "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks*, 18(5-6), 602-610. [https://doi.org/10.1016/j.neunet.2005.06.042](https://doi.org/10.1016/j.neunet.2005.06.042)
6. Huang, Z., Xu, W., & Yu, K. (2015). "Bidirectional LSTM-CRF models for sequence tagging." *arXiv preprint arXiv:1508.01991*. [https://arxiv.org/abs/1508.01991](https://arxiv.org/abs/1508.01991)
7. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). "Attention is all you need." *Advances in Neural Information Processing Systems (NeurIPS)*. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
8. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of deep bidirectional transformers for language understanding." *Proceedings of NAACL-HLT*. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
9. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper, and lighter." *arXiv preprint arXiv:1910.01108*. [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
10. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2019). "BioBERT: A pre-trained biomedical language representation model for biomedical text mining." *Bioinformatics*, 36(4), 1234-1240. [https://doi.org/10.1093/bioinformatics/btz682](https://doi.org/10.1093/bioinformatics/btz682)


---

## **2. Overview**

1. Train and compare multiple NER models using the **CoNLL-2003** dataset.
2. Evaluate the performance of **CRF, BiLSTM, BiLSTM-CRF, BERT, and DistilBERT** models.
3. Experiment with **DistilBERT on the BC5CDR dataset** to analyze its generalization ability in a domain-specific NER task.
4. Conduct error analysis and discuss misclassifications.
5. Reflect on findings and limitations while suggesting potential improvements.

---

## **3. Dataset Description**

### **CoNLL-2003 Dataset**
- **Format:** BIO-tagging with named entity categories.
- **Statistics:**
  - Training samples: **14,041**
  - Validation samples: **3,250**
  - Test samples: **3,453**
- **NER Tags:** ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

### **BC5CDR Dataset** (Biomedical domain-specific dataset)
- **Format:** BIO-tagging with **Chemical** and **Disease** entities.
- **Statistics:**
  - Training samples: **5,228**
  - Validation samples: **5,330**
  - Test samples: **5,865**
- **NER Tags:** ['O', 'B-Chemical', 'B-Disease', 'I-Disease', 'I-Chemical']

---

## **4. Models and Implementation**

### **4.1 Models Evaluated**
1. **Conditional Random Fields (CRF)**
2. **BiLSTM** (Bidirectional LSTM)
3. **BiLSTM-CRF**
4. **BERT** (Pretrained Transformer)
5. **DistilBERT** (Lightweight Transformer Model)
6. **DistilBERT on BC5CDR Dataset**

### **4.2 Hyperparameter Tuning**
| Model | Tuned Hyperparameters | Optimal Settings |
|--------|----------------------|-----------------|
| **CRF** | c1, c2 | c1 = 0.01, c2 = 0.01 |
| **BiLSTM** | Embedding Dim, Hidden Dim, LSTM Layers, Dropout Rate, Epochs | 200, 512, 3, 0.5, 11 |
| **BiLSTM-CRF** | Embedding Dim, Hidden Dim, LSTM Layers, Dropout Rate, Epochs | 400, 256, 3, 0.5, 17 |
| **BERT** | None | Batch Size = 16, Epochs = 9 |
| **DistilBERT** | None | Batch Size = 16, Epochs = 6 |

### **4.3 Training Time**
| Model | Approx. Training Time | Hardware |
|--------|----------------------|----------|
| CRF | 5-15 min | CPU |
| BiLSTM | ~30-40 min(200+ epochs) | P100 GPU |
| BiLSTM-CRF | ~45-60 min(200+ epochs) |P100 GPU |
| BERT | 15-20 min (10 epochs) | P100 GPU |
| DistilBERT | 10-15 min (10 epochs) | P100 GPU |

---

## **5. Evaluation Metrics**

All models are evaluated using **entity-level F1-score**. Entity-level F1-score is chosen as the primary metric because:

- It effectively balances precision and recall, making it a popular choice for NER.
- It ensures that an entity is only considered correct if all tokens are correctly labeled. Token-level accuracy may give misleadingly high results due to a large number of 'O' labels.
---

## **6. Experimental Results**

| Model               | Dataset       | Precision | Recall | F1-score |
| ------------------- | ------------ | --------- | ------ | -------- |
| CRF                | CoNLL-2003    | -         | -      | 0.7914   |
| BiLSTM             | CoNLL-2003    | -         | -      | 0.7413   |
| BiLSTM-CRF         | CoNLL-2003    | -         | -      | 0.7678   |
| BERT               | CoNLL-2003    | 0.8956    | 0.8994 | 0.8975   |
| DistilBERT         | CoNLL-2003    | 0.8888    | 0.8932 | 0.8910   |
| DistilBERT         | BC5CDR        | 0.8760    | 0.8995 | 0.8876   |

---

## **7. Misclassification Analysis** (Based on BiLSTM-CRF)

### **7.1 Misclassification Distribution**
| True Label | Predicted Label | Count |
|------------|----------------|-------|
| B-ORG | O | 235 |
| B-PER | O | 146 |
| B-MISC | O | 114 |
| B-ORG | B-LOC | 106 |
| B-LOC | O | 105 |
| I-PER | O | 100 |

### **7.2 Misclassification Samples (First 5 Errors)**
| Token | True Label | Predicted Label |
|--------|------------|----------------|
| JAPAN | B-LOC | B-PER |
| CHINA | B-PER | O |
| Nadim | B-PER | O |
| Ladki | I-PER | O |
| AL-AIN | B-LOC | B-ORG |

**Observations:**
- **PER and LOC confusion** suggests wrong labels may mislead models.
- **ORG and LOC confusion** indicates organization names sometimes contain location indicators.
- **Entity misclassification as 'O'** suggests that some entities were not recognized.

---

## **8. Discussion and Findings**

### **8.1 Model Comparisons**
1. **CRF and BiLSTM+CRF show reasonable performance**, xxxxxx.
2. **BERT and DistilBERT(F1 ~ 89%) significantly outperform traditional methods (F1 ~ 75%-80%)**, demonstrating the strength of transformers for NER.
3. **BiLSTM-CRF improves over BiLSTM alone (76.78% vs. 74.13%)**, validating the importance of CRF for structured prediction.
4. **DistilBERT achieves near BERT-level performance (F1: 89.10%) with fewer parameters**, making it a viable lightweight alternative.

### **8.2 Domain Adaptation (BC5CDR vs. CoNLL-2003)**
- DistilBERT performed comparably on **BC5CDR (88.76%) vs. CoNLL-2003 (89.10%)**.
- Lower precision in BC5CDR suggests medical NER requires domain-specific models (e.g., BioBERT, SciBERT).

### **8.3 Limitations & Future Work**
- **More fine-tuning** may further improve deep models.
- **CRF-based models can perform additional feature engineering**, limiting their generalization.
- **Exploring alternative architectures** such as SpanBERT and Transformer-CRF could enhance results.

---

## **9. Discussion and Findings**

### **9.1 Model Comparisons**
1. **Traditional methods such as CRF show reasonable performance**, offer interpretable decision-making and require less computational power, making them viable for resource-constrained environments.
2. **BERT and DistilBERT (F1 ~ 89%) significantly outperform traditional methods (F1 ~ 75%-80%)**, demonstrating the strength of transformers for NER, but they are more time and resource(GPU) consuming.
3. **BiLSTM-CRF improves over BiLSTM alone (76.78% vs. 74.13%)**, validating the importance of CRF for structured prediction.
4. **DistilBERT achieves near BERT-level performance (F1: 89.10%) with fewer parameters**, making it a viable lightweight alternative.

### **9.2 Domain Adaptation (BC5CDR vs. CoNLL-2003)**
- DistilBERT performed comparably on **BC5CDR (88.76%) vs. CoNLL-2003 (89.10%)**, with only a slight drop in performance. This is somewhat unexpected, as domain-specific datasets often introduce more challenges.
- **Possible explanations for this small performance gap:**
  1. **Dataset Simplicity:** BC5CDR primarily focuses on biomedical entities with well-defined entity structures.
  2. **Limited Entity Ambiguity:** The biomedical domain, while specialized, may have fewer ambiguous entities compared to general NER datasets. In contrast, CoNLL-2003 includes names of people, organizations, and locations, which may lead to greater variability.
  3. **Training Overlap or Pretraining Effect:** Transformer-based models like DistilBERT may have been pretrained on medical texts or similar corpora, reducing the domain shift when applied to BC5CDR.
  4. **Annotation Consistency:** The annotation guidelines in BC5CDR may be more consistent, making it easier for models to learn compared to datasets with more annotation subjectivity.

## **9.3 Limitations & Future Work**
- **More fine-tuning** may further improve deep models, exploring hyperparameter optimization tools like **Optuna**
- **CRF-based models can perform additional feature engineering**, limiting their generalization.
- **Exploring alternative architectures** such as SpanBERT and Transformer-CRF could enhance results.
- **Exploring the impact of different embedding strategies**:  
  - Comparing **static word embeddings** (e.g., Word2Vec, GloVe, FastText) vs. **contextual embeddings** (e.g., BERT, ELMo) could provide insights into their effectiveness for different entity types.  
  - **Domain-specific embeddings** trained on biomedical corpora (e.g., BioWordVec, SciBERT embeddings) may improve NER performance in specialized datasets like BC5CDR.  
- **More in-depth misclassification analysis** could provide further insights, including false positive/negative patterns, boundary errors, and entity-type-specific misclassifications.
- **Improving code quality** by gaining deeper familiarity with **PyTorch and Transformer architectures**, optimizing model implementation, enhancing training efficiency, and automating training, hyperparameter tuning, and result logging workflows.  