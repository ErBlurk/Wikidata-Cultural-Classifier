# Automatic Wikidata Items Cultural Classification: Agnostic, Representative, Exclusive

##### Multilingual Natural Language Processing - Master in Artificial Intelligence and Robotics, Sapienza University of Rome

---

### Authors:
> 1986191: Leonardo Mariut \
> 2202212: Kevin Giannandrea \
> 1984056: Christian Bianchi

---

[Original repository here](https://github.com/giankev/wikidata_cultural_classifier.git)

[Detailed project report here (PDF)](Report_Cultural_Classifier.pdf)

---

## Overview

Classify Wikipedia/Wikidata items as **agnostic**, **representative**, or **exclusive** using knowledge enriched features and a non-LM baseline.

This repo is a compact, reproducible implementation of a knowledge augmented classifier that combines Wikidata and Wikipedia features with classical ML. It focuses on the significant parts only: feature extraction, XGBoost modeling, evaluation and a few transformer baselines.

## Highlights

* Dataset \~6k items labeled as agnostic, representative, exclusive.
* Features from Wikidata: sitelink counts, claim counts, presence of properties like P17, and derived stats such as translation entropy.
* Wikipedia scraping adds page length and number of internal links.
* Final model: tuned XGBoost on all features, 77% dev accuracy. Hyperparams: `colsample_bytree=0.8`, `learning_rate=0.03`, `max_depth=6`, `min_child_weight=1`, `n_estimators=270`, `subsample=0.8`.
* Transformer baselines (BERT, RoBERTa, DeBERTa-v3) trained on title and description with metadata. Best BERT-base: 77.3% accuracy, 75.7% macro F1.
* Graph experiments explored per-item Wikidata QID graphs and graph2vec embeddings but were not included in the final pipeline due to scalability limits.

## Notes on methodology

* Feature engineering is central: sitelink statistics and culturally relevant claims are strong signals.
* Translation entropy of sitelinks captures title diversity across languages and helps separate agnostic from exclusive items.
* Simpler feature based models outperformed more complex graph pipelines on dev data, so we selected XGBoost for final use.

## Results

| Config             | USE\_GRAPH | USE\_EMBED | USE\_SITELINK | Accuracy   | Precision   | Recall   | F1-score   |
| ------------------ | :--------: | :--------: | :-----------: | ---------: | ----------: | -------: | ---------: |
| Sitelinks only     |    False   |    False   |      True     |     0.6522 |      0.6524 |   0.6462 |     0.6444 |
| Graph only         |    True    |    False   |     False     |     0.5819 |      0.5816 |   0.5819 |     0.5779 |
| Embeddings only    |    False   |    True    |     False     |     0.5351 |      0.5367 |   0.5439 |     0.5397 |
| Graph + Embeddings |    True    |    True    |     False     |     0.5853 |      0.5863 |   0.5858 |     0.5811 |
| Graph + Sitelinks  |    True    |    False   |      True     |     0.5786 |      0.5790 |   0.5818 |     0.5765 |

**Table 1.** Performance of a 3-layer DNN trained on different combinations of experimental features. `USE_GRAPH` includes mesoscale and topological properties, `USE_EMBED` uses Graph2Vec embeddings, `USE_SITELINK` includes sitelink count and translation entropy. Arrows indicate that higher values are better.

---

| Model                 | Accuracy   | F1-score   | Precision   |
| --------------------- | ---------: | ---------: | ----------: |
| XGBoost (Final)       |       0.77 |       0.76 |        0.76 |
| Stacking (XGB + NN)   |       0.76 |       0.76 |        0.76 |
| DeBERTa-v3 (LM-based) |       0.77 |      0.755 |        0.76 |
| RoBERTa (LM-based)    |       0.77 |      0.758 |        0.76 |
| BERT-base (LM-based)  |      0.773 |      0.757 |        0.77 |
| Graph-based (DNN)     |       0.58 |       0.58 |        0.58 |

**Table 2.** Accuracy and macro-averaged F1 and precision across models. Transformer classifiers perform on par with XGBoost, with BERT slightly ahead in accuracy. Graph-based models underperform due to scalability limits and limited semantic abstraction.
