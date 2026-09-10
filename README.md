# Credit Card Fraud Detection with SMOTE
# SMOTE ile Kredi Kartı Dolandırıcılık Tespiti

A machine learning project focusing on detecting fraudulent credit card transactions by addressing extreme class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).

Sentetik Azınlık Aşırı Örnekleme Tekniği (SMOTE) kullanarak aşırı sınıf dengesizliğini ele alan ve kredi kartı dolandırıcılık işlemlerini tespit etmeye odaklanan makine öğrenmesi projesi.

> **Task / Görev:** Binary Classification / İkili Sınıflandırma · **Domain / Alan:** Finance & Cybersecurity / Finans & Siber Güvenlik

---

## Project Structure / Proje Yapısı

```text
credit-card-fraud-smote-analysis/
├── notebooks/
│   └── credit-card-fraud-analysis.ipynb   # Exploratory analysis, SMOTE application, and modeling
├── data/
│   └── (Dataset files should be placed here)
├── src/
│   └── __init__.py          # Placeholder for future modularization
├── requirements.txt
└── .gitignore
└── README.md
```

## Quick Start / Hızlı Başlangıç

```bash
git clone https://github.com/sametcsk/credit-card-fraud-smote-analysis.git
cd credit-card-fraud-smote-analysis
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt

# Launch Jupyter Notebook / Jupyter Notebook'u başlatın
jupyter notebook notebooks/credit-card-fraud-analysis.ipynb
```

## Pipeline Overview / Pipeline Adımları

1. **Data Preprocessing** — Handling anonymized PCA features and scaling `Time` and `Amount` variables.
   *Veri Ön İşleme — Anonimleştirilmiş PCA özelliklerinin işlenmesi ve `Zaman` ile `Miktar` değişkenlerinin ölçeklenmesi.*
2. **Imbalanced Class Handling** — Applying SMOTE to oversample the minority class (fraudulent transactions).
   *Dengesiz Sınıf Yönetimi — Azınlık sınıfını (dolandırıcılık işlemleri) aşırı örneklemek için SMOTE uygulanması.*
3. **Modeling** — Training classifiers (e.g., Logistic Regression, Random Forest) on both the original and SMOTE-augmented datasets.
   *Modelleme — Hem orijinal hem de SMOTE ile artırılmış veri setlerinde sınıflandırıcıların (örn. Lojistik Regresyon, Random Forest) eğitilmesi.*
4. **Evaluation** — Comparing models using Precision, Recall, F1-Score, and AUPRC (Area Under the Precision-Recall Curve) to highlight the impact of SMOTE.
   *Değerlendirme — SMOTE'nin etkisini vurgulamak için Kesinlik, Duyarlılık, F1-Skoru ve AUPRC (Kesinlik-Duyarlılık Eğrisi Altındaki Alan) kullanılarak modellerin karşılaştırılması.*

## Tech Stack / Kullanılan Teknolojiler

`Python` · `Scikit-learn` · `Imbalanced-learn (SMOTE)` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## License / Lisans

Educational purposes.
Eğitim amaçlıdır.
