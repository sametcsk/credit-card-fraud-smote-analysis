# 💳 Kredi Kartı Dolandırıcılığı Tespiti: Pipeline & Imbalanced Learning

[Open the notebook on nbviewer](https://nbviewer.org/github/sametcsk/credit-card-fraud-smote-analysis/blob/main/credit-card-fraud-analysis.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20Imbalanced--Learn-green)](https://imbalanced-learn.org/)

> **⚠️ Önemli Not:** GitHub, büyük Jupyter Notebook dosyalarını (.ipynb) render ederken bazen hata verebilir. Projenin kodlarını, grafiklerini ve analizlerini eksiksiz görüntülemek için lütfen yukarıdaki **"Open In nbviewer"** rozetine tıklayın.

## 🎯 Proje Hakkında
Finansal veri setlerinde karşılaşılan en büyük zorluk **Sınıf Dengesizliğidir (Class Imbalance)**. Bu projede kullanılan Avrupa kredi kartı veri setinde, 284.807 işlemden sadece **492'si (%0.17)** dolandırıcılık içermektedir.

Böyle bir veri setinde standart bir model "Her işlem güvenlidir" tahmini yapsa bile **%99.8 Accuracy (Doğruluk)** skoruna ulaşır, ancak banka milyonlarca dolar kaybeder. Bu proje, bu "Accuracy Tuzağına" düşmeden, dolandırıcıları yakalamak için **Veri Sızıntısını (Data Leakage)** önleyen özel bir Pipeline mimarisi sunmaktadır.

## 🛠️ Teknik Mimari ve Yaklaşım

Projede "Data Leakage" problemini çözmek için **SMOTE** işlemi, veriyi ayırmadan önce değil, **Cross-Validation döngüsü içinde** uygulanmıştır.

### 1. Veri Ön İşleme (Preprocessing)
* **Log Transformation:** `Amount` (Tutar) değişkeni aşırı çarpık (skewed) olduğu için Log dönüşümü ile normalize edildi.
* **Robust Scaler:** Dolandırıcılık işlemleri genelde aykırı değer (Outlier) içerdiği için, ortalama yerine medyanı kullanan RobustScaler tercih edildi.
* **Time Engineering:** Saniye cinsinden olan zaman verisi, dolandırıcıların aktivite saatlerini yakalamak için **Saat (Hour)** bilgisine dönüştürüldü.

### 2. Yarıştırılan Stratejiler
Aşağıdaki 4 farklı strateji **Random Forest** algoritması üzerinde test edilmiştir:
1.  **Baseline:** Hiçbir örnekleme yapılmadı (Referans modeli).
2.  **Class Weights:** Algoritmaya "azınlık sınıfına hata yaparsan daha fazla ceza kes" talimatı verildi.
3.  **SMOTE (Synthetic Minority Oversampling Technique):** Eğitim setinde sentetik dolandırıcı verileri üretildi.
4.  **SMOTE + Tomek Links:** Sentetik üretim sonrası, sınıflar arası sınır ihlali yapan gürültülü veriler temizlendi.

## 📊 Sonuçlar ve Performans

Dengesiz verilerde en güvenilir metrik olan **Precision-Recall Curve (AUPRC)** kullanılmıştır.


| Model | Recall (Yakalama Oranı) | Precision (Kesinlik) | F1-Score | PR AUC (Genel Başarı) |
|-------|-------------------------|----------------------|----------|-----------------------|
| **SMOTE (RF)** | **0.82** | 0.89 | 0.85 | **0.816** |
| SMOTE + Tomek | 0.82 | 0.89 | 0.85 | 0.816 |
| Class Weights | 0.75 | **0.94** | 0.83 | 0.811 |
| Baseline | 0.78 | 0.93 | 0.85 | 0.806 |

### 💡 İş Analizi (Business Insight)
* **Baseline Model:** Recall %78 seviyesinde kalırken, **SMOTE** entegreli model bunu **%82'ye** çıkarmıştır.
* **Kritik Karar:** SMOTE kullanımı, bankanın yakaladığı dolandırıcı sayısını artırırken, yanlış alarm (False Positive) oranını kabul edilebilir seviyede tutmuştur. Finansal risk yönetimi açısından **SMOTE Pipeline** en verimli çözümdür.

## 💻 Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için:

```bash
# Repoyu klonlayın
git clone https://github.com/sametcsk/credit-card-fraud-smote-analysis.git

# Klasöre gidin
cd credit-card-fraud-smote-analysis

# Gerekli kütüphaneleri yükleyin
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Notebook'u başlatın

jupyter notebook credit-card-fraud-analysis.ipynb
