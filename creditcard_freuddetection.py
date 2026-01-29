#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, make_scorer, recall_score, precision_score

from imblearn.pipeline import make_pipeline as make_pipeline_imb 
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv("creditcard.csv")
print("İlk 5 satır:")
df.head()


# In[ ]:


print(f"Boyut: {df.shape}\n")

print("Sütün bilgileri:")
print(df.info())

print("İstatistiksel özet:")
print(df.describe())


# In[ ]:


print("Tekrar eden satır sayısı:")
df.duplicated().sum()


# In[ ]:


print("Eksik değer kontrolü:")
print(df.isnull().sum())


# In[ ]:


print(f"Orijinal Boyut: {df.shape}")
df.drop_duplicates(inplace=True)
print(f"Duplicates Silindi, Yeni Boyut: {df.shape}")


# In[ ]:


class_counts = df['Class'].value_counts()
print(f"\nSınıf dağılımı:\n{class_counts}")

class_percentages = df['Class'].value_counts(normalize=True) * 100
print(f"\nSınıf yüzdeleri:\n{class_percentages}")

print(f"0= Normal işlemler: {class_counts[0]} ({class_percentages[0]:.2f}%)")
print(f"1= Dolandırıcılık işlemleri: {class_counts[1]} ({class_percentages[1]:.2f}%)")


# In[ ]:


plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, hue='Class', palette=['blue', 'red'])
plt.title('Sınıf Dağılımı (0: Normal, 1: Fraud)')
plt.xlabel('İşlem Türü')
plt.ylabel('Sayı')
plt.show()


# In[ ]:


imbalenced_ratio = class_counts[0] / class_counts[1]
print(f"\nSınıf dengesizliği oranı (Normal/Fraud): {imbalenced_ratio:.2f}")


# In[ ]:


df["Hour"] = (df["Time"] // 3600) % 24
df=df.drop(columns=['Time'], axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

sns.histplot(df[df['Class'] == 0]['Hour'], bins=24, color='blue', ax=ax1, kde=False)
ax1.set_title('Normal İşlemlerin Saatlik Dağılımı')
ax1.set_ylabel('İşlem Sayısı')

sns.histplot(df[df['Class'] == 1]['Hour'], bins=24, color='red', ax=ax2, kde=False)
ax2.set_title('Dolandırıcılık (Fraud) İşlemlerinin Saatlik Dağılımı')
ax2.set_xlabel('Saat (0-24)')
ax2.set_ylabel('İşlem Sayısı')

plt.show()


# In[ ]:


df["Amount_log"] = np.log1p(df["Amount"])
df=df.drop(columns=['Amount'], axis=1)


# In[ ]:


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Eğitim setindeki Boyutu:", X_train.shape)
print("Test setindeki Boyutu:", X_test.shape)


# In[ ]:


model_results = {}

def run_model_scenario(pipeline, X_train, y_train, X_test, y_test, name):
    print(f"\n{'='*20} {name} {'='*20}")

    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'recall': 'recall', 'precision': 'precision', 'f1': 'f1'}
    cv_results = cross_validate(pipeline, X_train, y_train, cv=kf, scoring=scoring,n_jobs=-1)

    print(f"Ortalama Recall: {cv_results['test_recall'].mean():.4f}")
    print(f"Ortalama Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Ortalama F1-Score: {cv_results['test_f1'].mean():.4f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nTest Seti Performans Raporu:")
    print(classification_report(y_test, y_pred, digits=4))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    model_results[name] = {
        'recall': recall,
        'precision': precision,
        'auc': pr_auc,
        'y_test': y_test,
        'y_pred': y_pred
    }


# In[ ]:


# 1. BASELINE 
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_base = make_pipeline_imb(rf_base)
run_model_scenario(pipe_base, X_train_scaled, y_train, X_test_scaled, y_test, "Baseline (RF)")

# 2. CLASS WEIGHTS 
rf_weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
pipe_weighted = make_pipeline_imb(rf_weighted)
run_model_scenario(pipe_weighted, X_train_scaled, y_train, X_test_scaled, y_test, "Class Weights (RF)")

# 3. SMOTE 
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_smote = make_pipeline_imb(SMOTE(random_state=42), rf_smote)
run_model_scenario(pipe_smote, X_train_scaled, y_train, X_test_scaled, y_test, "SMOTE (RF)")

# 4. SMOTE + TOMEK
rf_st = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
pipe_st = make_pipeline_imb(SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=42), rf_st)
run_model_scenario(pipe_st, X_train_scaled, y_train, X_test_scaled, y_test, "SMOTE + Tomek (RF)")


# In[ ]:


summery_list = []
for name, metrics in model_results.items():
    y_test = metrics['y_test']
    y_pred = metrics['y_pred']

    recall=recall_score(y_test, y_pred)
    precision=precision_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    auc_score=metrics['auc']

    summery_list.append([name, recall, precision, f1, auc_score])

df_summary = pd.DataFrame(summery_list, columns=['Model', 'Recall', 'Precision', 'F1-Score', 'AUC-PR'])
df_summary = df_summary.sort_values(by='Recall', ascending=False).reset_index(drop=True)

for name, metrics in model_results.items():
    plt.plot(metrics['recall'], metrics['precision'], lw=2, 
             label=f'{name} (AUC = {metrics["auc"]:.3f})')  


plt.xlabel('Recall (Duyarlılık - Fraud Yakalama Oranı)', fontsize=12)
plt.ylabel('Precision (Kesinlik - Doğru Alarm Oranı)', fontsize=12)
plt.title('Precision-Recall Eğrisi: Şampiyon Kim?', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()



# In[ ]:




