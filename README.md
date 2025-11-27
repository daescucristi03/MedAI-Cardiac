# README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** Universitatea POLITEHNICA București – Facultatea de Inginerie Industrială și Robotică (FIIR)  
**Student:** Daescu Cristian-Narcis  
**Data:** 20 Nov 2025

## Introducere

Acest document descrie activitățile realizate în Etapa 3 a proiectului „Sistem Inteligent de Diagnostic Cardiac bazat pe Rețele Neuronale”. În această etapă au fost analizate, curățate, transformate și structurate datele necesare antrenării modelului neuronal. Scopul este obținerea unui dataset robust, coerent și reproductibil, care să permită instruirea unui model fiabil de clasificare a riscului cardiac.

## 1. Structura Repository-ului Github (Etapa 3)

```
medai-cardiac/
├── README.md
├── docs/
│   └── datasets/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
├── ml/
│   ├── scalers.pkl
│   └── src/
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── inference.py
│       └── model_def.py
├── config/
└── requirements.txt
```

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

Origine: date generate sintetic pentru proiect universitar  
Modul de achiziție: generare programatică (Python – pandas + numpy)  
Motivație: replicarea statistică a distribuțiilor tipice din dataset-urile clinice UCI Heart Disease și Cleveland Clinic  
Perioada: generat în 2025

### 2.2 Caracteristicile dataset-ului

Număr total observații: 5000  
Număr caracteristici: 10 features + 1 label  
Tipuri de date: numerice + categoriale  
Format: CSV  
Structură label: 0 = pacient sănătos, 1 = boală cardiacă

### 2.3 Caracteristici (features)

| Caracteristică | Tip | Unitate | Descriere | Domeniu |
|----------------|-----|---------|-----------|---------|
| age | numeric | ani | vârsta pacientului | 20–80 |
| sex | categorial | - | 0=feminin, 1=masculin | {0,1} |
| resting_bp | numeric | mmHg | tensiune a pacientului în repaus | 90–200 |
| chol | numeric | mg/dl | colesterol total | 150–400 |
| fasting_glucose | categorial | - | glicemie <120 mg/dl | {0,1} |
| rest_ecg | categorial | - | rezultat ECG | {0,1,2} |
| max_hr | numeric | bpm | puls maxim | 80–200 |
| exercise_angina | categorial | - | angină la efort | {0,1} |
| oldpeak | numeric | mm | depresie ST | 0–6 |
| st_slope | categorial | - | pantă ST | {0,1,2} |
| heart_disease | label | - | 0/1 | - |

## 3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive aplicate

- medie, mediană, deviație standard
- interval minim–maxim
- distribuții prin histograme
- outlieri evaluați cu IQR

### 3.2 Calitatea datelor

- fără valori lipsă
- distribuții realiste
- clase moderate echilibrate
- fără anomalii numerice severe

### 3.3 Probleme identificate

- necesitatea standardizării variabilelor numerice
- encoding pentru variabilele categoriale

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor

- eliminare duplicate
- verificare valori aberante
- validare domenii

### 4.2 Transformarea caracteristicilor

- standardizare pentru numeric
- encoding valori categoriale
- conversie într-un array numeric pentru model

### 4.3 Împărțirea datasetului

| Subset | Procent | Mostre |
|--------|---------|---------|
| Train | 70% | 3500 |
| Validation | 15% | 750 |
| Test | 15% | 750 |

Principii:

- stratificare pe label
- fără data-leakage
- scalere calculate doar pe train

### 4.4 Salvări generate

- data/splits/*.csv
- ml/scalers.pkl
- models/best_model.joblib

## 5. Fișiere generate în această etapă

- data/raw/
- data/splits/
- ml/scalers.pkl
- src/preprocessing.py
- README.md

## 6. Stare Etapă

- [x] Structură repository configurată
- [x] Dataset analizat
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată  
