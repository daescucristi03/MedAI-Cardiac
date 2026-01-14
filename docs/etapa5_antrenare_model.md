# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Daescu Cristian
**Link Repository GitHub:** https://github.com/daescucristi03/MedAI-Cardiac
**Data predării:** 14.01.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape.

**Obiectiv principal:** Antrenarea efectivă a modelului ResNet-CNN-LSTM definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă MedAI-Cardiac.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset (realizat prin Generatorul Sintetic)

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție ≥40% date originale** în `data/generated/`
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită (`src/neural_network/model.py`)
- [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

Am utilizat generatorul sintetic pentru a crea un dataset echilibrat, combinând datele reale PTB-XL cu date sintetice care conțin patologii specifice (ST Elevation, T-Wave Inversion).

**Procesul de pregătire:**
1.  **Generare:** Scriptul `src/preprocessing/prepare_dataset.py` a fost actualizat pentru a include date sintetice.
2.  **Curățare:** Aplicare filtru Butterworth High-Pass (0.5Hz) pentru eliminarea baseline wander.
3.  **Normalizare:** Z-score normalization (StandardScaler).
4.  **Split:** Stratified Split 70% Train / 15% Val / 15% Test.

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

1. **Antrenare model** definit în Etapa 4 pe setul final de date.
2. **Minimum 10 epoci**, batch size 32.
3. **Împărțire stratificată** train/validation/test: 70% / 15% / 15%.
4. **Tabel justificare hiperparametri** (vezi mai jos).
5. **Metrici calculate pe test set:**
   - **Acuratețe:** ~89%
   - **F1-score (macro):** ~0.87
6. **Salvare model antrenat** în `src/neural_network/saved_model.pth`.
7. **Integrare în UI din Etapa 4:** UI-ul încarcă acum modelul antrenat și oferă predicții reale.

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare standard pentru AdamW, oferă un echilibru bun între viteza de convergență și stabilitate. |
| Batch size | 32 | Compromis optim între stabilitatea gradientului și utilizarea memoriei GPU. |
| Number of epochs | 30 | Suficient pentru convergență datorită arhitecturii ResNet. |
| Optimizer | AdamW | Gestionează bine gradienții sparși și include Weight Decay pentru regularizare. |
| Loss function | BCELoss | Binary Cross Entropy este standardul pentru clasificare binară (Infarct vs Normal). |
| Activation functions | ReLU (CNN), Sigmoid (Output) | ReLU previne vanishing gradient în rețele adânci; Sigmoid mapează output-ul la probabilitate [0,1]. |

---

### Nivel 2 – Recomandat (85-90% din punctaj)

1. **Early Stopping** - Implementat (patience=5).
2. **Scheduler** - `ReduceLROnPlateau` pentru ajustarea fină a ratei de învățare.
3. **Augmentări relevante domeniu:**
   - **Zgomot Gaussian:** Adăugat în generator pentru a simula interferențe electrice.
   - **Baseline Wander:** Adăugat sinusoidal de joasă frecvență.
4. **Analiză erori context industrial** (vezi mai jos).

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

### 1. Pe ce clase greșește cel mai mult modelul?
Modelul tinde să aibă **False Negatives** (nu detectează infarctul) în cazurile de **Infarct Non-STEMI** (unde nu există supradenivelare ST evidentă, ci doar modificări subtile ale undei T). Aceste semnale sunt foarte similare cu cele normale pentru un CNN standard.

### 2. Ce caracteristici ale datelor cauzează erori?
Zgomotul de înaltă frecvență (EMG - activitate musculară) poate masca undele P mici sau poate crea artefacte care seamănă cu o undă R, derutând algoritmul. De asemenea, baseline wander-ul puternic nefiltrat poate fi interpretat greșit ca ST elevation.

### 3. Ce implicații are pentru aplicația industrială?
**False Negatives sunt CRITICE.** Dacă modelul spune "Normal" unui pacient cu infarct, acesta poate muri.
**False Positives sunt acceptabile.** Dacă modelul spune "Infarct" unui pacient sănătos, acesta va face investigații suplimentare (enzime cardiace), ceea ce costă timp/bani, dar nu viața.
**Prioritate:** Maximizarea Recall-ului (Sensibilității), chiar cu prețul scăderii Preciziei.

### 4. Ce măsuri corective propuneți?
1.  **Logică Hibridă:** Am implementat un sistem de vot care combină predicția AI cu o analiză euristică a segmentului ST. Dacă euristica detectează o anomalie gravă, suprascrie AI-ul.
2.  **Ajustarea pragului de decizie:** Scăderea pragului de la 0.5 la 0.3 pentru clasa "Infarct" pentru a crește sensibilitatea. (Implementat prin slider-ul "Sensitivity" în UI).
3.  **Augmentare specifică:** Antrenarea cu mai multe exemple de Non-STEMI și zgomot muscular simulat.

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența respectă fluxul din State Machine:

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Generatorul din `src/modules/ecg_processor.py` produce date sintetice sau încarcă CSV. |
| `PREPROCESS` | `SignalCleaner` aplică aceleași filtre ca la antrenare. |
| `RN_INFERENCE` | `model.py` încarcă `saved_model.pth` și face forward pass real. |
| `DISPLAY` | UI-ul afișează riscul calculat și Harta de Saliență (XAI). |

---

## Structura Repository-ului la Finalul Etapei 5

```
cardio_risk_project/
├── docs/
│   ├── etapa5_antrenare_model.md      # ← ACEST FIȘIER
│   ├── state_machine.png              # Din Etapa 4
│   └── screenshots/
│       └── inference_real.png         # Demo UI cu model antrenat
├── src/
│   ├── neural_network/
│   │   ├── model.py                   # Arhitectura ResNet-LSTM
│   │   ├── train_model.py             # Script antrenare
│   │   ├── evaluate_model.py          # Script evaluare
│   │   └── saved_model.pth            # Modelul ANTRENAT
│   └── app.py                         # UI Principal
├── results/                            
│   └── training_logs.txt              # Log-uri antrenare
└── requirements.txt
```

---

## Instrucțiuni de Rulare

### 1. Antrenare model (Dacă doriți re-antrenare)

```bash
python src/train_model.py
```

### 2. Evaluare pe test set

```bash
python src/evaluate_model.py
```

### 3. Lansare UI cu model antrenat

```bash
streamlit run src/app.py
```

---

**Mult succes! Această etapă demonstrează că Sistemul MedAI-Cardiac funcționează în condiții reale!**
