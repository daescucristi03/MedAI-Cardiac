# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Daescu Cristian
**Link Repository GitHub:** https://github.com/daescucristi03/MedAI-Cardiac
**Data predării:** 14.01.2025

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape.

**Obiectiv principal:** Maturizarea completă a Sistemului MedAI-Cardiac prin optimizarea modelului CNN-LSTM, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**CONTEXT IMPORTANT:** 
- Aceasta este **ULTIMA VERSIUNE înainte de examen**.
- Proiectul este **COMPLET și FUNCȚIONAL**.

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

- [x] **Model antrenat** salvat în `src/neural_network/saved_model.pth`
- [x] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [x] **Tabel hiperparametri** cu justificări completat
- [x] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [x] **State Machine** implementat conform definiției din Etapa 4

---

## Cerințe

### Tabel Experimente de Optimizare

Am realizat **4 experimente** cu variații sistematice pentru a îmbunătăți performanța modelului:

| **Exp#** | **Modificare față de Baseline (Etapa 5)** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| Baseline | CNN-LSTM, LR=0.001, Batch=16 | 0.82 | 0.79 | 10 min | Referință (Etapa 5) |
| Exp 1 | Adăugare Dropout 0.5 în LSTM | 0.84 | 0.81 | 11 min | Reducere overfitting, generalizare mai bună |
| Exp 2 | Creștere neuroni LSTM (128 -> 256) | 0.83 | 0.80 | 15 min | Îmbunătățire marginală, cost computațional mare |
| Exp 3 | Learning Rate Scheduler (StepLR) | 0.86 | 0.83 | 12 min | Convergență mai fină în ultimele epoci |
| Exp 4 | **Logică Hibridă (AI + Heuristic)** | **0.92** | **0.90** | **N/A** | **BEST** - Combină AI cu reguli clinice (ST-Analysis) |

**Justificare alegere configurație finală:**
Am ales **Exp 4** (Model Hibrid) ca soluție finală pentru că:
1. Oferă cel mai bun **F1-score (0.90)**, critic în medicină.
2. **Logica Hibridă** compensează limitările datelor de antrenare prin aplicarea unor reguli euristice (analiza segmentului ST) peste predicția AI.
3. **Polarizarea Probabilităților**: Am implementat o funcție de post-procesare care crește încrederea modelului (Confidence) pentru cazurile clare, oferind un feedback mai ferm medicului.

---

## 1. Actualizarea Aplicației Software în Etapa 6 

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `saved_model.pth` (Baseline) | `saved_model.pth` + Heuristic Logic | +10% accuracy prin ensemble learning |
| **Threshold alertă** | 0.5 (fix) | Dinamic (bazat pe Sensitivity) | Adaptare la contextul clinic (screening vs diagnostic) |
| **Explainability (XAI)** | Hărți de saliență simple | Hărți de saliență + Confidence Score | Feedback vizual și numeric mai clar |
| **UI - Feedback** | Text simplu | Carduri vizuale (High/Low Risk) + Metrici clinice | Interfață profesională, stil "Medical Dashboard" |
| **Management Pacienți** | Inexistent | Sistem complet EHR (MongoDB) | Necesitate critică pentru un produs real |

### Diagrama State Machine Actualizată

Fluxul a fost rafinat pentru a include pasul de **Explainability (XAI)** și **Salvare în Cloud**:

`PREPROCESS` → `RN_INFERENCE` → `HEURISTIC_CHECK` → `CONFIDENCE_SCALING` → `DISPLAY` → `SAVE_TO_DB`

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**
- **Clasa "Normal":** Precision 94%, Recall 92%. Modelul recunoaște foarte bine ritmul sinusal normal.
- **Clasa "Infarct (MI)":** Precision 88%, Recall 91%.
- **Observație:** Recall-ul pentru Infarct a crescut semnificativ datorită logicii hibride care penalizează anomaliile ST clare pe care AI-ul le-ar putea rata.

### 2.2 Analiza Detaliată a 5 Exemple Greșite

| **Index** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
| #12 | Infarct | Normal | 0.42 | Infarct Non-STEMI (fără ST elevation clar) | Antrenare pe mai multe date Non-STEMI |
| #45 | Normal | Infarct | 0.65 | Zgomot puternic (baseline wander) interpretat ca ST elevation | Filtrare mai agresivă (High-pass 0.6Hz) |
| #88 | Infarct | Normal | 0.38 | Unda T inversată subtil | Creșterea ponderii pentru T-wave inversion în Loss |
| #102 | Normal | Infarct | 0.55 | Artefacte musculare (EMG) | Augmentare cu zgomot EMG în antrenare |
| #156 | Infarct | Normal | 0.48 | Ritm cardiac foarte rapid (Tahicardie) maschează ST | Normalizare temporală (resampling) bazată pe R-R interval |

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

**Abordare:** Hybrid Ensemble (AI + Rule-based).

**Axe de optimizare explorate:**
1.  **Augmentare:** Generarea de semnale sintetice cu patologii specifice.
2.  **Arhitectură:** ResNet-CNN pentru extragerea trăsăturilor locale fine.
3.  **Post-procesare:** Polarizarea probabilităților pentru a crește încrederea deciziei.

### 3.3 Raport Final Optimizare

**Model baseline (Etapa 5):**
- Accuracy: 0.82
- F1-score: 0.79

**Model optimizat (Etapa 6 - Hibrid):**
- Accuracy: 0.92 (+10%)
- F1-score: 0.90 (+11%)

**Configurație finală aleasă:**
- Arhitectură: ResNet-CNN-LSTM
- Logică: Ensemble (40% AI, 40% ST-Analysis, 20% HR-Analysis)
- Epoci: 30 (Early Stopping)

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target Industrial** | **Status** |
|-------------|-------------|-------------|-------------|----------------------|------------|
| Accuracy | ~50% (Random) | 82% | 92% | ≥90% | **ATINS** |
| F1-score (macro) | ~0.50 | 0.79 | 0.90 | ≥0.85 | **ATINS** |
| Recall (Infarct) | N/A | 75% | 91% | ≥90% | **ATINS** |
| Latență inferență | 50ms | 45ms | 48ms | ≤100ms | **OK** |

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Performanței Finale

**Obiective atinse:**
- [x] Model RN funcțional cu accuracy >90% (în mod hibrid).
- [x] Integrare completă în aplicație software (EHR, Generator, Model, UI).
- [x] Sistem Explainable AI (XAI) funcțional.
- [x] Pipeline end-to-end robust la zgomot moderat.

**Obiective parțial atinse:**
- [ ] Detecția infarctelor Non-STEMI rămâne o provocare fără date reale masive.

### 5.2 Limitări Identificate

1. **Limitări date:** Datele reale (PTB-XL) sunt dezechilibrate. Deși am compensat cu date sintetice, "gap-ul" de realitate există.
2. **Limitări model:** Arhitectura este complexă, necesitând resurse GPU pentru antrenare rapidă.
3. **Limitări validare:** Testarea s-a făcut pe un subset hold-out și date sintetice.

### 5.3 Direcții de Cercetare și Dezvoltare

**Pe termen scurt (1-3 luni):**
1. Colectare de date reale specifice pentru Non-STEMI.
2. Exportarea modelului în format ONNX pentru inferență mai rapidă.

**Pe termen mediu (3-6 luni):**
1. Integrarea cu dispozitive EKG portabile (prin Bluetooth).
2. Implementarea unui sistem de alertare automată prin SMS/Email pentru medici.

### 5.4 Lecții Învățate

**Tehnice:**
1. **Hybrid AI > Pure AI (pe date puține).** Combinarea rețelelor neuronale cu reguli clinice clasice a oferit cea mai bună performanță și robustețe.
2. **Explainability este crucial.** Într-un domeniu medical, un model "Black Box" este inutil. Hărțile de saliență au adăugat o valoare imensă proiectului.

**Proces:**
1. Dezvoltarea iterativă a permis rafinarea continuă a interfeței și a logicii de business.
2. Feedback-ul vizual rapid (prin Streamlit) a accelerat procesul de debugging.

---

## Structura Repository-ului la Finalul Etapei 6

```
cardio_risk_project/
├── docs/
│   ├── etapa6_optimizare_concluzii.md  # ← ACEST FIȘIER
│   ├── etapa5_antrenare_model.md       # Din Etapa 5
│   ├── confusion_matrix_optimized.png  # Matricea de confuzie finală
│   └── screenshots/
│       ├── inference_optimized.png     # Demo UI final
│       └── ui_demo.png                 # Din Etapa 4
├── src/
│   ├── neural_network/
│   │   ├── model.py                    # Arhitectura Finală
│   │   ├── train_model.py              # Script antrenare optimizat
│   │   ├── evaluate_model.py           # Script evaluare detaliată
│   │   └── saved_model.pth             # Modelul OPTIMIZAT
│   ├── modules/                        # Module Backend
│   └── app.py                          # UI Final cu XAI și Generator
├── results/
│   └── final_metrics.json              # Metrici finale
└── requirements.txt
```

---

**Mult succes! Această etapă încheie dezvoltarea Sistemului MedAI-Cardiac!**
