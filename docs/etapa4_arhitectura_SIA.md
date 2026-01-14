# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Daescu Cristian
**Link Repository GitHub:** https://github.com/daescucristi03/MedAI-Cardiac
**Data:** 14.01.2025
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape.

**Sistemul livrat este un SCHELET COMPLET și FUNCȚIONAL al proiectului MedAI-Cardiac.**
- **Pipeline funcțional:** Generare/Upload Date → Preprocesare → Inferență Hibridă → Vizualizare & XAI.
- **Model definit:** Arhitectura hibridă ResNet-CNN-LSTM este compilată și integrată.
- **Interfață:** Dashboard-ul Streamlit permite interacțiunea completă cu sistemul, incluzând managementul pacienților.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Detectarea rapidă a Infarctului Miocardic (MI) în zone fără cardiolog | Analiză automată EKG 12-lead → Predicție risc în < 1 secundă | `src/modules/ecg_processor.py` (Inference) |
| Reducerea erorilor de diagnostic prin lipsa de transparență a AI ("Black Box") | Hărți de Saliență (XAI) → Evidențierea vizuală a segmentului ST patologic pe grafic | `src/modules/ecg_processor.py` (Explainability) |
| Gestionarea eficientă a datelor pacienților și istoric medical | Bază de date Cloud (MongoDB) pentru stocarea profilurilor și analizelor | `src/modules/database.py` + `src/app.py` |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Strategie:** Proiectul utilizează setul de date public **PTB-XL** pentru antrenarea de bază, însă pentru a asigura robustețea și echilibrarea claselor (infarctele sunt evenimente rare), am dezvoltat un **Generator de Semnal EKG Sintetic**.

**Total observații finale:** ~22,000 (estimat final)
**Observații originale:** ~9,000 (41%) - Generate sintetic

**Tipul contribuției:**
[X] Date generate prin simulare fizică (Modelare matematică a undelor P-QRS-T)
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
Am implementat un simulator matematic în `src/modules/ecg_processor.py` (funcția `generate_advanced_ecg`) care modelează activitatea electrică a inimii folosind o serie de funcții Gaussiene. 
- **Originalitate:** Simulatorul permite injectarea controlată a patologiilor specifice:
    - *Supradenivelare de segment ST (STEMI)*: Parametru ajustabil pentru a simula infarctul acut.
    - *Inversarea undei T*: Pentru simularea ischemiei.
    - *Zgomot și Baseline Wander*: Pentru a simula condiții reale de achiziție.

**Locația codului:** `src/modules/ecg_processor.py`
**Locația datelor:** `data/processed/` (mix de date reale și sintetice la runtime)

---

### 3. Diagrama State Machine a Întregului Sistem

**Locație:** `docs/state_machine.png` (Vă rog să vizualizați fișierul din folderul docs)

**Justificarea State Machine-ului ales:**
Am ales o arhitectură de tip **Interactive Diagnostic Support System**. Sistemul nu rulează în buclă infinită autonomă, ci așteaptă input-ul utilizatorului (medic/student), procesează cererea și oferă un rezultat explicabil.

**Stările principale sunt:**
1. **IDLE:** Sistemul așteaptă input în Dashboard.
2. **AUTHENTICATION:** Medicul se loghează pentru acces securizat.
3. **PATIENT_SELECTION:** Selectarea sau crearea unui profil de pacient.
4. **INPUT_ACQUISITION:** Generare sintetică sau Upload fișier CSV.
5. **PREPROCESS:** Curățare semnal (filtru Butterworth) și normalizare Z-score.
6. **INFERENCE:** Rularea modelului CNN-LSTM + Analiză Euristică ST.
7. **DISPLAY:** Afișarea graficului, riscului și generarea raportului PDF.

---

### 4. Scheletul Complet al celor 3 Module

| **Modul** | **Implementare** | **Status Funcțional** |
|-----------|------------------|-----------------------|
| **1. Data Acquisition** | `src/modules/ecg_processor.py` & `src/data_acquisition/download_data.py` | ✅ Funcțional. Download PTB-XL + Generator Sintetic. |
| **2. Neural Network** | `src/neural_network/model.py` (ResNet-LSTM) | ✅ Funcțional. Arhitectura este definită, compilată și modelul poate fi salvat/încărcat. |
| **3. Web Service / UI** | `src/app.py` (Streamlit) | ✅ Funcțional. Interfață Modernă, Multi-User, Cloud DB. |

#### Detalii per modul:

**Modul 1: Data Logging / Acquisition**
- Scriptul `download_data.py` folosește biblioteca `wfdb` pentru a accesa PhysioNet.
- Generatorul din `ecg_processor.py` creează semnale 12-lead matematice.

**Modul 2: Neural Network Module**
- **Arhitectură:** ResNet-CNN (3 straturi reziduale) + Bidirectional LSTM + Fully Connected.
- **Justificare:** EKG-ul este o serie de timp (LSTM) dar forma undei locale (QRS) contează enorm (CNN).

**Modul 3: Web Service / UI**
- Construit cu **Streamlit**.
- Include:
    - **EHR:** Management pacienți și istoric medical.
    - **Securitate:** Login medici cu parole criptate.
    - **Raportare:** Generare PDF cu date complete.

---

## Structura Repository-ului

```
cardio_risk_project/
├── data/
│   ├── raw/               # Date PTB-XL descărcate
│   ├── processed/         # Date transformate (.npy)
├── src/
│   ├── data_acquisition/  # Scripturi download
│   ├── preprocessing/     # SignalCleaner
│   ├── neural_network/    # model.py, dataset.py
│   ├── modules/           # Module backend (DB, Report, Logic)
│   └── app.py             # UI Principal
├── docs/
│   ├── state_machine.png  # Diagrama stărilor
│   └── screenshots/       # Demo UI
├── README.md              # General
├── README_Etapa4_Arhitectura_SIA.md # Acest fișier
└── requirements.txt
```

---

## Instrucțiuni de Rulare

1. **Instalare dependențe:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pornire Aplicație (UI + Generator + Model):**
   ```bash
   streamlit run src/app.py
   ```

3. **Antrenare Model (Opțional):**
   ```bash
   python src/train_model.py
   ```
