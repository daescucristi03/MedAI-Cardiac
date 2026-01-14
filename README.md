# ❤️ MedAI-Cardiac: Cardiovascular Risk Prediction System

**Student:** Daescu Cristian  
**Instituție:** POLITEHNICA București – FIIR  
**Disciplina:** Rețele Neuronale  
**Status:** Finalizat (Etapa 6)

---

## 📋 Prezentare Generală

**MedAI-Cardiac** este un sistem complet de Inteligență Artificială (SIA) conceput pentru a asista medicii în detectarea rapidă a riscului de **Infarct Miocardic (MI)** pe baza semnalelor EKG.

Sistemul integrează un pipeline complet de date, o rețea neuronală hibridă **CNN-LSTM** și o interfață grafică modernă cu funcționalități de **Explainable AI (XAI)** și **Electronic Health Record (EHR)**.

### 🚀 Funcționalități Cheie
- **Analiză EKG 12-Lead:** Procesează semnale complexe pentru a detecta anomalii subtile.
- **Logică Hibridă (AI + Heuristic):** Combină Deep Learning cu reguli clinice (ST-segment analysis) pentru robustețe maximă.
- **Generator de Date Sintetice:** Simulează patologii cardiace (ST-Elevation, T-Wave Inversion) pentru testare.
- **Explainable AI (XAI):** Utilizează Hărți de Saliență pentru a evidenția vizual zonele suspecte pe graficul EKG.
- **Platformă Multi-User:** Autentificare medici, profil editabil și istoric activitate.
- **Dosar Pacient (EHR):** Gestionare pacienți, istoric medical și rapoarte PDF profesionale.
- **Cloud Database:** Stocare securizată în MongoDB Atlas.

---

## 🏗️ Arhitectura Sistemului

Proiectul este structurat modular, urmând fluxul unui sistem industrial de AI:

1.  **Data Acquisition:** Descărcare date reale (PTB-XL) + Generare date sintetice.
2.  **Preprocessing:** Filtrare (Butterworth High-Pass), Normalizare Z-score.
3.  **Neural Network:**
    *   **ResNet-CNN:** Extrage trăsături spațiale locale (forma undei QRS) cu conexiuni reziduale.
    *   **Bidirectional LSTM:** Analizează evoluția temporală a bătăilor inimii.
4.  **UI & Deployment:** Aplicație Streamlit cu design modern (Glassmorphism).

---

## 📂 Structura Repository-ului

```
cardio_risk_project/
├── docs/                          # Documentație detaliată per etapă
│   ├── etapa4_arhitectura_SIA.md  # Arhitectura și State Machine
│   ├── etapa5_antrenare_model.md  # Procesul de antrenare și metrici
│   ├── etapa6_optimizare_concluzii.md # Optimizare finală și concluzii
│   ├── state_machine.png          # Diagrama stărilor
│   └── screenshots/               # Capturi de ecran din aplicație
│
├── src/                           # Codul sursă
│   ├── app.py                     # Aplicația Principală (UI + Logică)
│   ├── neural_network/            # Modulul AI
│   │   ├── model.py               # Definiția arhitecturii CNN-LSTM
│   │   ├── train_model.py         # Script antrenare
│   │   ├── evaluate_model.py      # Script evaluare
│   │   └── saved_model.pth        # Modelul antrenat și optimizat
│   ├── preprocessing/             # Modulul de procesare date
│   │   ├── signal_cleaner.py      # Filtrare și normalizare
│   │   └── prepare_dataset.py     # Pipeline de date
│   ├── data_acquisition/          # Scripturi descărcare date
│   └── modules/                   # Module auxiliare
│       ├── database.py            # Conexiune MongoDB
│       ├── report.py              # Generator PDF
│       ├── ui_style.py            # CSS și Design
│       └── ecg_processor.py       # Logică business EKG
│
├── data/                          # Folder date (ignorat de git, structură locală)
│   ├── raw/
│   └── processed/
│
├── requirements.txt               # Dependențe Python
└── README.md                      # Acest fișier
```

---

## 💻 Instalare și Rulare

### 1. Clonare Repository
```bash
git clone https://github.com/daescucristi03/MedAI-Cardiac.git
cd MedAI-Cardiac
```

### 2. Configurare Mediu Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalare Dependențe
```bash
pip install -r requirements.txt
```

### 4. Rulare Aplicație (Dashboard)
Aceasta va deschide interfața web în browser.
```bash
streamlit run src/app.py
```

---

## 📊 Performanță (Etapa 6)

Modelul final optimizat a obținut următoarele rezultate pe setul de testare:

| Metrică | Valoare | Observații |
|---------|---------|------------|
| **Accuracy** | **89%** | Îmbunătățire semnificativă prin augmentare |
| **F1-Score** | **0.87** | Echilibru excelent între Precision și Recall |
| **Recall (Infarct)** | **88%** | Critic pentru a nu rata cazurile grave |
| **Latență** | **48ms** | Potrivit pentru analiză în timp real |

---

## 📜 Licență și Credite

Acest proiect a fost dezvoltat ca parte a cursului de **Rețele Neuronale** la **POLITEHNICA București**.
Setul de date utilizat pentru antrenare include subseturi din **PTB-XL** (PhysioNet).

---
**Contact:** Daescu Cristian
