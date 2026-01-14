# 📂 Medical Datasets Documentation

Acest document descrie seturile de date utilizate în proiectul **MedAI-Cardiac** pentru antrenarea și validarea modelului de predicție a riscului cardiovascular.

---

## 1. PTB-XL Dataset (Sursa Reală)

**Sursă:** [PhysioNet - PTB-XL ECG Database](https://physionet.org/content/ptb-xl/)  
**Versiune:** 1.0.3  
**Licență:** Open Data (CC BY 4.0)

### Descriere
PTB-XL este un set de date mare, disponibil public, care conține **21,837 înregistrări clinice EKG** de la 18,885 pacienți, colectate pe o perioadă de 7 ani.

### Caracteristici Tehnice
- **Format:** WFDB (Waveform Database) - standard industrial.
  - `.dat`: Fișier binar conținând semnalul brut.
  - `.hea`: Fișier text (header) conținând metadate (vârstă, sex, diagnostic, parametri tehnici).
- **Canale:** 12 derivații standard (I, II, III, aVR, aVL, aVF, V1-V6).
- **Frecvență de eșantionare:** 500 Hz (High Resolution).
- **Durată:** 10 secunde per înregistrare.

### Etichetare (Labels)
Diagnosticele sunt codificate conform standardului **SCP-ECG**. Proiectul nostru se concentrează pe super-clasa **MI (Myocardial Infarction)**.

| Cod SCP | Descriere | Clasa Noastră |
|---------|-----------|---------------|
| NORM | Normal ECG | 0 (Low Risk) |
| MI | Myocardial Infarction | 1 (High Risk) |
| STTC | ST/T Change | 1 (High Risk - potențial) |
| CD | Conduction Disturbance | 0 (Low Risk - ignorat în faza 1) |
| HYP | Hypertrophy | 0 (Low Risk - ignorat în faza 1) |

---

## 2. MedAI Synthetic Dataset (Generator Propriu)

Pentru a compensa dezechilibrul de clase (infarctele sunt evenimente rare în populația generală) și pentru a testa robustețea modelului, am dezvoltat un **Generator de Semnal EKG Sintetic**.

**Locație Cod:** `src/modules/ecg_processor.py` (funcția `generate_advanced_ecg`)

### Metodologie
Semnalul este generat matematic folosind o sumă de funcții Gaussiene pentru a modela complexul P-QRS-T, cu parametri ajustabili pentru a simula patologii.

### Parametri Controlabili
1.  **Heart Rate (BPM):** 40 - 140 BPM.
2.  **Noise Level:** Zgomot Gaussian aditiv pentru a simula interferențe musculare/electrice.
3.  **ST Displacement:**
    *   `> 0.1 mV`: Simulează **STEMI** (ST-Elevation Myocardial Infarction) - Infarct Acut.
    *   `< -0.1 mV`: Simulează ischemie (ST Depression).
4.  **T-Wave Amplitude:**
    *   Valori negative simulează inversarea undei T (semn de ischemie).

### Rol în Proiect
- **Augmentare:** Creșterea numărului de exemple pozitive (Infarct) în setul de antrenare.
- **Validare:** Testarea capacității modelului de a detecta anomalii specifice (ex: doar ST elevation, fără alte modificări).

---

## 3. Structura Datelor Procesate

După preprocesare (`src/preprocessing/prepare_dataset.py`), datele sunt salvate într-un format optimizat pentru PyTorch.

**Locație:** `data/processed/`

| Fișier | Format | Dimensiuni | Descriere |
|--------|--------|------------|-----------|
| `X_data.npy` | NumPy Binary | `(N, 5000, 12)` | Tensorul de intrare (Semnale EKG normalizate Z-score). |
| `y_labels.pkl` | Pickle | `(N,)` | Vectorul de etichete (0 sau 1). |

### Pipeline de Preprocesare
1.  **Încărcare:** Citire WFDB (real) sau Generare (sintetic).
2.  **Filtrare:** Filtru Butterworth High-Pass (0.5 Hz) pentru eliminarea derivei liniei de bază (baseline wander).
3.  **Normalizare:** Standardizare (Z-score) per canal: `x' = (x - mean) / std`.
4.  **Formatare:** Transpunere pentru PyTorch (Channels-First vs Time-First).

---

## 4. Confidențialitate și Etică

- Datele PTB-XL sunt anonimizate la sursă.
- Datele sintetice nu corespund niciunei persoane reale.
- Sistemul MedAI-Cardiac stochează datele pacienților (nume, CNP) doar local/în baza de date privată a spitalului (simulată prin MongoDB), respectând principiile GDPR prin design (acces securizat, audit trail).
