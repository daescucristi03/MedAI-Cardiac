# Medical Terms Reference – MedAI-Cardiac

This document summarizes all medical terms and clinical variables used in the MedAI-Cardiac project.  
It provides definitions, ranges, units, and diagnostic meanings for both input features and model outputs.

---

# 1. Clinical Input Features

## 1.1 Feature Summary Table

| Variable Name       | Description                                 | Type        | Range / Values         | Unit      |
|--------------------|----------------------------------------------|-------------|-------------------------|-----------|
| age                | Patient age                                  | numeric     | 20–80                  | years     |
| sex                | Biological sex                               | categorical | {0=female, 1=male}     | -         |
| resting_bp         | Resting blood pressure                       | numeric     | 90–200                 | mmHg      |
| chol               | Total cholesterol                            | numeric     | 150–400                | mg/dL     |
| fasting_glucose    | Fasting blood sugar                          | categorical | {0=normal,1=high}      | -         |
| rest_ecg           | Resting electrocardiogram result             | categorical | {0,1,2}                | -         |
| max_hr             | Maximum heart rate during exercise           | numeric     | 80–200                 | bpm       |
| exercise_angina    | Angina triggered by exercise                 | categorical | {0=no,1=yes}           | -         |
| oldpeak            | Exercise-induced ST depression               | numeric     | 0–6                    | mm        |
| st_slope           | Slope of the ST segment during exercise      | categorical | {0=down,1=flat,2=up}   | -         |

---

## 1.2 Detailed Definitions

### **Age**
- Numeric value representing patient age in years.
- Cardiovascular risk increases significantly with age.

---

### **Sex**
- 0 = Female
- 1 = Male
- Male patients tend to have higher risk before age 60.

---

### **Resting Blood Pressure (resting_bp)**
- Systolic blood pressure measured at rest.
- Values >140 mmHg indicate hypertension.

---

### **Cholesterol (chol)**
- Total serum cholesterol level.
- High cholesterol contributes to coronary artery narrowing.

---

### **Fasting Glucose**
- 0 = Normal (<120 mg/dL)
- 1 = High
- High fasting glucose is a risk factor for diabetes and cardiac disease.

---

### **Resting ECG (rest_ecg)**
| Value | Meaning                         |
|-------|---------------------------------|
| 0     | Normal ECG                      |
| 1     | ST-T abnormality                |
| 2     | Left ventricular hypertrophy    |

---

### **Max Heart Rate (max_hr)**
- Peak heart rate during exercise.
- Lower-than-expected max HR may indicate cardiac dysfunction.

---

### **Exercise-Induced Angina**
- 0 = No chest pain during exercise
- 1 = Chest pain present
- Strongly correlated with coronary artery blockages.

---

### **Oldpeak**
- Depression of the ST segment during exercise (vs rest).
- Values >2 mm strongly suggest myocardial ischemia.

---

### **ST Slope**
| Value | Interpretation      | Risk Level |
|-------|----------------------|------------|
| 0     | Downsloping          | High       |
| 1     | Flat                 | Medium     |
| 2     | Upsloping            | Low        |

---

# 2. Output Labels & Risk Predictions

### **Binary Classification (heart_disease)**
- 0 = No disease
- 1 = Disease present

### **Risk Score**
- Probability 0.0–1.0 of heart disease.

### **Risk Level**
| Score        | Category |
|--------------|----------|
| <0.33        | low      |
| 0.33–0.66    | medium   |
| >0.66        | high     |

---

# 3. Multi-Diagnostic Heuristic Outputs

These are **educational** estimates, not clinical diagnostics.

### **CHD – Coronary Heart Disease**
- Narrowing of coronary arteries due to plaque buildup.

### **CAD – Coronary Artery Disease**
- General term for arterial narrowing restricting blood flow.

### **MI – Myocardial Infarction**
- “Heart attack”“ caused by arterial obstruction.

### **Ischemia**
- Inadequate blood supply to the heart muscle.

---

# 4. Explanation Factors

### **Top Factors**
A simple heuristic listing the three highest raw clinical values among:
- cholesterol
- resting blood pressure
- oldpeak

Used for interpretability in inference.

---

# 5. Disclaimer

This document and the entire MedAI-Cardiac system are intended **exclusively for academic and research purposes.**  
They must **not** be used for real medical diagnosis, treatment, or decision-making.