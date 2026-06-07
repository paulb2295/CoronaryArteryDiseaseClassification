# 🫀 Coronary Artery Disease Classification

A machine learning project for the prediction of **Coronary Artery Disease (CAD)** using clinical, laboratory, electrocardiographic, and echocardiographic patient data.

Dataset: https://www.kaggle.com/datasets/saeedeheydarian/classification-of-coronary-artery-disease

The project explores the complete machine learning pipeline, including:

* Data preprocessing and feature engineering
* Exploratory Data Analysis (EDA)
* Feature selection and multicollinearity reduction
* Patient segmentation using unsupervised learning
* Classification using multiple machine learning models
* Ensemble prediction
* Interactive deployment with Gradio

---

## 📋 Project Overview

Coronary Artery Disease (CAD) remains one of the leading causes of morbidity and mortality worldwide. Early identification of high-risk patients can support clinical decision-making and improve patient outcomes.

This project aims to develop a machine learning-based decision support system capable of predicting the presence of CAD using routinely collected clinical data.

The workflow combines:

1. **Unsupervised Learning**

   * K-Means clustering for patient segmentation
   * Identification of patient archetypes

2. **Supervised Learning**

   * Polynomial Logistic Regression
   * Decision Tree
   * Random Forest
   * XGBoost

3. **Ensemble Learning**

   * Soft voting based on model probabilities
   * Improved robustness and predictive performance

---

## 📊 Dataset

The dataset contains demographic, clinical, laboratory, ECG, and echocardiographic features such as:

### Demographics

* Age
* Sex
* BMI

### Cardiovascular Risk Factors

* Current smoker
* Ex-smoker
* Family history of CAD

### Clinical Examination

* Blood pressure
* Pulse rate
* Edema
* Weak peripheral pulse
* Lung rales
* Cardiac murmurs

### Symptoms

* Typical chest pain
* Atypical chest pain
* Dyspnea
* CCS Functional Class

### Electrocardiography (ECG)

* Q wave
* ST elevation
* ST depression
* T wave inversion
* Left ventricular hypertrophy
* Poor R progression

### Laboratory Data

* Fasting blood sugar
* Creatinine
* Triglycerides
* LDL cholesterol
* HDL cholesterol
* Blood urea nitrogen
* ESR

### Echocardiography

* Ejection fraction
* Regional wall motion abnormalities
* Valvular heart disease

### Target Variable

* `cath`

  * 0 = No CAD
  * 1 = CAD present

---

## 🔬 Methodology

### Data Preprocessing


*Feature Selection and Multicollinearity Reduction

Prior to model development, the dataset was examined for constant features and highly correlated variables. Features with no variability, such as exertional cp, were removed because they do not provide any predictive information.

Several pairs of variables exhibited strong correlations, indicating potential redundancy. Examples include bp and htn, fbs and dm, bmi and obesity, lung rales and ef-tte, systolic murmur and vhd, as well as lymph and neut. Retaining multiple highly correlated features can introduce multicollinearity, increase model complexity, and reduce interpretability without substantially improving predictive performance.

To address this, a subset of variables was selected based on three criteria:

Clinical relevance to coronary artery disease and cardiovascular risk assessment.
Reduction of redundant information by retaining the most informative variable from highly correlated feature groups.
Preservation of key demographic, clinical, electrocardiographic, laboratory, and echocardiographic information required for downstream analysis.

The resulting feature set includes demographic characteristics (age, sex, bmi), cardiovascular risk factors (current smoker, ex-smoker, fh), clinical examination findings (bp, pr, edema, weak peripheral pulse, lung rales, murmurs), symptom-related variables (typical chest pain, atypical, dyspnea, function class), ECG findings (q wave, st elevation, st depression, tinversion, lvh, poor r progression), laboratory measurements (fbs, cr, tg, ldl, hdl, bun, esr), echocardiographic indicators (ef-tte, region rwma, vhd), and the target variable (cath).

This selection balances predictive information and feature independence, reducing redundancy while maintaining clinically meaningful variables.

* Correlation analysis
![Correlation Map](image-1.png)

* Generating synthetic data for healthy patients
Initial dataset had an uneven distribution between healthy people and CAD patients
![Initial data set cath distribution](image-2.png)

Using CTGANSynthesizer model 129 synthetic data for healthy patients were created, balancing the distribution: 50% healthy people, 50% CAD patients.
Real healthy people vs Synthetic healthy people distributions:
![Real healthy people vs Synthetic healthy people distributions](image-3.png)

### Patient Segmentation

Patients are grouped into three clusters using K-Means clustering.

![Patient Clusters](image-4.png)

The resulting cluster assignment is transformed into:

* `cluster0`
* `cluster1`
* `cluster2`
* `cluster3`

to be used as additional predictive features.

### Classification Models

The following models were trained and evaluated:

* Polynomial Logistic Regression
A pipeline for polynomial Logistic regression was created following these steps:
1. Outlier Capping
2. Polynomial Generation
3. Scaling using StandardScaler()
4. Regularization

* Decision Tree
* Random Forest
* XGBoost

For all trained models, GridSearchCV was used to obtain the best set of hyperparameters for each model for the problem at hand.
For validation, StratifiedKFold with 5 splits was used for all trained models.

### Ensemble Model

Final prediction is obtained using soft voting:

```python
combined_prob = (logistic_probability + xgboost_probability) / 2
prediction = (combined_prob >= 0.18)
```
The ensemble proved to be inferior to using XGBoost, which is why XGBoost model was chosen for the final application.

---

## 📈 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* F1 Score
* ROC Curve
* Area Under the Curve (AUC)

Confusion matrices are used to analyze:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

Best Model: XGBoost with custom threshold of 0.13
This model was chosen because it has Recall 100% an 0 False neagatives. It also had the best Precision of all models.

![alt text](image.png)

## 🚀 Gradio Application

A Gradio web application is included to allow interactive testing of the trained models.

Users can input:

* Demographic information
* Symptoms
* Physical examination findings
* ECG findings
* Laboratory values
* Echocardiographic measurements

The application:

1. Creates a patient feature vector
2. Assigns the patient to a cluster
3. Generates ensemble predictions
4. Returns:

   * CAD probability
   * CAD classification

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/paulb2295/CoronaryArteryDiseaseClassification.git

cd CoronaryArteryDiseaseClassification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Open the notebooks in Jupyter:

```bash
jupyter notebook
```

Run the notebooks sequentially:

1. data_analysis_preprocessing
2. models_training
3. models_evaluation
4. cad_gui

---

## 📁 Repository Structure

```text
CoronaryArteryDiseaseClassification/
│
├── data/
│   ├── CAD.csv
│   └── preprocessed_cad.csv
│
├── notebooks/
│   ├── data_analysis_preprocessing
│   ├── models_training
│   ├── models_evaluation
│   └── cad_gui [gradio application]
│
├── models/
│   ├── poly_log_reg
│   ├── ...
│   └── xgb_clf
│
├── requirements.txt
└── README.md
```

---

## ⚠ Disclaimer

This project is intended for educational and research purposes only.

The predictions generated by the models are not intended to replace professional medical judgment, diagnosis, or treatment decisions.

Any clinical use should be performed only after extensive external validation and regulatory approval.


## 👨‍💻 Author

**Paul Borlea**

GitHub: https://github.com/paulb2295
