# indicators-for-periodontitis
Identification of important risk indicators for periodontitis based on machine learning approach

All works were done in Jupyter Notebook environment. Following shows essential part of codes in .py format. 

0. health_information_based_features.xlsx
Data file in excel format
Column stands for each feature
ID: Identification number
sex: 1 male 2 female
age: age
incm: Household income 1: Low 2: Middle-low 3: Middle-high 4: High
edu: education 1. Elementary graduation 2. Middle school graduation 3. High school graduation 4. College graduation
occp: occupation 1. Manager, Professionals 2. Office worker 3. Service, Selling management 4. agriculture and forestry and fishing industry 5. Machinary 6. Elementary Workers 7. No job
BP1: Self-rated stress level 1. Very stressed 2. Stressed 3. Slightly stressed 4. Stressless
HE_HP: Blood pressure 1. Normal 2. Pre-hypertension 3. Hypertension
HE_HbA1c	: Glycated hemoglobin*
OR1_2: Dental visit within a year
agegroup: 1: 30~39 2: 40~49 3: 50~59 4: 60~70 5: 70<
marr: marriage 1.Married 2. Single 3. Divorce, Bereavement
chol: HDL cholesterol 0. <240 mg/dL 1. >=240 mg/dL
tg: Triglycerides 0. <200 mg/dL 1. >=200 mg/dL
wc: Abdominal obesity 0: No 1: Yes
glu2: Fasting blood glucose 0: <100 mg/dL 1: ≥100 mg/dL
metabolic: Metabolic syndrome 0: No 1: Yes
bmi: Body mass index (kg/m2)	1. <23 2. 23-25 3.≥25
dm: Diabetes 1. Normal 2. Pre-diabetes 3. Diabetes
hscrp: Hs-CRP (mg/L) 1. <1 mg/L: low risk 2. 1-3 mg/L: intermediate risk 3. >3 mg/L: high risk
smoke: smoking 1. Current smoker 2. Former smoker 3. Non-smoker
BM: Use of oral hygiene products 0. No 1. Yes
brushing: Frequency of toothbrushing per day 1. 1-2 times 2. ≥3 times
BD: Drinking quantity per occasion (glasses) 1. ≤2 2. 3-4 3. 5-6 4. 7-9 5. ≥10
h_drink: Frequency of binge drinking 1. Never 2. <1 time/month 3. 1 time/month 4. 1 time/week 5. Almost everyday
dental1: 0: CPI 0-2 (No periodontitis) 1: CPI 3 non-severe periodontitis 2: CPI 4<= severe periodontitis

1. preprocessing.py
Convert excel file into well arranged numpy array data. 

2. T-SNE.py
Proceed T-SNE analysis and draw figure.

3. model_train_importance.py
Train classifiers and get results: performance, feature importance, SHAP summary plot, SHAP dependence plot 
