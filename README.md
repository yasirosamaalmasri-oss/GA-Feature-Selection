
```
# 🧬 Genetic Algorithm Feature Selection (NSGA-II)
```
This project implements a **Genetic Algorithm (NSGA-II)** to automatically select the most relevant features from large datasets.  
The goal is to improve model accuracy, reduce training time, and prevent overfitting by minimizing the number of selected features.

---
```
## 📁 Project Structure
```

GA-Feature-Selection/
│
├── src/
│   ├── main.py
│   ├── ga_nsga2.py
│   ├── evaluator.py
│
├── webapp/
│   └── app.py
│
├── results/
│
├── requirements.txt
└── README.md

````

---
````
## ⚙️ How to Run Locally
1. **Install required dependencies**
   ```bash
   pip install -r requirements.txt


2. **Run the Streamlit web app**

   ```bash
   streamlit run webapp/app.py
   ```

---
```
# 🌐 Live Web App
```
You can try the deployed version here:
👉 [**Launch App on Streamlit Cloud**](https://ga-feature-selection-hbirtrno7gf48n5vjevo2z.streamlit.app/)

---
```
## 📊 Project Overview
```
* Algorithm used: **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**
* Evaluation model: **Random Forest Classifier**
* The system automatically detects whether the task is classification or regression.
* Comparison is provided between **NSGA-II**, **SelectKBest**, and **RFE** statistical methods.
* Includes an interactive web interface for uploading datasets, viewing optimal features, and comparing results.

---
```
## 👥 Team Members
```
| Member       | Responsibilities                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| YASIR_149638   | Project supervision, environment setup, GitHub deployment, hosting, and UI enhancements (emojis, user messages, dataset link) |
| aisha_shamoot_159220 | Implementation of `main.py` and `ga_nsga2.py` (core genetic algorithm logic)                                                  |
| Rasha_167204 | Development of the Streamlit web interface (`webapp/app.py`)                                                                  |
| ebtesam_161083 | Preparation and documentation of the final project report                                                                     |
| garam_ | Implementation of the feature evaluation function (`evaluator.py`)                                                            |



---
```
## 🧩 Requirements
```
* Python 3.10 or higher
* Libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `deap` ....

---
```
## 📂 Dataset
```
If you’d like to test using the same dataset we used in development,
you can download it directly from this link:
👉 [Download Original Dataset](https://drive.google.com/file/d/1_6ytYq_tcTXMnYrcCBcpzbXgU7kpSFd7/view)

