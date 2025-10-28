import warnings
warnings.filterwarnings("ignore")
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.linear_model import LinearRegression

# ✅ إعداد المسارات
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in [ROOT_DIR, SRC_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from ga_nsga2 import run_nsga2  # ✅ استدعاء الخوارزمية الوراثية فقط

# إعداد الصفحة
st.set_page_config(page_title="Genetic Algorithm Feature Selection", layout="wide")
st.title("🧬 GA Feature Selection Web App")

st.markdown("""
This web app runs a **Genetic Algorithm (NSGA-II)** for feature selection.  
Please upload your dataset to start the process.
""")

# --- تحميل البيانات ---
st.header("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# ✅ إذا تم رفع الملف
data = pd.read_csv(uploaded_file)
st.success("✅ File uploaded successfully!")
st.write("Preview of the dataset:")
st.dataframe(data.head())

# --- اختيار العمود الهدف ---
st.header("🎯 Target Column Selection")
target_column = st.selectbox("Select the target column:", options=list(data.columns))

# --- إعدادات الخوارزمية ---
st.header("⚙️ Algorithm Settings")
pop_size = st.number_input("Population size:", min_value=10, max_value=200, value=10)
n_gen = st.number_input("Number of generations:", min_value=5, max_value=100, value=5)
mutation_rate = st.slider("Mutation rate:", 0.0, 1.0, 0.1)
crossover_rate = st.slider("Crossover rate:", 0.0, 1.0, 0.8)

# --- تشغيل ---
if st.button("🚀 Run Algorithm"):
    with st.spinner(" Please wait... this may take 3–5 minutes depending on your dataset size."):
        try:
            # ✅ تنظيف البيانات
            clean_data = data.copy()
            clean_data = clean_data.apply(pd.to_numeric, errors='coerce')
            clean_data = clean_data.fillna(0)

            X = clean_data.drop(columns=[target_column])
            y = clean_data[target_column]

            # ✅ تحديد نوع المهمة
            task_type = "classification"
            if y.nunique() > 15 or y.dtype.kind in ["f", "M"]:
                task_type = "regression"

            st.info(f"🔍 Task type detected: **{task_type.upper()}**")

            # ✅ تنفيذ الخوارزمية الوراثية
            solutions, cache = run_nsga2(
                X=X.values,
                y=y.values,
                pop_size=pop_size,
                ngen=n_gen,
                mutpb=mutation_rate,
                cxpb=crossover_rate
            )

            # ✅ إعادة تقييم الحلول
            st.subheader("📊 Re-evaluating Solutions using RandomForest")
            for sol in solutions:
                if 'features' in sol and len(sol['features']) > 0:
                    X_selected = X.iloc[:, sol['features']]
                    if task_type == "classification":
                        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                        scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy')
                    else:
                        model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
                        scores = cross_val_score(model, X_selected, y, cv=3, scoring='r2')
                    sol['score'] = scores.mean()
                else:
                    sol['score'] = 0.0

            # ✅ عرض النتائج
            st.success(" Execution completed successfully!")
            st.subheader("Solutions Results:")

            for i, sol in enumerate(solutions):
                st.write(f"**Solution {i + 1}:**")
                st.write(f"- Number of selected features: {len(sol['features'])}")
                st.write(f"- Features: {sol['features']}")
                st.write(f"- Score: {sol['score']:.4f}")

            # ✅ حفظ النتائج
            output_path = os.path.join(ROOT_DIR, "output_solutions.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                for i, sol in enumerate(solutions):
                    f.write(f"Solution {i + 1}:\n")
                    f.write(f"  Features: {sol['features']}\n")
                    f.write(f"  Score: {sol['score']:.4f}\n\n")

            st.download_button(
                label=" Download NSGA-II Results",
                data=open(output_path, "r", encoding="utf-8").read(),
                file_name="output_solutions.txt"
            )

            # ✅ مقارنة مع الطرق الإحصائية التقليدية
            st.markdown("---")
            st.subheader("🔍 Comparison with Traditional Statistical Methods")

            try:
                best_features_count = len(max(solutions, key=lambda s: len(s['features']))['features'])
                st.write(f"Selecting approximately **{best_features_count} features** for comparison.")

                if task_type == "classification":
                    score_func = f_classif
                    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                    scoring = "accuracy"
                else:
                    score_func = f_regression
                    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
                    scoring = "r2"

                # --- SelectKBest ---
                selector = SelectKBest(score_func=score_func, k=min(best_features_count, X.shape[1]))
                X_new = selector.fit_transform(X, y)
                scores = cross_val_score(model, X_new, y, cv=3, scoring=scoring)
                selectkbest_score = scores.mean()

                # --- RFE ---
                rfe_estimator = LinearRegression() if task_type == "regression" else RandomForestClassifier(n_estimators=10, random_state=42)
                rfe = RFE(rfe_estimator, n_features_to_select=min(best_features_count, X.shape[1]), step=0.1)
                X_rfe = rfe.fit_transform(X, y)
                scores = cross_val_score(model, X_rfe, y, cv=3, scoring=scoring)
                rfe_score = scores.mean()

                # --- عرض المقارنة ---
                df_compare = pd.DataFrame({
                    "Method": ["NSGA-II", "SelectKBest", "RFE"],
                    "Number of Features": [best_features_count, X_new.shape[1], X_rfe.shape[1]],
                    "Score": [
                        np.mean([s['score'] for s in solutions if 'score' in s]),
                        selectkbest_score,
                        rfe_score
                    ]
                })

                st.dataframe(df_compare.style.format({"Score": "{:.4f}"}))

            except Exception as e:
                st.warning(f"⚠️ Comparison with traditional methods failed: {e}")

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
