import pandas as pd
from ga_nsga2 import run_nsga2  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
from datetime import datetime

if __name__ == "__main__":

    # ===== إعداد البيانات =====
    data_file = "phpB0xrNj.csv"  
    target_column = "class"

    df = pd.read_csv(data_file)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Starting NSGA-II optimization...")
    print(f"Dataset shape: {X.shape}")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Running feature selection with NSGA-II...")
    params = {
        "generations": 5,
        "population_size": 10,
        "data_sampling": 0.7,
        "mutation_prob": 0.2,
        "crossover_prob": 0.8,
        "n_jobs": 4
    }
    for k, v in params.items():
        print(f"- {k}: {v}")
    print("\nThis may take a few minutes...\n")

    # ✅ تنفيذ NSGA-II
    solutions, cache = run_nsga2(
        X.values,
        y.values,
        ngen=params["generations"],
        pop_size=params["population_size"],
        subsample_frac=params["data_sampling"],
        mutpb=params["mutation_prob"],
        cxpb=params["crossover_prob"],
        n_jobs=params["n_jobs"]
    )

    # ===== إعادة تقييم أفضل الحلول =====
    if solutions:
        print("\nRe-evaluating top solutions on full dataset...")
        for sol in solutions:
            if 'features' in sol and len(sol['features']) > 0:
                X_selected = X.iloc[:, sol['features']]
                model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy')
                sol['score'] = scores.mean()

    # ===== حفظ النتائج =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"feature_selection_results_{timestamp}.txt")

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Feature Selection Results\n")
        f.write("=======================\n\n")

        f.write("Parameters:\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\nDataset shape: {}\n\n".format(X.shape))
        f.write(f"Solutions Found: {len(solutions)}\n\n")

        for i, sol in enumerate(solutions, 1):
            try:
                f.write(f"Solution {i}:\n")
                f.write(f"  Accuracy Score: {sol['score']:.4f}\n")
                f.write(f"  Number of Features Selected: {len(sol['features'])}\n")
                f.write("  Selected Features:\n")

                feature_names = [f"f{idx+1}" for idx in sol['features']]
                for j in range(0, len(feature_names), 10):
                    f.write("    " + ", ".join(feature_names[j:j+10]) + "\n")
                f.write("\n")
            except Exception:
                f.write(f"Solution {i}: (unprintable structure)\n")
                f.write(repr(sol) + "\n\n")

    print("\nOptimization completed!")
    print(f"Results saved to: {results_file}")
    print("\nTop 3 solutions:\n")

    for i, sol in enumerate(solutions[:3], 1):
        try:
            print(f"Solution {i}:")
            print(f"  Score: {sol.get('score', 0.0):.4f}")
            print(f"  Features selected: {sol.get('n_features', 'N/A')}")
            print(f"  Feature indices (first 10): {sol['features'][:10]}{'...' if len(sol['features']) > 10 else ''}")
        except Exception:
            print(f"Solution {i}: {repr(sol)}")
