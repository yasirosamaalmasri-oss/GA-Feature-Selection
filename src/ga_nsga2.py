import numpy as np
from deap import base, creator, tools, algorithms
from evaluator import train_and_score
import random

def run_nsga2(X, y, ngen=10, pop_size=20, subsample_frac=1.0,
              mutpb=0.2, cxpb=0.8, n_jobs=1, seed=42):

    # ===== ضبط العشوائية لضمان القابلية للتكرار =====
    random.seed(seed)
    np.random.seed(seed)

    n_features = X.shape[1]

    # ===== إنشاء تراكيب البيانات الخاصة بالخوارزمية الوراثية =====
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: random.randint(0, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ===== دالة التقييم =====
    def evaluate(individual):
        score = train_and_score(X, y, individual, subsample_frac=subsample_frac)
        n_feats = int(np.sum(individual))
        return score, n_feats

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    # ===== إنشاء التعداد الأولي للسكان =====
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5)

    print("\nRunning NSGA-II...")

    # ===== تنفيذ الخوارزمية =====
    pop, logbook = algorithms.eaMuPlusLambda(
        population=pop,
        toolbox=toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        halloffame=hof,
        verbose=True
    )

    # ===== استخراج أفضل الحلول =====
    solutions = []
    for ind in hof:
        score, n_feats = evaluate(ind)
        selected_features = [i for i, bit in enumerate(ind) if bit == 1]
        solutions.append({
            "features": selected_features,
            "score": float(score),
            "n_features": n_feats
        })

    cache = {"logbook": logbook, "hof": hof}

    print("\n✅ NSGA-II completed successfully.")
    print(f"Top {len(solutions)} solutions found.")
    return solutions, cache
