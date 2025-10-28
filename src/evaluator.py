# ===== evaluator.py =====
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_and_score(X, y, individual, task='classification', n_splits=3, subsample_frac=1.0, seed=42):
    """
    دالة لتقييم أداء مجموعة من الميزات (features) المحددة بواسطة الخوارزمية الوراثية.
    تُرجع النتيجة (score) على شكل عدد عشري بين 0 و 1.
    """

    # ===== تحويل المدخلات إلى مصفوفات NumPy =====
    X = np.asarray(X)
    y = np.asarray(y)
    mask = np.array(individual, dtype=bool)
    nsel = int(mask.sum())

    # إذا لم يتم اختيار أي ميزة نُرجع نتيجة 0
    if nsel == 0:
        return 0.0

    X_sel = X[:, mask]

    # ===== تطبيق أخذ عينات فرعية (اختياري لتسريع التجارب) =====
    if 0 < subsample_frac < 1.0:
        rng = np.random.RandomState(seed)
        n = X_sel.shape[0]
        k = max(10, int(n * subsample_frac))  # حجم العينة الدنيا للتسريع
        k = min(k, n)
        idx = rng.choice(n, size=k, replace=False)
        X_sel = X_sel[idx]
        y_sel = y[idx]
    else:
        y_sel = y

    # ===== استخدام نموذج RandomForest بسيط للتقييم =====
    model = RandomForestClassifier(
        n_estimators=10, 
        max_depth=3, 
        random_state=seed, 
        n_jobs=1
    )

    try:
        # تقسيم البيانات إلى تدريب واختبار
        Xtr, Xte, ytr, yte = train_test_split(X_sel, y_sel, test_size=0.3, random_state=seed)
        model.fit(Xtr, ytr)
        score = float(model.score(Xte, yte))

        # في حال ظهور NaN نعيد 0
        if np.isnan(score):
            score = 0.0
        return score

    # ===== معالجة استثناءات محتملة =====
    except Exception:
        try:
            Xtr, Xte, ytr, yte = train_test_split(X_sel, y_sel, test_size=0.3, random_state=seed)
            val = float(model.score(Xte, yte))
            if np.isnan(val):
                val = 0.0
            return val
        except Exception:
            return 0.0
