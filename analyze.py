import pickle
import numpy as np

sc = pickle.load(open('d:/Projects/Diabetes/scaler (1).pkl', 'rb'))
lr = pickle.load(open('d:/Projects/Diabetes/logistic_diabetes_model (1).pkl', 'rb'))
rf = pickle.load(open('d:/Projects/Diabetes/Random_forest_diabetes_model (1).pkl', 'rb'))

print("=== SCALER ===")
feats = list(sc.feature_names_in_) if hasattr(sc, 'feature_names_in_') else ['f'+str(i) for i in range(sc.n_features_in_)]
print("Features:", feats)
print("Samples seen:", sc.n_samples_seen_)
print("Mean:", np.round(sc.mean_, 4))
print("Scale:", np.round(sc.scale_, 4))

print()
print("=== LOGISTIC REGRESSION ===")
print("Classes:", lr.classes_)
print("Solver:", lr.solver, "| Penalty:", lr.penalty, "| C:", lr.C)
print("Iterations run:", lr.n_iter_)
print("Intercept:", lr.intercept_)
coefs = lr.coef_[0]
idx = np.argsort(np.abs(coefs))[::-1]
print("Coefficients ranked by abs value:")
for i in idx:
    print("  {:25s}: {:+.4f}".format(feats[i], coefs[i]))

print()
print("=== RANDOM FOREST ===")
print("Classes:", rf.classes_)
print("N estimators:", rf.n_estimators)
print("Max depth:", rf.max_depth)
print("Criterion:", rf.criterion)
print("Class weight:", rf.class_weight)
imp = rf.feature_importances_
idx2 = np.argsort(imp)[::-1]
print("Feature importances (Gini):")
for i in idx2:
    bar = "#" * int(imp[i] * 50)
    print("  {:25s}: {:.4f} {}".format(feats[i], imp[i], bar))
depths = [t.get_depth() for t in rf.estimators_]
print("Tree depths: min={}, max={}, avg={:.1f}".format(min(depths), max(depths), np.mean(depths)))
print("Total trees:", len(rf.estimators_))
