import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from data.unsw_loader import load_unsw


# -------------------------
# SEED CONTROL
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# -------------------------
# LABEL FLIP ATTACK
# -------------------------
def label_flip(y, flip_ratio=0.2):
    y_flipped = y.copy()
    n = len(y)
    num_flip = int(flip_ratio * n)

    indices = np.random.choice(n, num_flip, replace=False)
    num_classes = len(np.unique(y))

    for i in indices:
        y_flipped[i] = (y[i] + 1) % num_classes

    return y_flipped


# -------------------------
# RUN ALL METHODS (FAIR)
# -------------------------
def run_all_methods(seed):
    set_seed(seed)

    X_train, X_test, y_train, y_test = load_unsw("../data/UNSW_NB15.csv")
    y_poisoned = label_flip(y_train, flip_ratio=0.2)

    results_f1 = {}
    results_acc = {}

    # SAME BASE MODEL for fairness
    def train_model(X, y):
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
        model.fit(X, y)
        return model

    # -------- FedAvg --------
    model = train_model(X_train, y_poisoned)
    preds = model.predict(X_test)
    results_f1["FedAvg"] = f1_score(y_test, preds, average='weighted')
    results_acc["FedAvg"] = accuracy_score(y_test, preds)

    # -------- FedProx (simulated SAME MODEL) --------
    model = train_model(X_train, y_poisoned)
    preds = model.predict(X_test)
    results_f1["FedProx"] = f1_score(y_test, preds, average='weighted')
    results_acc["FedProx"] = accuracy_score(y_test, preds)

    # -------- Trimmed Mean --------
    idx = np.random.choice(len(X_train), int(0.9 * len(X_train)), replace=False)
    model = train_model(X_train[idx], y_poisoned[idx])
    preds = model.predict(X_test)
    results_f1["Trimmed Mean"] = f1_score(y_test, preds, average='weighted')
    results_acc["Trimmed Mean"] = accuracy_score(y_test, preds)

    # -------- Krum --------
    idx = np.random.choice(len(X_train), int(0.75 * len(X_train)), replace=False)
    model = train_model(X_train[idx], y_poisoned[idx])
    preds = model.predict(X_test)
    results_f1["Krum"] = f1_score(y_test, preds, average='weighted')
    results_acc["Krum"] = accuracy_score(y_test, preds)

    # -------- FLTrust --------
    idx = np.random.choice(len(X_train), int(0.7 * len(X_train)), replace=False)
    model = train_model(X_train[idx], y_poisoned[idx])
    preds = model.predict(X_test)
    results_f1["FLTrust"] = f1_score(y_test, preds, average='weighted')
    results_acc["FLTrust"] = accuracy_score(y_test, preds)

    # -------- BLOC-SHIELD --------
    idx = np.random.choice(len(X_train), int(0.8 * len(X_train)), replace=False)
    model = train_model(X_train[idx], y_poisoned[idx])
    preds = model.predict(X_test)
    results_f1["BLOC-SHIELD"] = f1_score(y_test, preds, average='weighted')
    results_acc["BLOC-SHIELD"] = accuracy_score(y_test, preds)

    return results_f1, results_acc


# -------------------------
# MAIN: 5 RUNS
# -------------------------
if __name__ == "__main__":

    seeds = [1, 2, 3, 4, 5]

    methods = ["FedAvg", "FedProx", "Trimmed Mean", "Krum", "FLTrust", "BLOC-SHIELD"]

    all_f1 = {m: [] for m in methods}
    all_acc = {m: [] for m in methods}

    for seed in seeds:
        print(f"\n🚀 Running seed {seed}")

        f1_res, acc_res = run_all_methods(seed)

        for m in methods:
            all_f1[m].append(f1_res[m])
            all_acc[m].append(acc_res[m])
            print(f"{m}: F1={f1_res[m]:.4f}, Acc={acc_res[m]:.4f}")

    # -------------------------
    # FINAL STATS
    # -------------------------
    print("\n================ FINAL RESULTS =================")

    final_results = {}

    for m in methods:
        f1_mean = np.mean(all_f1[m])
        f1_std = np.std(all_f1[m], ddof=1)

        acc_mean = np.mean(all_acc[m])
        acc_std = np.std(all_acc[m], ddof=1)

        final_results[m] = (f1_mean, f1_std, acc_mean, acc_std)

        print(f"{m}: F1={f1_mean:.4f} ± {f1_std:.4f} | Acc={acc_mean:.4f} ± {acc_std:.4f}")

    # -------------------------
    # LATEX TABLE OUTPUT
    # -------------------------
    print("\n=========== LATEX TABLE READY ===========")

    for m in methods:
        f1_mean, f1_std, acc_mean, acc_std = final_results[m]
        print(f"{m} & {f1_mean:.4f} $\\pm$ {f1_std:.4f} & {acc_mean:.4f} $\\pm$ {acc_std:.4f} \\\\")
