# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:15:03 2026
EVO-BOOST CODE
@author: robot
"""
import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ===================== Global random seed and hyperparameters =====================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
LAMBDA = 0.001
N_REPEAT = 10
N_SPLIT = 5

# Fixed weight coefficients for misclassified and hard positive samples
W_WRONG = 2
W_HARD = 2

# Class for constructing single sparse feature chain
class VariableLengthChain(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_indices, depth=2, max_features=4):
        self.feature_indices = tuple(sorted(feature_indices))
        self.depth = depth
        self.max_features = max_features
        self.model = None
        self.f1_val = 0.0
        self.chain_len = len(self.feature_indices)
        self.fitness = 0.0

    def fit(self, X, y, sample_weight=None):
        feats = list(self.feature_indices)
        if len(feats) < 2 or len(feats) > self.max_features:
            self.f1_val = 0.0
            self.fitness = -9999
            return self
        
        X_sub = X[:, feats]
        self.model = DecisionTreeClassifier(
            max_depth=self.depth,
            min_samples_leaf=0.01,
            class_weight="balanced",
            random_state=SEED
        )
        self.model.fit(X_sub, y, sample_weight=sample_weight)
        pred = self.model.predict(X_sub)
        self.f1_val = f1_score(y, pred, zero_division=0)
        self.fitness = (self.f1_val ** 1.5) - LAMBDA * self.chain_len
        return self

    def predict_proba(self, X):
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict_proba(X[:, self.feature_indices])[:, 1]

# EVO-Boost evolutionary cascaded boosting framework
class EVOBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=6, depth=2, max_chain_len=4, 
                 w_wrong=2.0, w_hard=2.0, cand_times=10):
        self.n_estimators = n_estimators
        self.depth = depth
        self.max_chain_len = max_chain_len
        self.w_wrong = w_wrong
        self.w_hard = w_hard
        self.cand_times = cand_times
        self.chains = []
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None

    def set_feature_names(self, names):
        self.feature_names = names

    def set_scaler_stats(self, mean_arr, std_arr):
        self.scaler_mean = mean_arr
        self.scaler_std = std_arr

    # Convert standardized threshold back to original feature scale
    def _std2origin(self, std_val, feat_idx):
        if self.scaler_mean is None or self.scaler_std is None:
            return round(std_val, 2)
        orig = std_val * self.scaler_std[feat_idx] + self.scaler_mean[feat_idx]
        if orig.is_integer():
            return int(orig)
        return round(orig, 2)

    def _get_sample_weight(self, y, prev_prob):
        weight = np.ones(len(y), dtype=np.float32)
        if len(self.chains) == 0:
            return weight
        y_pred = (prev_prob > 0.5).astype(int)
        wrong_idx = y_pred != y
        hard_idx = (y == 1) & (prev_prob < 0.5)
        weight[wrong_idx] *= self.w_wrong
        weight[hard_idx] *= self.w_hard
        return weight / np.mean(weight)

    def _gen_chain(self, X):
        nf = X.shape[1]
        length = random.randint(2, self.max_chain_len)
        feats = sorted(random.sample(range(nf), length))
        return VariableLengthChain(feats, self.depth, self.max_chain_len)

    def fit(self, X, y):
        self.chains.clear()
        cumul_proba = np.zeros(len(y))
        for _ in range(self.n_estimators):
            sw = self._get_sample_weight(y, cumul_proba)
            best_chain = None
            best_fit = -np.inf
            for __ in range(self.cand_times):
                chain = self._gen_chain(X)
                chain.fit(X, y, sw)
                if chain.fitness > best_fit:
                    best_fit = chain.fitness
                    best_chain = chain
            self.chains.append(best_chain)
            cumul_proba += best_chain.predict_proba(X)
        return self

    def predict_proba(self, X):
        preds = np.array([c.predict_proba(X) for c in self.chains])
        return np.mean(preds, axis=0)

    # Output interpretable decision rules for publication
    def explain_rules(self, top_k=6):
   

        if not self.chains:
            print("No chains available.")
            return
        
        # =========================
        # Chain Feature Selection Summary
        # =========================
        print("\nEVO-Boost Chain Feature Selection Summary:")
        for i, ch in enumerate(self.chains, 1):
            feat_names = [self.feature_names[j] for j in ch.feature_indices]
            print(f"  Chain {i}: {feat_names} (f1={ch.f1_val:.4f})")
        
        
        best_chain = max(self.chains, key=lambda c: c.f1_val)
        tree = best_chain.model.tree_
        feat_idx = list(best_chain.feature_indices)
    
        print("\n" + "=" * 70)
        print("         EVO-Boost Rules (Strict Decision Tree Paths)")
        print("=" * 70)
    
        def dfs(node, conditions):
            
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                n_samples = int(tree.n_node_samples[node])
                value = tree.value[node][0]
    
               
                if value.sum() == 0:
                    return
    
                prob = value[1] / value.sum()
                label = "High Risk" if prob >= 0.5 else "Low Risk"
    
                cond_str = " ∧ ".join(conditions) if conditions else "All samples"
                print(f"\nRule:")
                print(f"  IF  {cond_str}")
                print(f"  THEN {label}  (p={prob:.2f}, n={n_samples})")
                return
    
            
            local_fid = tree.feature[node]
            global_fid = feat_idx[local_fid]
    
            fname = self.feature_names[global_fid]
            th_std = tree.threshold[node]
    
            
            th_raw = self._std2origin(th_std, global_fid)
    
           
            dfs(
                tree.children_left[node],
                conditions + [f"{fname} ≤ {th_raw}"]
            )
    
            
            dfs(
                tree.children_right[node],
                conditions + [f"{fname} > {th_raw}"]
            )
    
        dfs(0, [])
        print("\n" + "=" * 70)

# Main experimental workflow
if __name__ == "__main__":
    DATA_FILE = "give me some credit.csv"
    df = pd.read_csv(DATA_FILE)
    X_raw = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feat_names = df.columns[:-1].tolist()

    imp = SimpleImputer(strategy="most_frequent")
    X_imp = imp.fit_transform(X_raw)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)
    mean_all = scaler.mean_
    std_all = scaler.scale_

    evo = EVOBoost(
        n_estimators=6,
        depth=2,
        max_chain_len=4,
        w_wrong=W_WRONG,
        w_hard=W_HARD,
        cand_times=10
    )
    evo.set_feature_names(feat_names)
    evo.set_scaler_stats(mean_all, std_all)

    metric_records = []
    print(f"Start Experiment | Dataset: {DATA_FILE} | W_WRONG={W_WRONG}, W_HARD={W_HARD}")
    print(f"Repeat={N_REPEAT}, K-Fold={N_SPLIT}\n")

    for r in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED + r)
        for tr, te in skf.split(X_std, y):
            evo.fit(X_std[tr], y[tr])
            prob = evo.predict_proba(X_std[te])
            pred = (prob > 0.5).astype(int)

            auc = roc_auc_score(y[te], prob)
            rec = recall_score(y[te], pred, zero_division=0)
            pre = precision_score(y[te], pred, zero_division=0)
            f1 = f1_score(y[te], pred, zero_division=0)
            metric_records.append([auc, rec, pre, f1])

    # Calculate mean and std of all metrics
    metric_arr = np.array(metric_records)
    mean_auc = np.mean(metric_arr[:, 0])
    std_auc = np.std(metric_arr[:, 0])
    mean_rec = np.mean(metric_arr[:, 1])
    std_rec = np.std(metric_arr[:, 1])
    mean_pre = np.mean(metric_arr[:, 2])
    std_pre = np.std(metric_arr[:, 2])
    mean_f1 = np.mean(metric_arr[:, 3])
    std_f1 = np.std(metric_arr[:, 3])

    print("="*60)
    print("               EVO-Boost Quantitative Results")
    print("="*60)
    print(f"AUC      : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Recall   : {mean_rec:.4f} ± {std_rec:.4f}")
    print(f"Precision: {mean_pre:.4f} ± {std_pre:.4f}")
    print(f"F1-Score : {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*60)

    #Output extracted interpretable rules
    evo.explain_rules() 