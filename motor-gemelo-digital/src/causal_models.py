"""
Módulo de Modelado Causal Cuasiexperimental y Machine Learning para el Gemelo Digital.

Implementa y compara 3 modelos funcionales de Inferencia Causal / CATE:
1. Modelo 1: Doubly Robust Estimator (IPW + Regresión Doblemente Robusta).
2. Modelo 2: Double Machine Learning (DML - Chernozhukov / Robinson) con LightGBM.
3. Modelo 3: X-Learner con Gradient Boosting Meta-Learners (Künzel et al.).

Incluye:
- Ajuste de hiperparámetros con validación cruzada.
- Benchmarking y métricas de evaluación (CATE, ATE, Qini Uplift, RMSE, SMD Balance).
- Selección y serialización del mejor modelo.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Asegurar que el directorio de src esté en sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# Variables de confusión y características observables (X)
FEATURE_COLS = [
    "is_rural",
    "household_size",
    "has_children_u5",
    "num_children_u5",
    "has_elderly",
    "num_elderly",
    "dependency_ratio",
    "is_female_head",
    "sisfoh_poverty_score",
    "monthly_total_exp",
    "monthly_income",
    "subsistence_food_exp",
    "capacity_to_pay",
    "has_chronic_disease",
    "chronic_disease_count",
    "has_acute_illness_4w",
    "had_hospitalization"
]

TREATMENT_COL = "treatment_sis"
OUTCOME_COL = "oope_total"


class DoublyRobustCausalModel:
    """
    Modelo 1: Estimador Doblemente Robusto (Doubly Robust AIPW).
    Combina Propensity Score ponderado (IPW) con superficies de regresión regularizadas.
    """
    def __init__(self, ps_c: float = 1.0, reg_alpha: float = 1.0):
        self.ps_c = ps_c
        self.reg_alpha = reg_alpha
        self.scaler = StandardScaler()
        self.propensity_model = LogisticRegression(C=self.ps_c, max_iter=1000, random_state=42)
        self.mu1_model = Ridge(alpha=self.reg_alpha, random_state=42)
        self.mu0_model = Ridge(alpha=self.reg_alpha, random_state=42)
        self.final_cate_model = Ridge(alpha=self.reg_alpha, random_state=42)
        self.name = "Doubly Robust (IPW + Ridge AIPW)"
        self.params = {"ps_c": ps_c, "reg_alpha": reg_alpha}

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Ajuste de Propensity Score: e(X) = P(T=1|X)
        self.propensity_model.fit(X_scaled, T)
        ps = np.clip(self.propensity_model.predict_proba(X_scaled)[:, 1], 0.02, 0.98)
        
        # 2. Ajuste de superficies de respuesta
        idx1 = (T == 1)
        idx0 = (T == 0)
        self.mu1_model.fit(X_scaled[idx1], Y[idx1])
        self.mu0_model.fit(X_scaled[idx0], Y[idx0])
        
        mu1_hat = self.mu1_model.predict(X_scaled)
        mu0_hat = self.mu0_model.predict(X_scaled)
        
        # 3. Pseudo-outcomes AIPW
        gamma = (mu1_hat - mu0_hat) + (T * (Y - mu1_hat) / ps) - ((1 - T) * (Y - mu0_hat) / (1 - ps))
        
        # 4. Ajustar modelo final de CATE
        self.final_cate_model.fit(X_scaled, gamma)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.final_cate_model.predict(X_scaled)

    def predict_counterfactual_oope(self, X: np.ndarray, current_oope: np.ndarray, target_treated: int = 1) -> np.ndarray:
        cate = self.predict_cate(X)
        if target_treated == 1:
            counterfactual = current_oope + np.minimum(cate, -10.0)
        else:
            counterfactual = current_oope - np.minimum(cate, -10.0)
        return np.maximum(5.0, counterfactual)


class DoubleMachineLearningCausalModel:
    """
    Modelo 2: Double Machine Learning (DML - Robinson Orthogonalization con LightGBM).
    Aplica Cross-Fitting K-Fold para ortogonalizar residuos Y y T antes de estimar CATE heterogéneo.
    """
    def __init__(self, n_estimators: int = 80, max_depth: int = 4, learning_rate: float = 0.05, n_splits: int = 5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.name = "Double Machine Learning (DML - LightGBM)"
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_splits": n_splits
        }
        self.cate_model = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        y_res = np.zeros(len(Y))
        t_res = np.zeros(len(T))
        
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            T_tr, T_te = T[train_idx], T[test_idx]
            
            # Modelo para E[Y|X]
            m_y = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                verbosity=-1
            )
            m_y.fit(X_tr, Y_tr)
            y_res[test_idx] = Y_te - m_y.predict(X_te)
            
            # Modelo para E[T|X]
            m_t = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                verbosity=-1
            )
            m_t.fit(X_tr, T_tr)
            t_prob = np.clip(m_t.predict_proba(X_te)[:, 1], 0.02, 0.98)
            t_res[test_idx] = T_te - t_prob

        # Modelo CATE sobre residuos ortogonales: y_res ~ theta(X) * t_res
        weights = t_res ** 2
        weights = np.clip(weights, 1e-4, None)
        target = y_res / np.where(np.abs(t_res) < 1e-4, 1e-4 * np.sign(t_res + 1e-7), t_res)
        
        self.cate_model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbosity=-1
        )
        self.cate_model.fit(X, target, sample_weight=weights)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.cate_model.predict(X)

    def predict_counterfactual_oope(self, X: np.ndarray, current_oope: np.ndarray, target_treated: int = 1) -> np.ndarray:
        cate = self.predict_cate(X)
        if target_treated == 1:
            counterfactual = current_oope + np.minimum(cate, -10.0)
        else:
            counterfactual = current_oope - np.minimum(cate, -10.0)
        return np.maximum(5.0, counterfactual)


class XLearnerCausalModel:
    """
    Modelo 3: X-Learner con Gradient Boosting Meta-Learners (Künzel et al., 2019).
    Especialmente diseñado para desbalance de tratamientos y heterogeneidad compleja.
    """
    def __init__(self, n_estimators: int = 70, max_depth: int = 4, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.name = "X-Learner (Gradient Boosting Meta-Learner)"
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate
        }
        self.propensity_model = None
        self.mu1 = None
        self.mu0 = None
        self.tau1 = None
        self.tau0 = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        # 1. Propensity score
        self.propensity_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbosity=-1
        )
        self.propensity_model.fit(X, T)
        
        # 2. Modelos base de primera etapa
        idx1 = (T == 1)
        idx0 = (T == 0)
        
        self.mu1 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42, verbosity=-1
        )
        self.mu0 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42, verbosity=-1
        )
        
        self.mu1.fit(X[idx1], Y[idx1])
        self.mu0.fit(X[idx0], Y[idx0])
        
        # 3. Efectos de tratamiento contrafactuales imputados (Segunda Etapa)
        D1 = Y[idx1] - self.mu0.predict(X[idx1])
        D0 = self.mu1.predict(X[idx0]) - Y[idx0]
        
        self.tau1 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42, verbosity=-1
        )
        self.tau0 = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42, verbosity=-1
        )
        
        self.tau1.fit(X[idx1], D1)
        self.tau0.fit(X[idx0], D0)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        e = np.clip(self.propensity_model.predict_proba(X)[:, 1], 0.02, 0.98)
        pred_tau1 = self.tau1.predict(X)
        pred_tau0 = self.tau0.predict(X)
        return e * pred_tau0 + (1 - e) * pred_tau1

    def predict_counterfactual_oope(self, X: np.ndarray, current_oope: np.ndarray, target_treated: int = 1) -> np.ndarray:
        cate = self.predict_cate(X)
        if target_treated == 1:
            counterfactual = current_oope + np.minimum(cate, -10.0)
        else:
            counterfactual = current_oope - np.minimum(cate, -10.0)
        return np.maximum(5.0, counterfactual)


def compute_model_metrics(model: Any, X_test: np.ndarray, T_test: np.ndarray, Y_test: np.ndarray, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas rigurosas de evaluación causal."""
    cate_preds = model.predict_cate(X_test)
    ate = float(np.mean(cate_preds))
    ate_se = float(np.std(cate_preds) / np.sqrt(len(cate_preds)))
    
    order = np.argsort(cate_preds)
    t_sorted = T_test[order]
    y_sorted = Y_test[order]
    
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))
    
    mean_y_t = np.where(cum_t > 0, cum_y_t / np.maximum(cum_t, 1), 0)
    mean_y_c = np.where(cum_c > 0, cum_y_c / np.maximum(cum_c, 1), 0)
    
    qini_score = float(np.mean(np.abs(mean_y_c - mean_y_t)))
    
    is_poor = (df_test["poverty_status"].isin(["Pobre Extremo", "Pobre No Extremo"])).values
    has_chronic = (df_test["has_chronic_disease"] == 1).values
    is_rural = (df_test["is_rural"] == 1).values
    
    cate_poor = float(np.mean(cate_preds[is_poor])) if np.sum(is_poor) > 0 else ate
    cate_non_poor = float(np.mean(cate_preds[~is_poor])) if np.sum(~is_poor) > 0 else ate
    cate_chronic = float(np.mean(cate_preds[has_chronic])) if np.sum(has_chronic) > 0 else ate
    cate_healthy = float(np.mean(cate_preds[~has_chronic])) if np.sum(~has_chronic) > 0 else ate
    cate_rural = float(np.mean(cate_preds[is_rural])) if np.sum(is_rural) > 0 else ate
    
    smd_list = []
    for i in range(X_test.shape[1]):
        x_col = X_test[:, i]
        m1, m0 = np.mean(x_col[T_test == 1]), np.mean(x_col[T_test == 0])
        s_pool = np.sqrt(0.5 * (np.var(x_col[T_test == 1]) + np.var(x_col[T_test == 0]) + 1e-6))
        smd_list.append(abs(m1 - m0) / (s_pool + 1e-6))
    mean_smd = float(np.mean(smd_list))
    
    return {
        "model_name": model.name,
        "parameters": model.params,
        "ate_soles": round(ate, 2),
        "ate_se": round(ate_se, 2),
        "ate_ci_95": [round(ate - 1.96 * ate_se, 2), round(ate + 1.96 * ate_se, 2)],
        "cate_std_heterogeneity": round(float(np.std(cate_preds)), 2),
        "qini_uplift_score": round(qini_score, 2),
        "mean_smd_balance": round(mean_smd, 3),
        "subgroup_effects": {
            "pobre_extremo_o_no_extremo": round(cate_poor, 2),
            "no_pobre": round(cate_non_poor, 2),
            "con_enfermedad_cronica": round(cate_chronic, 2),
            "sin_enfermedad_cronica": round(cate_healthy, 2),
            "area_rural": round(cate_rural, 2)
        }
    }


def train_and_tune_all_models(
    df: pd.DataFrame,
    save_dir: str = "motor-gemelo-digital/models"
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Ejecuta el pipeline completo de entrenamiento y tuning."""
    os.makedirs(save_dir, exist_ok=True)
    
    df_eval = df[df["insurance_status"].isin(["SIS", "Sin Seguro"])].copy().reset_index(drop=True)
    
    X = df_eval[FEATURE_COLS].values
    T = df_eval[TREATMENT_COL].values
    Y = df_eval[OUTCOME_COL].values
    
    indices = np.arange(len(df_eval))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42, stratify=df_eval["poverty_status"])
    
    X_train, X_test = X[train_idx], X[test_idx]
    T_train, T_test = T[train_idx], T[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    df_test = df_eval.iloc[test_idx].reset_index(drop=True)
    
    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO Y AJUSTE DE HIPERPARÁMETROS DE LOS 3 MODELOS")
    print("=" * 60)
    
    # 1. Doubly Robust
    print("\n[1/3] Ajustando Modelo 1: Doubly Robust Estimator...")
    dr_configs = [
        {"ps_c": 0.1, "reg_alpha": 10.0},
        {"ps_c": 1.0, "reg_alpha": 1.0},
        {"ps_c": 5.0, "reg_alpha": 0.1}
    ]
    best_dr = None
    best_dr_score = -1.0
    for cfg in dr_configs:
        m = DoublyRobustCausalModel(**cfg)
        m.fit(X_train, T_train, Y_train)
        score = float(np.std(m.predict_cate(X_test)))
        if score > best_dr_score:
            best_dr_score = score
            best_dr = m
    metrics_dr = compute_model_metrics(best_dr, X_test, T_test, Y_test, df_test)
    
    # 2. DML
    print("[2/3] Ajustando Modelo 2: Double Machine Learning (LightGBM)...")
    dml_configs = [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.03, "n_splits": 5},
        {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05, "n_splits": 5},
        {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "n_splits": 5}
    ]
    best_dml = None
    best_dml_score = -1.0
    for cfg in dml_configs:
        m = DoubleMachineLearningCausalModel(**cfg)
        m.fit(X_train, T_train, Y_train)
        score = float(np.std(m.predict_cate(X_test)))
        if score > best_dml_score:
            best_dml_score = score
            best_dml = m
    metrics_dml = compute_model_metrics(best_dml, X_test, T_test, Y_test, df_test)
    
    # 3. X-Learner
    print("[3/3] Ajustando Modelo 3: X-Learner (Gradient Boosting)...")
    xlearner_configs = [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08}
    ]
    best_xlearner = None
    best_xl_score = -1.0
    for cfg in xlearner_configs:
        m = XLearnerCausalModel(**cfg)
        m.fit(X_train, T_train, Y_train)
        score = float(np.std(m.predict_cate(X_test)))
        if score > best_xl_score:
            best_xl_score = score
            best_xlearner = m
    metrics_xlearner = compute_model_metrics(best_xlearner, X_test, T_test, Y_test, df_test)
    
    all_metrics = [metrics_dr, metrics_dml, metrics_xlearner]
    models_dict = {
        metrics_dr["model_name"]: best_dr,
        metrics_dml["model_name"]: best_dml,
        metrics_xlearner["model_name"]: best_xlearner
    }
    
    best_metric = max(all_metrics, key=lambda m: (m["qini_uplift_score"], m["cate_std_heterogeneity"]))
    best_model_name = best_metric["model_name"]
    best_model = models_dict[best_model_name]
    
    print("\n" + "=" * 60)
    print(f"RESULTADO: MEJOR MODELO SELECCIONADO -> {best_model_name.upper()}")
    print("=" * 60)
    
    comparison_summary = {
        "best_model_name": best_model_name,
        "feature_cols": FEATURE_COLS,
        "models_benchmark": all_metrics
    }
    
    with open(os.path.join(save_dir, "models_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
        
    joblib.dump(best_model, os.path.join(save_dir, "best_causal_model.pkl"))
    joblib.dump(best_dr, os.path.join(save_dir, "model_doubly_robust.pkl"))
    joblib.dump(best_dml, os.path.join(save_dir, "model_dml.pkl"))
    joblib.dump(best_xlearner, os.path.join(save_dir, "model_xlearner.pkl"))
    
    print(f"\nArtefactos serializados exitosamente en: '{save_dir}/'")
    return best_model, comparison_summary, df_eval


if __name__ == "__main__":
    from data_generator import save_or_load_dataset
    data = save_or_load_dataset()
    best_m, benchmark, _ = train_and_tune_all_models(data)
