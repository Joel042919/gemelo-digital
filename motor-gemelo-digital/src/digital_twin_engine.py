"""
Motor del Gemelo Digital de Microsimulación Poblacional para Evaluación de Políticas de Salud.

Aplica intervenciones contrafactuales de Cobertura Sanitaria Universal (CSU)
a nivel de microdatos de hogares e individuos utilizando modelos causales entrenados.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Agregar 'src' al path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_generator import (
    save_or_load_dataset,
    POVERTY_LINE_PER_CAPITA,
    EXTREME_POVERTY_LINE_PER_CAPITA
)
from causal_models import (
    FEATURE_COLS,
    DoublyRobustCausalModel,
    DoubleMachineLearningCausalModel,
    XLearnerCausalModel
)


class DigitalTwinEngine:
    """
    Motor de microsimulación del Gemelo Digital de Salud Pública.
    """
    def __init__(
        self,
        data_path: Optional[str] = None,
        models_dir: Optional[str] = None
    ):
        if data_path is None:
            data_path = os.path.join(BASE_DIR, "data", "enaho_synthetic_microdata.csv")
        if models_dir is None:
            models_dir = os.path.join(BASE_DIR, "models")

        self.data_path = data_path
        self.models_dir = models_dir
        self.df_base = save_or_load_dataset(filepath=self.data_path)
        self.models = self._load_available_models()
        self.active_model_name = "Doubly Robust (IPW + Ridge AIPW)" if "Doubly Robust (IPW + Ridge AIPW)" in self.models else list(self.models.keys())[0]

    def _load_available_models(self) -> Dict[str, Any]:
        """Carga los modelos entrenados desde el disco."""
        models = {}
        path_dr = os.path.join(self.models_dir, "model_doubly_robust.pkl")
        path_dml = os.path.join(self.models_dir, "model_dml.pkl")
        path_xl = os.path.join(self.models_dir, "model_xlearner.pkl")
        path_best = os.path.join(self.models_dir, "best_causal_model.pkl")

        if os.path.exists(path_dr):
            m = joblib.load(path_dr)
            models[m.name] = m
        if os.path.exists(path_dml):
            m = joblib.load(path_dml)
            models[m.name] = m
        if os.path.exists(path_xl):
            m = joblib.load(path_xl)
            models[m.name] = m
        if not models and os.path.exists(path_best):
            m = joblib.load(path_best)
            models[m.name] = m
            
        return models

    def set_active_model(self, model_name: str):
        """Permite alternar entre los 3 modelos causales entrenados."""
        if model_name in self.models:
            self.active_model_name = model_name
        else:
            raise ValueError(f"Modelo '{model_name}' no encontrado en los modelos disponibles: {list(self.models.keys())}")

    def simulate_policy(
        self,
        coverage_expansion_rate: float = 0.80,     # Expansión de cobertura sobre no asegurados (0.0 a 1.0)
        targeting_strategy: str = "sisfoh_pobreza", # 'sisfoh_pobreza', 'regional_prioritaria', 'cronicos_vulnerables', 'universal_aleatorio'
        meds_subsidy_depth: float = 0.50,          # Profundidad de subsidio en medicamentos (0.0 a 1.0 adicional)
        catastrophic_cap_enabled: bool = False,    # Techo de gasto de seguridad catastrófica
        catastrophic_cap_ratio: float = 0.30,      # Máximo % de la capacidad de pago antes de subsidio estatal 100%
        model_name: Optional[str] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Ejecuta la microsimulación contrafactual del Gemelo Digital.
        """
        np.random.seed(random_seed)
        
        if model_name and model_name in self.models:
            selected_model = self.models[model_name]
        else:
            selected_model = self.models[self.active_model_name]

        df_sim = self.df_base.copy()
        n_total = len(df_sim)
        
        # Identificar población actualmente No Asegurada
        uninsured_mask = (df_sim["insurance_status"] == "Sin Seguro")
        uninsured_indices = df_sim[uninsured_mask].index.tolist()
        n_uninsured = len(uninsured_indices)
        
        # Determinar número de nuevos afiliados según la meta de política
        n_to_affiliate = int(np.round(n_uninsured * np.clip(coverage_expansion_rate, 0.0, 1.0)))
        
        # Criterio de Priorización / Focalización
        df_uninsured = df_sim.loc[uninsured_indices].copy()
        
        if targeting_strategy == "sisfoh_pobreza":
            poverty_order = {"Pobre Extremo": 0, "Pobre No Extremo": 1, "No Pobre": 2}
            df_uninsured["p_rank"] = df_uninsured["poverty_status"].map(poverty_order)
            df_uninsured = df_uninsured.sort_values(by=["p_rank", "sisfoh_poverty_score"], ascending=[True, True])
            selected_affiliates = df_uninsured.index[:n_to_affiliate].tolist()
            
        elif targeting_strategy == "regional_prioritaria":
            high_priority_depts = ["Huancavelica", "Cajamarca", "Ayacucho", "Puno", "Huánuco", "Loreto", "Apurímac", "Pasco", "Cusco"]
            df_uninsured["is_high_dept"] = df_uninsured["department"].isin(high_priority_depts).astype(int)
            df_uninsured = df_uninsured.sort_values(by=["is_high_dept", "is_rural", "sisfoh_poverty_score"], ascending=[False, False, True])
            selected_affiliates = df_uninsured.index[:n_to_affiliate].tolist()
            
        elif targeting_strategy == "cronicos_vulnerables":
            df_uninsured["vuln_score"] = (
                df_uninsured["chronic_disease_count"] * 3.0 +
                df_uninsured["has_elderly"] * 2.0 +
                df_uninsured["has_children_u5"] * 1.5 +
                df_uninsured["had_hospitalization"] * 4.0
            )
            df_uninsured = df_uninsured.sort_values(by=["vuln_score", "sisfoh_poverty_score"], ascending=[False, True])
            selected_affiliates = df_uninsured.index[:n_to_affiliate].tolist()
            
        else: # universal_aleatorio
            np.random.shuffle(uninsured_indices)
            selected_affiliates = uninsured_indices[:n_to_affiliate]

        # Crear variables contrafactuales en el Gemelo Digital
        df_sim["new_insurance_status"] = df_sim["insurance_status"]
        df_sim.loc[selected_affiliates, "new_insurance_status"] = "SIS (Nuevo Afiliado)"
        df_sim["gained_coverage"] = 0
        df_sim.loc[selected_affiliates, "gained_coverage"] = 1

        # Estimación contrafactual del Gasto de Bolsillo (OOPE) con el Modelo Causal
        X_affiliates = df_sim.loc[selected_affiliates, FEATURE_COLS].values
        current_oope_affiliates = df_sim.loc[selected_affiliates, "oope_total"].values
        
        if len(selected_affiliates) > 0:
            simulated_oope_affiliates = selected_model.predict_counterfactual_oope(
                X_affiliates, current_oope_affiliates, target_treated=1
            )
            if meds_subsidy_depth > 0:
                simulated_oope_affiliates = simulated_oope_affiliates * (1.0 - 0.35 * meds_subsidy_depth)
        else:
            simulated_oope_affiliates = np.array([])

        df_sim["simulated_oope_total"] = df_sim["oope_total"]
        if len(selected_affiliates) > 0:
            df_sim.loc[selected_affiliates, "simulated_oope_total"] = simulated_oope_affiliates

        # Aplicar techo catastrófico si está activo
        if catastrophic_cap_enabled:
            max_allowed_oope = df_sim["capacity_to_pay"] * catastrophic_cap_ratio
            df_sim["simulated_oope_total"] = np.minimum(df_sim["simulated_oope_total"], max_allowed_oope)

        # Recalcular Indicadores OMS Contrafactuales
        df_sim["sim_che_10"] = (df_sim["simulated_oope_total"] / df_sim["monthly_total_exp"] > 0.10).astype(int)
        df_sim["sim_che_25"] = (df_sim["simulated_oope_total"] / df_sim["monthly_total_exp"] > 0.25).astype(int)
        df_sim["sim_che_40_capacity"] = (df_sim["simulated_oope_total"] / df_sim["capacity_to_pay"] > 0.40).astype(int)
        
        sim_exp_post_health = df_sim["monthly_total_exp"] - df_sim["simulated_oope_total"]
        sim_exp_post_health_pc = sim_exp_post_health / df_sim["household_size"]
        df_sim["sim_impoverished_by_health"] = (
            (df_sim["poverty_status"] == "No Pobre") & (sim_exp_post_health_pc < POVERTY_LINE_PER_CAPITA)
        ).astype(int)
        
        df_sim["oope_savings"] = df_sim["oope_total"] - df_sim["simulated_oope_total"]

        # Resumen Nacional
        base_coverage = (df_sim["insurance_status"] != "Sin Seguro").mean() * 100
        sim_coverage = (df_sim["new_insurance_status"] != "Sin Seguro").mean() * 100

        base_che_10 = df_sim["che_10"].mean() * 100
        sim_che_10 = df_sim["sim_che_10"].mean() * 100
        red_che_10 = base_che_10 - sim_che_10

        base_che_25 = df_sim["che_25"].mean() * 100
        sim_che_25 = df_sim["sim_che_25"].mean() * 100
        red_che_25 = base_che_25 - sim_che_25

        base_che_40 = df_sim["che_40_capacity"].mean() * 100
        sim_che_40 = df_sim["sim_che_40_capacity"].mean() * 100
        red_che_40 = base_che_40 - sim_che_40

        base_impov = df_sim["impoverished_by_health"].mean() * 100
        sim_impov = df_sim["sim_impoverished_by_health"].mean() * 100
        red_impov = base_impov - sim_impov

        total_savings_monthly = float(df_sim["oope_savings"].sum())
        mean_savings_per_affiliate = float(df_sim.loc[selected_affiliates, "oope_savings"].mean()) if len(selected_affiliates) > 0 else 0.0

        # Desglose por Quintil
        quintile_summary = []
        for q, grp in df_sim.groupby("income_quintile", observed=False):
            quintile_summary.append({
                "quintile": str(q),
                "n_households": len(grp),
                "base_coverage_pct": round((grp["insurance_status"] != "Sin Seguro").mean() * 100, 2),
                "sim_coverage_pct": round((grp["new_insurance_status"] != "Sin Seguro").mean() * 100, 2),
                "base_che_40_pct": round(grp["che_40_capacity"].mean() * 100, 2),
                "sim_che_40_pct": round(grp["sim_che_40_capacity"].mean() * 100, 2),
                "che_40_reduction_pts": round((grp["che_40_capacity"].mean() - grp["sim_che_40_capacity"].mean()) * 100, 2),
                "avg_oope_before": round(grp["oope_total"].mean(), 2),
                "avg_oope_after": round(grp["simulated_oope_total"].mean(), 2),
                "avg_savings_soles": round(grp["oope_savings"].mean(), 2)
            })

        # Desglose por Departamento
        dept_summary = []
        for dept, grp in df_sim.groupby("department"):
            dept_summary.append({
                "department": dept,
                "n_households": len(grp),
                "new_affiliates_count": int(grp["gained_coverage"].sum()),
                "base_coverage_pct": round((grp["insurance_status"] != "Sin Seguro").mean() * 100, 2),
                "sim_coverage_pct": round((grp["new_insurance_status"] != "Sin Seguro").mean() * 100, 2),
                "base_che_40_pct": round(grp["che_40_capacity"].mean() * 100, 2),
                "sim_che_40_pct": round(grp["sim_che_40_capacity"].mean() * 100, 2),
                "che_40_reduction_pts": round((grp["che_40_capacity"].mean() - grp["sim_che_40_capacity"].mean()) * 100, 2),
                "base_impoverished_pct": round(grp["impoverished_by_health"].mean() * 100, 2),
                "sim_impoverished_pct": round(grp["sim_impoverished_by_health"].mean() * 100, 2),
                "total_savings_soles": round(float(grp["oope_savings"].sum()), 2)
            })

        # Desglose por Vulnerabilidad
        vuln_groups = {
            "Pobre Extremo": df_sim["poverty_status"] == "Pobre Extremo",
            "Pobre No Extremo": df_sim["poverty_status"] == "Pobre No Extremo",
            "No Pobre": df_sim["poverty_status"] == "No Pobre",
            "Con Enfermedades Crónicas": df_sim["has_chronic_disease"] == 1,
            "Sin Enfermedades Crónicas": df_sim["has_chronic_disease"] == 0,
            "Área Rural": df_sim["is_rural"] == 1,
            "Área Urbana": df_sim["is_rural"] == 0,
            "Con Adultos Mayores": df_sim["has_elderly"] == 1,
            "Con Niños < 5 años": df_sim["has_children_u5"] == 1
        }
        
        vuln_summary = []
        for label, mask in vuln_groups.items():
            grp = df_sim[mask]
            vuln_summary.append({
                "subgroup": label,
                "n_households": len(grp),
                "base_che_40_pct": round(grp["che_40_capacity"].mean() * 100, 2),
                "sim_che_40_pct": round(grp["sim_che_40_capacity"].mean() * 100, 2),
                "reduction_pts": round((grp["che_40_capacity"].mean() - grp["sim_che_40_capacity"].mean()) * 100, 2),
                "avg_monthly_savings_soles": round(grp["oope_savings"].mean(), 2)
            })

        return {
            "policy_parameters": {
                "coverage_expansion_rate": coverage_expansion_rate,
                "targeting_strategy": targeting_strategy,
                "meds_subsidy_depth": meds_subsidy_depth,
                "catastrophic_cap_enabled": catastrophic_cap_enabled,
                "catastrophic_cap_ratio": catastrophic_cap_ratio,
                "causal_model_used": selected_model.name
            },
            "kpi_national_summary": {
                "total_population_sample": n_total,
                "uninsured_baseline_count": n_uninsured,
                "newly_affiliated_count": n_to_affiliate,
                "coverage_pct_before": round(base_coverage, 2),
                "coverage_pct_after": round(sim_coverage, 2),
                "coverage_increase_pts": round(sim_coverage - base_coverage, 2),
                
                "che_10_pct_before": round(base_che_10, 2),
                "che_10_pct_after": round(sim_che_10, 2),
                "che_10_reduction_pts": round(red_che_10, 2),
                "che_10_relative_reduction_pct": round((red_che_10 / base_che_10) * 100, 2) if base_che_10 > 0 else 0.0,
                
                "che_25_pct_before": round(base_che_25, 2),
                "che_25_pct_after": round(sim_che_25, 2),
                "che_25_reduction_pts": round(red_che_25, 2),
                
                "che_40_capacity_before": round(base_che_40, 2),
                "che_40_capacity_after": round(sim_che_40, 2),
                "che_40_reduction_pts": round(red_che_40, 2),
                "che_40_relative_reduction_pct": round((red_che_40 / base_che_40) * 100, 2) if base_che_40 > 0 else 0.0,
                
                "impoverished_pct_before": round(base_impov, 2),
                "impoverished_pct_after": round(sim_impov, 2),
                "impoverished_reduction_pts": round(red_impov, 2),
                
                "total_monthly_oope_savings_soles": round(total_savings_monthly, 2),
                "avg_monthly_savings_per_new_affiliate_soles": round(mean_savings_per_affiliate, 2)
            },
            "quintile_equity_summary": pd.DataFrame(quintile_summary),
            "department_summary": pd.DataFrame(dept_summary),
            "vulnerability_summary": pd.DataFrame(vuln_summary),
            "simulated_microdata_sample": df_sim.head(100)
        }


if __name__ == "__main__":
    engine = DigitalTwinEngine()
    print(f"Gemelo Digital inicializado con {len(engine.df_base)} agentes/hogares.")
    print(f"Modelos disponibles: {list(engine.models.keys())}")
    print("\nEjecutando Simulación de Prueba: Expansión de Cobertura al 90% con Focalización SISFOH Pobreza...")
    result = engine.simulate_policy(coverage_expansion_rate=0.90, targeting_strategy="sisfoh_pobreza")
    kpis = result["kpi_national_summary"]
    print(f"- Cobertura: {kpis['coverage_pct_before']}% -> {kpis['coverage_pct_after']}% (+{kpis['coverage_increase_pts']} pts)")
    print(f"- Gasto Catastrófico CHE 40% Capacidad: {kpis['che_40_capacity_before']}% -> {kpis['che_40_capacity_after']}% (Reducción: -{kpis['che_40_reduction_pts']} pts)")
    print(f"- Empobrecimiento por Salud: {kpis['impoverished_pct_before']}% -> {kpis['impoverished_pct_after']}% (-{kpis['impoverished_reduction_pts']} pts)")
    print(f"- Ahorro mensual directo de los hogares: S/. {kpis['total_monthly_oope_savings_soles']:,.2f}")
