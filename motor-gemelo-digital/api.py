"""
API REST con FastAPI para el Gemelo Digital de Salud Pública.

Expone endpoints para consultar métricas de modelos, ejecutar simulaciones de políticas
y realizar limpieza y auditoría de microdatos con LangChain.
"""

import os
import sys
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from digital_twin_engine import DigitalTwinEngine
from data_cleaning_langchain import LangChainDataCleaner


app = FastAPI(
    title="Gemelo Digital de Salud Pública & LangChain Cleaner API",
    description="Motor de Microsimulación Cuasiexperimental de Cobertura Universal y Gasto Catastrófico con Asistente LangChain",
    version="1.1.0"
)

# Instancias globales
engine = DigitalTwinEngine()
cleaner = LangChainDataCleaner()


class PolicySimulationRequest(BaseModel):
    coverage_expansion_rate: float = Field(0.85, ge=0.0, le=1.0, description="Tasa de cobertura para no asegurados (0.0 a 1.0)")
    targeting_strategy: str = Field("sisfoh_pobreza", description="Estrategia: sisfoh_pobreza, regional_prioritaria, cronicos_vulnerables, universal_aleatorio")
    meds_subsidy_depth: float = Field(0.50, ge=0.0, le=1.0, description="Profundidad de subsidio a medicamentos (0.0 a 1.0)")
    catastrophic_cap_enabled: bool = Field(True, description="Habilitar techo de gasto de seguridad catastrófica")
    catastrophic_cap_ratio: float = Field(0.30, ge=0.05, le=0.80, description="Tope de % de capacidad de pago")
    model_name: Optional[str] = Field(None, description="Modelo causal a emplear")


class LangChainQueryRequest(BaseModel):
    user_query: str = Field(..., description="Consulta metodológica o de limpieza para LangChain")


@app.get("/")
def root():
    return {
        "title": "Gemelo Digital de Salud Pública API",
        "version": "1.1.0",
        "langchain_cleaning_module": "active",
        "status": "online",
        "population_sample_size": len(engine.df_base),
        "available_causal_models": list(engine.models.keys()),
        "active_model": engine.active_model_name,
        "docs_url": "/docs"
    }


@app.get("/models")
def get_models_benchmark():
    path = os.path.join(BASE_DIR, "models", "models_benchmark.json")
    if os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Benchmark data not found"}


@app.get("/models/statistical-tests")
def get_statistical_tests():
    """Retorna la suite completa de 5 dimensiones de pruebas estadísticas y refutación causal."""
    path = os.path.join(BASE_DIR, "models", "comprehensive_statistical_tests.json")
    if os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Statistical tests data not found"}


@app.post("/simulate")
def simulate_policy(req: PolicySimulationRequest):
    res = engine.simulate_policy(
        coverage_expansion_rate=req.coverage_expansion_rate,
        targeting_strategy=req.targeting_strategy,
        meds_subsidy_depth=req.meds_subsidy_depth,
        catastrophic_cap_enabled=req.catastrophic_cap_enabled,
        catastrophic_cap_ratio=req.catastrophic_cap_ratio,
        model_name=req.model_name
    )
    
    return {
        "policy_parameters": res["policy_parameters"],
        "kpi_national_summary": res["kpi_national_summary"],
        "quintile_equity_summary": res["quintile_equity_summary"].to_dict(orient="records"),
        "department_summary": res["department_summary"].to_dict(orient="records"),
        "vulnerability_summary": res["vulnerability_summary"].to_dict(orient="records"),
        "sample_microdata": res["simulated_microdata_sample"].head(10).to_dict(orient="records")
    }


@app.get("/data/summary")
def get_baseline_summary():
    df = engine.df_base
    return {
        "total_households": len(df),
        "departments_count": df["department"].nunique(),
        "baseline_coverage_pct": round((df["insurance_status"] != "Sin Seguro").mean() * 100, 2),
        "baseline_che_10_pct": round(df["che_10"].mean() * 100, 2),
        "baseline_che_25_pct": round(df["che_25"].mean() * 100, 2),
        "baseline_che_40_capacity_pct": round(df["che_40_capacity"].mean() * 100, 2),
        "baseline_impoverished_pct": round(df["impoverished_by_health"].mean() * 100, 2),
        "average_monthly_oope_soles": round(df["oope_total"].mean(), 2)
    }


@app.post("/cleaning/audit")
def audit_and_clean_with_langchain(sample_size: int = Query(50, ge=10, le=500)):
    raw_df = cleaner.generate_dirty_mock_sample(sample_size)
    clean_df, report = cleaner.clean_dataset_with_langchain(raw_df)
    return {
        "audit_report": report,
        "sample_raw_data": raw_df.head(5).to_dict(orient="records"),
        "sample_cleaned_data": clean_df.head(5).to_dict(orient="records")
    }


@app.post("/cleaning/query")
def ask_langchain_cleaning_assistant(req: LangChainQueryRequest):
    response = cleaner.assistant_chain.invoke({
        "user_query": req.user_query,
        "context_info": "Gemelo Digital Perú ENAHO - Gasto Catastrófico y Cobertura Sanitaria Universal"
    })
    return {
        "query": req.user_query,
        "langchain_response": response
    }


@app.get("/crisp-dm/factors")
def get_crisp_dm_factors():
    """Retorna la tabla formal de factores (variables dependientes, independientes y tratamiento)."""
    from crisp_dm_engine import generate_factor_table
    df_factors = generate_factor_table()
    return df_factors.to_dict(orient="records")


@app.get("/crisp-dm/eda")
def get_crisp_dm_eda():
    """Retorna el análisis exploratorio de datos (EDA), estadísticos descriptivos y correlaciones."""
    path = os.path.join(BASE_DIR, "models", "crisp_dm_eda_summary.json")
    if os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    from crisp_dm_engine import run_full_eda
    return run_full_eda(engine.df_base)


@app.get("/crisp-dm/tests-matrix")
def get_crisp_dm_tests_matrix():
    """Retorna la matriz formal de pruebas estadísticas dividida en Paramétricas vs No Paramétricas."""
    path = os.path.join(BASE_DIR, "models", "crisp_dm_tests_matrix.json")
    if os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    from crisp_dm_engine import get_crisp_dm_statistical_tests_matrix
    return get_crisp_dm_statistical_tests_matrix()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
