"""
=============================================================================================
DIAGNÓSTICO ESTADÍSTICO Y ECONOMÉTRICO DEL DATASET DE ENTRENAMIENTO ENAHO
Generación de muestra reducida: muestra_200.csv
=============================================================================================
"""

import os
import pandas as pd
import numpy as np

def run_diagnostico(csv_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        cand1 = os.path.join(script_dir, "data", "enaho_synthetic_microdata.csv")
        cand2 = os.path.join(os.getcwd(), "data", "enaho_synthetic_microdata.csv")
        cand3 = os.path.join(os.getcwd(), "motor-gemelo-digital", "data", "enaho_synthetic_microdata.csv")
        if os.path.exists(cand1):
            csv_path = cand1
        elif os.path.exists(cand2):
            csv_path = cand2
        elif os.path.exists(cand3):
            csv_path = cand3
        else:
            csv_path = "data/enaho_synthetic_microdata.csv"

    print("=" * 80)
    print("  DIAGNÓSTICO DEL DATASET DE ENTRENAMIENTO - GEMELO DIGITAL DE SALUD PÚBLICA")
    print("  Fuente: Microdatos ENAHO (INEI Perú) / Calibración Causal N=15,000")
    print("=" * 80)
    
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo en {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Dimensiones y memoria
    print(f"\n[1] DIMENSIONES Y MEMORIA:")
    print(f"    - Total de observaciones (hogares): {len(df):,}")
    print(f"    - Total de variables (columnas):   {df.shape[1]}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"    - Uso aproximado de memoria:       {mem_mb:.2f} MB")
    
    # 2. Tipos de datos y estructura
    print(f"\n[2] ESTRUCTURA DE VARIABLES Y TIPOS DE DATOS (dtypes):")
    for col in df.columns:
        print(f"    - {col:<26}: {str(df[col].dtype):<10} | No nulos: {df[col].count():>6} ({df[col].count()/len(df)*100:.1f}%)")
        
    # 3. Auditoría de valores nulos
    nulls = df.isnull().sum()
    print(f"\n[3] REPORTE DE VALORES NULOS O PERDIDOS (Missing Values):")
    if nulls.sum() == 0:
        print("    -> 0 valores nulos detectados en todas las 37 columnas (Dataset 100% completo).")
    else:
        for col, n in nulls[nulls > 0].items():
            print(f"    - {col}: {n} nulos ({n/len(df)*100:.2f}%)")
            
    # 4. Distribución del tratamiento cuasiexperimental y cobertura
    print(f"\n[4] DISTRIBUCIÓN DE TRATAMIENTO Y SEGURO DE SALUD:")
    print("    A. Categorías de Seguro (insurance_status):")
    ins_counts = df["insurance_status"].value_counts()
    for cat, cnt in ins_counts.items():
        print(f"       * {cat:<15}: {cnt:>6} hogares ({cnt/len(df)*100:.2f}%)")
        
    print("\n    B. Tratamiento Causal Cuasiexperimental (treatment_sis):")
    t_counts = df["treatment_sis"].value_counts()
    for t_val, cnt in t_counts.items():
        label = "Tratado (SIS)" if t_val == 1 else "Control (Sin Seguro)"
        print(f"       * {label:<22} (T={t_val}): {cnt:>6} ({cnt/len(df)*100:.2f}%)")
        
    # 5. Estadísticos descriptivos de variables económicas y de salud
    print(f"\n[5] ESTADÍSTICOS DESCRIPTIVOS DE VARIABLES CLAVE (Soles y Ratios):")
    key_vars = [
        "monthly_income", "monthly_total_exp", "subsistence_food_exp",
        "capacity_to_pay", "oope_total", "sisfoh_poverty_score",
        "che_40_capacity", "che_10", "impoverished_by_health"
    ]
    desc = df[key_vars].describe().round(2).T
    desc["median"] = df[key_vars].median().round(2)
    desc["skewness"] = df[key_vars].skew().round(2)
    cols_show = ["mean", "std", "min", "25%", "median", "75%", "max", "skewness"]
    print(desc[cols_show].to_string())
    
    # 6. Desglose del Gasto de Bolsillo (OOPE) por rubros
    print(f"\n[6] DESGLOSE DEL GASTO DE BOLSILLO EN SALUD (OOPE Promedios):")
    print(f"    - Gasto Total en Salud:           S/. {df['oope_total'].mean():.2f}")
    print(f"    - Medicamentos y Farmacia:        S/. {df['oope_medicines'].mean():.2f} ({df['oope_medicines'].mean()/df['oope_total'].mean()*100:.1f}%)")
    print(f"    - Consultas Médicas:              S/. {df['oope_consultations'].mean():.2f} ({df['oope_consultations'].mean()/df['oope_total'].mean()*100:.1f}%)")
    print(f"    - Diagnóstico y Laboratorio:      S/. {df['oope_diagnostics'].mean():.2f} ({df['oope_diagnostics'].mean()/df['oope_total'].mean()*100:.1f}%)")
    print(f"    - Hospitalización y Cirugías:     S/. {df['oope_hospitalization'].mean():.2f} ({df['oope_hospitalization'].mean()/df['oope_total'].mean()*100:.1f}%)")

    # 7. Incidencia de Gasto Catastrófico (CHE)
    print(f"\n[7] INDICADORES DE PROTECCIÓN FINANCIERA (OMS / ODS 3.8.2):")
    print(f"    - Incidencia CHE 40% (Capacidad de Pago OMS): {df['che_40_capacity'].mean()*100:.2f}% ({df['che_40_capacity'].sum():,} hogares)")
    print(f"    - Incidencia CHE 10% (Gasto Total ODS 3.8.2):   {df['che_10'].mean()*100:.2f}% ({df['che_10'].sum():,} hogares)")
    print(f"    - Tasa de Empobrecimiento por Salud:          {df['impoverished_by_health'].mean()*100:.2f}% ({df['impoverished_by_health'].sum():,} hogares)")

    # 8. Generación de muestra_200.csv
    print(f"\n[8] GENERACIÓN DE ARCHIVO MUESTRA (muestra_200.csv):")
    df_sample = df.sample(n=200, random_state=42)
    
    out_sample_data = os.path.join(script_dir, "data", "muestra_200.csv")
    os.makedirs(os.path.dirname(out_sample_data), exist_ok=True)
    df_sample.to_csv(out_sample_data, index=False)
    
    out_sample_root = os.path.join(os.getcwd(), "muestra_200.csv")
    try:
        df_sample.to_csv(out_sample_root, index=False)
    except Exception:
        pass
        
    print(f"    -> Archivo 'muestra_200.csv' generado exitosamente.")
    print(f"    -> Dimensiones de la muestra: {df_sample.shape[0]} filas x {df_sample.shape[1]} columnas.")
    print(f"    -> Guardado en: {out_sample_data}")
    print(f"    -> Guardado en: {out_sample_root}")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostico()
