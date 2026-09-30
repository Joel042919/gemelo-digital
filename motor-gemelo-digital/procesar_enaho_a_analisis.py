"""
=============================================================================================
ETL REPRODUCIBLE: CONVERSIÓN DE MICRODATOS REALES ENAHO 2025 (INEI) A TABLA DE ANÁLISIS
PROYECTO: Gemelo Digital de Salud Pública - Cobertura Universal y Gasto Catastrófico
=============================================================================================
Este script procesa e integra los módulos oficiales de la Encuesta Nacional de Hogares 
(ENAHO - Investigación N° 1031 INEI Perú, año 2025):
  - Módulo 01: Características de la Vivienda y Hogar (Enaho01-2025-100.csv)
  - Módulo 02: Características de los Miembros del Hogar (Enaho01-2025-200.csv)
  - Módulo 03: Educación (Enaho01A-2025-300.csv)
  - Módulo 04: Salud individual y gasto de bolsillo (Enaho01A-2025-400.csv)
  - Módulo 34: Sumaria - Gastos, Ingresos y Factores de Expansión (Sumaria-2025.csv)

Salida por defecto: data/enaho_2025_analisis.csv
=============================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

DEPARTAMENTOS = {
    "01": "Amazonas", "02": "Áncash", "03": "Apurímac", "04": "Arequipa", "05": "Ayacucho",
    "06": "Cajamarca", "07": "Callao", "08": "Cusco", "09": "Huancavelica", "10": "Huánuco",
    "11": "Ica", "12": "Junín", "13": "La Libertad", "14": "Lambayeque", "15": "Lima",
    "16": "Loreto", "17": "Madre de Dios", "18": "Moquegua", "19": "Pasco", "20": "Piura",
    "21": "Puno", "22": "San Martín", "23": "Tacna", "24": "Tumbes", "25": "Ucayali"
}

def parse_num(series: pd.Series) -> pd.Series:
    """Convierte cadenas con coma decimal peruana a flotantes."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".").str.strip(),
        errors="coerce"
    ).fillna(0.0)

def procesar_enaho_a_tabla_analisis(
    path_modulo100: str = None,
    path_modulo200: str = None,
    path_modulo300: str = None,
    path_modulo400: str = None,
    path_sumaria: str = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Ejecuta el merge relacional de microdatos ENAHO 2025 y construye
    los indicadores de protección financiera (OMS / ODS 3.8.2).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # Rutas por defecto
    if path_modulo100 is None:
        path_modulo100 = os.path.join(data_dir, "1031-Modulo01", "Enaho01-2025-100.csv")
    if path_modulo200 is None:
        path_modulo200 = os.path.join(data_dir, "1031-Modulo02", "1031-Modulo02", "Enaho01-2025-200.csv")
    if path_modulo300 is None:
        path_modulo300 = os.path.join(data_dir, "1031-Modulo03", "1031-Modulo03", "Enaho01A-2025-300.csv")
    if path_modulo400 is None:
        path_modulo400 = os.path.join(data_dir, "1031-Modulo04", "1031-Modulo04", "Enaho01A-2025-400.csv")
    if path_sumaria is None:
        path_sumaria = os.path.join(data_dir, "1031-Modulo34", "1031-Modulo34", "Sumaria-2025.csv")
    if output_path is None:
        output_path = os.path.join(data_dir, "enaho_2025_analisis.csv")

    print("=" * 80)
    print("  ETL ENAHO 2025 -> TABLA DE ANÁLISIS PARA EL GEMELO DIGITAL")
    print("=" * 80)

    hogar_keys = ["CONGLOME", "VIVIENDA", "HOGAR"]
    persona_keys = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"]

    # -----------------------------------------------------------------------------------------
    # 1. MÓDULO 34: SUMARIA (GASTOS, INGRESOS, POBREZA Y FACTOR07)
    # -----------------------------------------------------------------------------------------
    print("\n[Paso 1/5] Cargando Módulo 34 (Sumaria-2025)...")
    df_sum = pd.read_csv(path_sumaria, sep=";", encoding="latin1", low_memory=False)
    for k in hogar_keys:
        df_sum[k] = df_sum[k].astype(str).str.strip()

    # Identificador de hogar
    df_sum["household_id"] = "H_" + df_sum["CONGLOME"] + "_" + df_sum["VIVIENDA"] + "_" + df_sum["HOGAR"]

    # Departamento y Área
    ubigeo_str = df_sum["UBIGEO"].astype(str).str.strip().str.zfill(6)
    dpto_code = ubigeo_str.str[:2]
    df_sum["department"] = dpto_code.map(DEPARTAMENTOS).fillna("Lima")
    
    estrato_num = pd.to_numeric(df_sum["ESTRATO"], errors="coerce").fillna(1)
    df_sum["is_rural"] = (estrato_num >= 6).astype(int)
    df_sum["area"] = np.where(df_sum["is_rural"] == 1, "Rural", "Urbano")

    # Variables de gasto e ingreso mensualizados (en soles corrientes)
    df_sum["household_size"] = pd.to_numeric(df_sum["MIEPERHO"], errors="coerce").fillna(1).astype(int)
    df_sum["monthly_total_exp"] = parse_num(df_sum["GASHOG2D"]) / 12.0
    df_sum["monthly_income"] = parse_num(df_sum["INGHOG2D"]) / 12.0

    # Pobreza oficial INEI
    pob_val = pd.to_numeric(df_sum["POBREZA"], errors="coerce").fillna(3)
    pob_map = {1: "Pobre Extremo", 2: "Pobre No Extremo", 3: "No Pobre"}
    df_sum["poverty_status"] = pob_val.map(pob_map).fillna("No Pobre")

    # Gasto alimentario normativo de subsistencia (Metodología OMS Ke Xu et al.)
    # LINPE: Línea de pobreza extrema mensual per cápita (canasta básica de alimentos)
    linpe_val = parse_num(df_sum["LINPE"])
    df_sum["subsistence_food_exp"] = linpe_val * df_sum["household_size"]
    
    # Capacidad de Pago (CTP) = Gasto total - Gasto subsistencia (piso de seguridad 10%)
    ctp_raw = df_sum["monthly_total_exp"] - df_sum["subsistence_food_exp"]
    df_sum["capacity_to_pay"] = np.maximum(ctp_raw, df_sum["monthly_total_exp"] * 0.10)

    # Gasto de Bolsillo en Salud (OOPE) según metodología oficial INEI (Sumaria GRU51HD):
    # GRU51HD: Gasto monetario total en salud anual (medicamentos, consultas, exámenes, hospitalización)
    # Se mensualiza dividiendo entre 12.0
    gasto_salud_anual = parse_num(df_sum["GRU51HD"])
    df_sum["oope_total"] = np.round(gasto_salud_anual / 12.0, 2)
    # Respaldo si GRU51HD fuese 0 pero SG42 reporta gasto
    sg42_mensual = np.round(parse_num(df_sum["SG42"]) / 12.0, 2)
    df_sum["oope_total"] = np.where(df_sum["oope_total"] > 0, df_sum["oope_total"], sg42_mensual)

    # Ratios de protección financiera e indicadores CHE
    df_sum["oope_share_total_exp"] = np.clip(
        df_sum["oope_total"] / np.maximum(df_sum["monthly_total_exp"], 1.0), 0.0, 1.0
    )
    df_sum["oope_share_capacity"] = np.clip(
        df_sum["oope_total"] / np.maximum(df_sum["capacity_to_pay"], 1.0), 0.0, 1.0
    )

    # Indicadores OMS / ODS 3.8.2
    df_sum["che_10"] = (df_sum["oope_share_total_exp"] > 0.10).astype(int)
    df_sum["che_25"] = (df_sum["oope_share_total_exp"] > 0.25).astype(int)
    df_sum["che_40_capacity"] = (df_sum["oope_share_capacity"] > 0.40).astype(int)

    # Empobrecimiento por motivos de salud (Impoverishment)
    # Hogar sobre la línea de pobreza pero que cae bajo ella tras descontar el OOPE
    linea_val = parse_num(df_sum["LINEA"]) * df_sum["household_size"]
    sobre_linea = df_sum["monthly_total_exp"] >= linea_val
    bajo_linea_post_oope = (df_sum["monthly_total_exp"] - df_sum["oope_total"]) < linea_val
    df_sum["impoverished_by_health"] = (sobre_linea & bajo_linea_post_oope).astype(int)

    # Factor de expansión oficial
    df_sum["factor07"] = parse_num(df_sum["FACTOR07"])

    # Quintil de ingreso
    inc_pc = df_sum["monthly_income"] / np.maximum(df_sum["household_size"], 1)
    df_sum["income_quintile"] = pd.qcut(
        inc_pc, q=5, 
        labels=["Q1 (Más Pobre)", "Q2", "Q3", "Q4", "Q5 (Más Rico)"]
    )
    
    # Puntaje Proxy SISFOH (calibrado del 1 al 99 según decil de gasto)
    exp_pct = df_sum["monthly_total_exp"].rank(pct=True) * 100.0
    df_sum["sisfoh_poverty_score"] = np.round(100.0 - exp_pct, 1).clip(1.0, 99.0)

    print(f"   -> Hogares en Sumaria: {len(df_sum):,}")

    # -----------------------------------------------------------------------------------------
    # 2. MÓDULO 04: SALUD INDIVIDUAL (SEGURO P419x, MORBILIDAD P401C/P402)
    # -----------------------------------------------------------------------------------------
    print("\n[Paso 2/5] Procesando Módulo 04 (Salud individual y afiliación)...")
    use_cols_m400 = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "P4191", "P4192", "P4193", "P4194", "P4195", "P4198", "P401C"]
    # Columnas opcionales de morbilidad aguda
    df_m400 = pd.read_csv(path_modulo400, sep=";", encoding="latin1", low_memory=False)
    for k in hogar_keys:
        df_m400[k] = df_m400[k].astype(str).str.strip()

    def is_affil(col_name: str) -> pd.Series:
        if col_name in df_m400.columns:
            return (pd.to_numeric(df_m400[col_name].astype(str).str.strip(), errors="coerce").fillna(2) == 1).astype(int)
        return pd.Series(0, index=df_m400.index)

    # CODIFICACIÓN OFICIAL INEI DE SEGUROS (P419x):
    # - P4195: Seguro Integral de Salud (SIS)
    # - P4191: EsSalud
    # - P4192/P4193/P4194: Privado / FFAA / Policiales / EPS
    # - P4198: Ninguno / Sin Seguro
    m_sis = is_affil("P4195")
    m_essalud = is_affil("P4191")
    m_privado = is_affil("P4192") | is_affil("P4193") | is_affil("P4194")
    m_sin_seguro = is_affil("P4198") | ((m_sis == 0) & (m_essalud == 0) & (m_privado == 0))

    df_m400["has_sis"] = m_sis
    df_m400["has_essalud"] = m_essalud
    df_m400["has_privado"] = m_privado
    df_m400["has_sin_seguro"] = m_sin_seguro
    
    # Enfermedad crónica (P401C == 1)
    df_m400["is_chronic"] = (pd.to_numeric(df_m400.get("P401C", 2), errors="coerce").fillna(2) == 1).astype(int)
    
    # Enfermedad aguda últimas 4 semanas (P4021 a P4025)
    acute_cols = [c for c in df_m400.columns if c.startswith("P402")]
    if acute_cols:
        df_m400["is_acute"] = (df_m400[acute_cols].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1).astype(int)
    else:
        df_m400["is_acute"] = 0

    # Hospitalización en el hogar
    df_m400["is_hosp"] = (df_m400.get("P414$05", 2) == 1).astype(int) if "P414$05" in df_m400.columns else 0

    # Desglose de Gastos Directos en Salud reportados en Módulo 400:
    # - P41602: Medicamentos y farmacia
    # - P41601: Consulta médica
    # - P41603 + P41604: Análisis, exámenes y radiografías
    # - P41605: Hospitalizaciones y cirugías
    df_m400["m400_med"] = parse_num(df_m400.get("P41602", pd.Series(0, index=df_m400.index)))
    df_m400["m400_cons"] = parse_num(df_m400.get("P41601", pd.Series(0, index=df_m400.index)))
    df_m400["m400_diag"] = parse_num(df_m400.get("P41603", pd.Series(0, index=df_m400.index))) + parse_num(df_m400.get("P41604", pd.Series(0, index=df_m400.index)))
    df_m400["m400_hosp"] = parse_num(df_m400.get("P41605", pd.Series(0, index=df_m400.index)))

    # Colapso a nivel de hogar
    agg_m400 = df_m400.groupby(hogar_keys).agg(
        sis_members=("has_sis", "sum"),
        essalud_members=("has_essalud", "sum"),
        private_members=("has_privado", "sum"),
        uninsured_members=("has_sin_seguro", "sum"),
        has_chronic_disease=("is_chronic", lambda x: int((x > 0).any())),
        chronic_disease_count=("is_chronic", "sum"),
        has_acute_illness_4w=("is_acute", lambda x: int((x > 0).any())),
        had_hospitalization=("is_hosp", lambda x: int((x > 0).any())),
        m400_med=("m400_med", "sum"),
        m400_cons=("m400_cons", "sum"),
        m400_diag=("m400_diag", "sum"),
        m400_hosp=("m400_hosp", "sum")
    ).reset_index()

    # Determinación categórica del seguro principal del hogar
    def clasificar_seguro_hogar(row):
        if row["sis_members"] > 0:
            return "SIS"
        elif row["essalud_members"] > 0:
            return "EsSalud"
        elif row["private_members"] > 0:
            return "Privado/Otro"
        else:
            return "Sin Seguro"

    agg_m400["insurance_status"] = agg_m400.apply(clasificar_seguro_hogar, axis=1)
    # Tratamiento cuasiexperimental (T=1: SIS, T=0: Sin Seguro)
    agg_m400["treatment_sis"] = np.where(agg_m400["insurance_status"] == "SIS", 1, 0)

    print(f"   -> Módulo 04 agregado a {len(agg_m400):,} hogares.")

    # -----------------------------------------------------------------------------------------
    # 3. MÓDULO 02: DEMOGRAFÍA DE MIEMBROS (EDAD, SEXO DEL JEFE, RATIO DEPENDENCIA)
    # -----------------------------------------------------------------------------------------
    print("\n[Paso 3/5] Procesando Módulo 02 (Demografía y miembros)...")
    df_m200 = pd.read_csv(path_modulo200, sep=";", encoding="latin1", low_memory=False)
    for k in hogar_keys:
        df_m200[k] = df_m200[k].astype(str).str.strip()

    edad = pd.to_numeric(df_m200["P208A"], errors="coerce").fillna(25)
    parentesco = df_m200["P203"].astype(str).str.strip()
    sexo = df_m200["P207"].astype(str).str.strip()

    df_m200["is_u5"] = (edad < 5).astype(int)
    df_m200["is_elderly"] = (edad >= 65).astype(int)
    df_m200["is_working_age"] = ((edad >= 15) & (edad < 65)).astype(int)

    # Identificación del Jefe de Hogar (P203 == '1')
    df_jefes = df_m200[parentesco == "1"].copy()
    df_jefes["is_female_head"] = (df_jefes["P207"].astype(str).str.strip() == "2").astype(int)
    df_jefes["head_gender"] = np.where(df_jefes["is_female_head"] == 1, "Mujer", "Hombre")
    jefes_agg = df_jefes[hogar_keys + ["is_female_head", "head_gender"]].drop_duplicates(subset=hogar_keys)

    # Agregación demográfica
    agg_m200 = df_m200.groupby(hogar_keys).agg(
        num_children_u5=("is_u5", "sum"),
        num_elderly=("is_elderly", "sum"),
        working_age_count=("is_working_age", "sum")
    ).reset_index()

    agg_m200["has_children_u5"] = (agg_m200["num_children_u5"] > 0).astype(int)
    agg_m200["has_elderly"] = (agg_m200["num_elderly"] > 0).astype(int)
    agg_m200["dependency_ratio"] = np.round(
        (agg_m200["num_children_u5"] + agg_m200["num_elderly"]) / np.maximum(agg_m200["working_age_count"], 1), 2
    )

    agg_demografia = pd.merge(agg_m200, jefes_agg, on=hogar_keys, how="left")
    agg_demografia["is_female_head"] = agg_demografia["is_female_head"].fillna(0).astype(int)
    agg_demografia["head_gender"] = agg_demografia["head_gender"].fillna("Hombre")
    print(f"   -> Módulo 02 procesado para {len(agg_demografia):,} hogares.")

    # -----------------------------------------------------------------------------------------
    # 4. MÓDULO 03: EDUCACIÓN DEL JEFE DE HOGAR (P301A)
    # -----------------------------------------------------------------------------------------
    print("\n[Paso 4/5] Procesando Módulo 03 (Educación del jefe)...")
    if os.path.exists(path_modulo300):
        df_m300 = pd.read_csv(path_modulo300, sep=";", encoding="latin1", low_memory=False)
        for k in hogar_keys:
            df_m300[k] = df_m300[k].astype(str).str.strip()
        # Filtrar jefes (CODPERSO == '01' o coincidente con parentesco)
        jefes_m300 = df_m300[df_m300["CODPERSO"].astype(str).str.strip() == "01"].copy()
        p301 = pd.to_numeric(jefes_m300["P301A"], errors="coerce").fillna(4)

        def categorizar_educacion(val):
            if val <= 4:
                return "Primaria o Menos"
            elif val <= 6:
                return "Secundaria"
            elif val <= 8:
                return "Superior No Universitaria"
            else:
                return "Superior Universitaria"

        jefes_m300["head_education"] = p301.map(categorizar_educacion)
        edu_hogar = jefes_m300[hogar_keys + ["head_education"]].drop_duplicates(subset=hogar_keys)
    else:
        edu_hogar = pd.DataFrame(columns=hogar_keys + ["head_education"])

    # -----------------------------------------------------------------------------------------
    # 5. INTEGRACIÓN RELACIONAL FINAL (MERGE 1:1 A NIVEL HOGAR)
    # -----------------------------------------------------------------------------------------
    print("\n[Paso 5/5] Integrando todos los módulos a la tabla final...")
    df_final = pd.merge(df_sum, agg_m400, on=hogar_keys, how="inner")
    df_final = pd.merge(df_final, agg_demografia, on=hogar_keys, how="left")
    
    if not edu_hogar.empty:
        df_final = pd.merge(df_final, edu_hogar, on=hogar_keys, how="left")
        df_final["head_education"] = df_final["head_education"].fillna("Secundaria")
    else:
        df_final["head_education"] = "Secundaria"

    # Desglose empírico del Gasto de Bolsillo (OOPE) por rubros según Módulo 400:
    tot_m400 = df_final["m400_med"] + df_final["m400_cons"] + df_final["m400_diag"] + df_final["m400_hosp"]
    # Si el hogar reportó montos directos en M400, usar su proporción empírica observada
    p_med = np.where(tot_m400 > 0, df_final["m400_med"] / tot_m400, 0.65)
    p_cons = np.where(tot_m400 > 0, df_final["m400_cons"] / tot_m400, 0.18)
    p_diag = np.where(tot_m400 > 0, df_final["m400_diag"] / tot_m400, 0.12)
    p_hosp = np.where(tot_m400 > 0, df_final["m400_hosp"] / tot_m400, 0.05)
    norm = p_med + p_cons + p_diag + p_hosp

    df_final["oope_medicines"] = np.round(df_final["oope_total"] * (p_med / norm), 2)
    df_final["oope_consultations"] = np.round(df_final["oope_total"] * (p_cons / norm), 2)
    df_final["oope_diagnostics"] = np.round(df_final["oope_total"] * (p_diag / norm), 2)
    df_final["oope_hospitalization"] = np.round(df_final["oope_total"] * (p_hosp / norm), 2)

    # Seleccionar y ordenar las 38 columnas analíticas
    columnas_ordenadas = [
        "household_id", "department", "area", "is_rural", "household_size",
        "has_children_u5", "num_children_u5", "has_elderly", "num_elderly", "dependency_ratio",
        "head_gender", "is_female_head", "head_education", "sisfoh_poverty_score", "poverty_status",
        "monthly_total_exp", "monthly_income", "subsistence_food_exp", "capacity_to_pay",
        "has_chronic_disease", "chronic_disease_count", "has_acute_illness_4w", "had_hospitalization",
        "insurance_status", "treatment_sis", "oope_total", "oope_medicines", "oope_consultations",
        "oope_diagnostics", "oope_hospitalization", "oope_share_total_exp", "oope_share_capacity",
        "che_10", "che_25", "che_40_capacity", "impoverished_by_health", "income_quintile",
        "factor07"
    ]
    
    # Mantener solo columnas disponibles
    cols_existentes = [c for c in columnas_ordenadas if c in df_final.columns]
    df_export = df_final[cols_existentes].copy()

    # Guardar archivo analítico
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_export.to_csv(output_path, index=False)
    print(f"\n¡ÉXITO! Tabla analítica generada exitosamente.")
    print(f"  -> Archivo:    {output_path}")
    print(f"  -> Filas:      {len(df_export):,} hogares")
    print(f"  -> Columnas:   {df_export.shape[1]}")
    print(f"  -> Tamaño:     {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print("=" * 80)
    
    return df_export

if __name__ == "__main__":
    procesar_enaho_a_tabla_analisis()
