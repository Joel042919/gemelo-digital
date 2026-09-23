"""
Módulo de Generación de Población Sintética y Microdatos Calibrados con ENAHO (INEI Perú) y OMS.

Genera microdatos realistas a nivel de hogar e individuo con:
- 24 Departamentos y estratos geográficos de Perú.
- Variables socioeconómicas (Ingreso, Gasto Total, Subsistencia, Pobreza SISFOH, Quintiles Q1-Q5).
- Variables de salud y morbilidad (Enfermedades crónicas, agudas, hospitalizaciones).
- Estado de aseguramiento (Sin Seguro, SIS, EsSalud, Privado).
- Gasto de bolsillo en salud (OOPE) desglosado.
- Indicadores de Gasto Sanitario Catastrófico (CHE 10%, CHE 25%, CHE 40% Capacidad de Pago) y Empobrecimiento.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# Parámetros regionales calibrados con ENAHO / INEI
DEPARTMENTS_DATA = {
    "Lima": {"poverty_rate": 0.21, "rural_rate": 0.02, "chronic_rate": 0.38, "uninsured_base": 0.18, "sis_base": 0.42, "essalud_base": 0.35, "private_base": 0.05, "cost_mult": 1.35},
    "Arequipa": {"poverty_rate": 0.16, "rural_rate": 0.10, "chronic_rate": 0.36, "uninsured_base": 0.15, "sis_base": 0.45, "essalud_base": 0.36, "private_base": 0.04, "cost_mult": 1.15},
    "Cusco": {"poverty_rate": 0.33, "rural_rate": 0.45, "chronic_rate": 0.31, "uninsured_base": 0.12, "sis_base": 0.68, "essalud_base": 0.18, "private_base": 0.02, "cost_mult": 0.95},
    "Cajamarca": {"poverty_rate": 0.44, "rural_rate": 0.65, "chronic_rate": 0.27, "uninsured_base": 0.11, "sis_base": 0.77, "essalud_base": 0.11, "private_base": 0.01, "cost_mult": 0.85},
    "Puno": {"poverty_rate": 0.42, "rural_rate": 0.50, "chronic_rate": 0.29, "uninsured_base": 0.10, "sis_base": 0.76, "essalud_base": 0.13, "private_base": 0.01, "cost_mult": 0.88},
    "Loreto": {"poverty_rate": 0.39, "rural_rate": 0.35, "chronic_rate": 0.26, "uninsured_base": 0.16, "sis_base": 0.71, "essalud_base": 0.12, "private_base": 0.01, "cost_mult": 1.10},
    "Piura": {"poverty_rate": 0.30, "rural_rate": 0.22, "chronic_rate": 0.33, "uninsured_base": 0.17, "sis_base": 0.61, "essalud_base": 0.20, "private_base": 0.02, "cost_mult": 1.00},
    "La Libertad": {"poverty_rate": 0.26, "rural_rate": 0.23, "chronic_rate": 0.34, "uninsured_base": 0.16, "sis_base": 0.58, "essalud_base": 0.23, "private_base": 0.03, "cost_mult": 1.05},
    "Junín": {"poverty_rate": 0.28, "rural_rate": 0.30, "chronic_rate": 0.32, "uninsured_base": 0.14, "sis_base": 0.62, "essalud_base": 0.22, "private_base": 0.02, "cost_mult": 0.95},
    "Ancash": {"poverty_rate": 0.27, "rural_rate": 0.36, "chronic_rate": 0.32, "uninsured_base": 0.15, "sis_base": 0.63, "essalud_base": 0.20, "private_base": 0.02, "cost_mult": 0.98},
    "Lambayeque": {"poverty_rate": 0.22, "rural_rate": 0.18, "chronic_rate": 0.35, "uninsured_base": 0.17, "sis_base": 0.56, "essalud_base": 0.24, "private_base": 0.03, "cost_mult": 1.02},
    "San Martín": {"poverty_rate": 0.29, "rural_rate": 0.33, "chronic_rate": 0.28, "uninsured_base": 0.19, "sis_base": 0.66, "essalud_base": 0.14, "private_base": 0.01, "cost_mult": 0.92},
    "Ica": {"poverty_rate": 0.14, "rural_rate": 0.12, "chronic_rate": 0.37, "uninsured_base": 0.14, "sis_base": 0.48, "essalud_base": 0.34, "private_base": 0.04, "cost_mult": 1.12},
    "Ayacucho": {"poverty_rate": 0.43, "rural_rate": 0.42, "chronic_rate": 0.29, "uninsured_base": 0.09, "sis_base": 0.78, "essalud_base": 0.12, "private_base": 0.01, "cost_mult": 0.88},
    "Huánuco": {"poverty_rate": 0.41, "rural_rate": 0.48, "chronic_rate": 0.28, "uninsured_base": 0.13, "sis_base": 0.74, "essalud_base": 0.12, "private_base": 0.01, "cost_mult": 0.90},
    "Ucayali": {"poverty_rate": 0.27, "rural_rate": 0.25, "chronic_rate": 0.30, "uninsured_base": 0.21, "sis_base": 0.62, "essalud_base": 0.16, "private_base": 0.01, "cost_mult": 1.02},
    "Apurímac": {"poverty_rate": 0.38, "rural_rate": 0.52, "chronic_rate": 0.30, "uninsured_base": 0.08, "sis_base": 0.80, "essalud_base": 0.11, "private_base": 0.01, "cost_mult": 0.86},
    "Amazonas": {"poverty_rate": 0.36, "rural_rate": 0.55, "chronic_rate": 0.27, "uninsured_base": 0.12, "sis_base": 0.75, "essalud_base": 0.12, "private_base": 0.01, "cost_mult": 0.90},
    "Tacna": {"poverty_rate": 0.18, "rural_rate": 0.09, "chronic_rate": 0.36, "uninsured_base": 0.14, "sis_base": 0.46, "essalud_base": 0.36, "private_base": 0.04, "cost_mult": 1.10},
    "Moquegua": {"poverty_rate": 0.15, "rural_rate": 0.13, "chronic_rate": 0.37, "uninsured_base": 0.12, "sis_base": 0.44, "essalud_base": 0.40, "private_base": 0.04, "cost_mult": 1.18},
    "Pasco": {"poverty_rate": 0.37, "rural_rate": 0.38, "chronic_rate": 0.31, "uninsured_base": 0.12, "sis_base": 0.69, "essalud_base": 0.18, "private_base": 0.01, "cost_mult": 0.92},
    "Tumbes": {"poverty_rate": 0.25, "rural_rate": 0.08, "chronic_rate": 0.35, "uninsured_base": 0.16, "sis_base": 0.60, "essalud_base": 0.22, "private_base": 0.02, "cost_mult": 1.02},
    "Huancavelica": {"poverty_rate": 0.47, "rural_rate": 0.68, "chronic_rate": 0.26, "uninsured_base": 0.07, "sis_base": 0.84, "essalud_base": 0.08, "private_base": 0.01, "cost_mult": 0.82},
    "Madre de Dios": {"poverty_rate": 0.17, "rural_rate": 0.21, "chronic_rate": 0.30, "uninsured_base": 0.25, "sis_base": 0.54, "essalud_base": 0.19, "private_base": 0.02, "cost_mult": 1.25}
}

# Línea de pobreza mensual per cápita INEI (promedio nacional aproximado en Soles)
POVERTY_LINE_PER_CAPITA = 415.0  # S/. por persona al mes
EXTREME_POVERTY_LINE_PER_CAPITA = 226.0  # S/. línea canasta alimentaria


def generate_synthetic_enaho_population(n_households: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de microdatos sintéticos representativo de la encuesta ENAHO / SUSALUD
    para simulación y evaluación de políticas de Cobertura Sanitaria Universal.
    """
    np.random.seed(random_state)
    
    # Distribución poblacional por departamentos (ponderada)
    dept_names = list(DEPARTMENTS_DATA.keys())
    dept_weights = [
        0.32, 0.045, 0.042, 0.045, 0.040, 0.032, 0.060, 0.058, 0.042, 0.036,
        0.040, 0.028, 0.027, 0.021, 0.023, 0.016, 0.013, 0.013, 0.011, 0.006,
        0.009, 0.008, 0.011, 0.005
    ]
    dept_weights = np.array(dept_weights) / sum(dept_weights)
    
    chosen_depts = np.random.choice(dept_names, size=n_households, p=dept_weights)
    
    records = []
    
    for i in range(n_households):
        dept = chosen_depts[i]
        d_info = DEPARTMENTS_DATA[dept]
        
        # Área de residencia (Urbana vs Rural)
        is_rural = 1 if np.random.rand() < d_info["rural_rate"] else 0
        area = "Rural" if is_rural else "Urbano"
        
        # Tamaño del hogar y composición
        hh_size = int(np.clip(np.random.negative_binomial(4, 0.5) + 1, 1, 9))
        
        # Niños < 5 años y adultos mayores (>=60 años)
        prob_u5 = 0.40 if is_rural else 0.28
        has_children_u5 = 1 if np.random.rand() < prob_u5 else 0
        num_children_u5 = np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05]) if has_children_u5 else 0
        
        prob_elderly = 0.38 if is_rural else 0.32
        has_elderly = 1 if np.random.rand() < prob_elderly else 0
        num_elderly = np.random.choice([1, 2], p=[0.80, 0.20]) if has_elderly else 0
        
        working_adults = max(1, hh_size - num_children_u5 - num_elderly)
        dependency_ratio = (num_children_u5 + num_elderly) / working_adults
        
        # Jefe de Hogar: Sexo y Educación
        is_female_head = 1 if np.random.rand() < 0.31 else 0
        head_gender = "Mujer" if is_female_head else "Hombre"
        
        if is_rural:
            edu_probs = [0.45, 0.42, 0.10, 0.03]  # Primaria, Secundaria, Sup No Univ, Sup Univ
        else:
            edu_probs = [0.15, 0.45, 0.22, 0.18]
        head_education = np.random.choice(
            ["Primaria o Menos", "Secundaria", "Superior No Universitaria", "Superior Universitaria"],
            p=edu_probs
        )
        
        # Estado socioeconómico y Línea de Pobreza / SISFOH
        poverty_base_prob = d_info["poverty_rate"] * (1.35 if is_rural else 0.85)
        poverty_base_prob = np.clip(poverty_base_prob, 0.05, 0.85)
        
        # Score SISFOH continuo (0 a 100, donde menor valor = más pobre/vulnerable)
        sisfoh_score = np.random.beta(2, 3) * 100 if np.random.rand() < poverty_base_prob else np.random.beta(4, 2) * 100
        sisfoh_score = float(np.clip(sisfoh_score, 1.0, 99.0))
        
        # Gasto e Ingreso del hogar calibrado
        mu_exp = 7.3 if not is_rural else 6.6
        sigma_exp = 0.55
        monthly_total_exp = float(np.exp(np.random.normal(mu_exp, sigma_exp)) * d_info["cost_mult"] * (0.8 + 0.05 * hh_size))
        
        exp_per_capita = monthly_total_exp / hh_size
        
        # Clasificación de pobreza INEI
        if exp_per_capita < EXTREME_POVERTY_LINE_PER_CAPITA:
            poverty_status = "Pobre Extremo"
        elif exp_per_capita < POVERTY_LINE_PER_CAPITA:
            poverty_status = "Pobre No Extremo"
        else:
            poverty_status = "No Pobre"
            
        # Gasto de subsistencia (Alimentos) según Ley de Engel
        food_share = np.clip(0.65 - 0.00012 * monthly_total_exp + (0.08 if is_rural else 0.0), 0.25, 0.75)
        subsistence_exp = float(monthly_total_exp * food_share)
        capacity_to_pay = max(10.0, monthly_total_exp - subsistence_exp)
        
        # Ingreso del hogar
        monthly_income = float(monthly_total_exp * np.random.lognormal(0.05, 0.25))
        
        # Carga de Morbilidad y Salud
        p_chronic = d_info["chronic_rate"] + (0.25 if has_elderly else 0.0) + (0.05 if is_female_head else 0.0)
        p_chronic = np.clip(p_chronic, 0.10, 0.85)
        has_chronic = 1 if np.random.rand() < p_chronic else 0
        chronic_count = int(np.random.choice([1, 2, 3], p=[0.70, 0.22, 0.08])) if has_chronic else 0
        
        # Enfermedad o síntoma agudo en las últimas 4 semanas
        p_acute = 0.35 + (0.15 if has_children_u5 else 0.0) + (0.10 if has_elderly else 0.0)
        has_acute_illness_4w = 1 if np.random.rand() < p_acute else 0
        
        # Hospitalización en el último año
        p_hosp = 0.06 + (0.08 if has_elderly else 0.0) + (0.05 if chronic_count >= 2 else 0.0)
        had_hospitalization = 1 if np.random.rand() < p_hosp else 0
        
        # Estado de Aseguramiento Base
        if poverty_status in ["Pobre Extremo", "Pobre No Extremo"]:
            ins_probs = [0.08, 0.84, 0.07, 0.01]
        elif head_education in ["Superior Universitaria", "Superior No Universitaria"] and not is_rural:
            ins_probs = [0.12, 0.25, 0.52, 0.11]
        else:
            ins_probs = [d_info["uninsured_base"], d_info["sis_base"], d_info["essalud_base"], d_info["private_base"]]
        
        ins_probs = np.array(ins_probs) / sum(ins_probs)
        insurance_type = np.random.choice(["Sin Seguro", "SIS", "EsSalud", "Privado/Otro"], p=ins_probs)
        
        # Indicador de tratamiento: SIS vs Sin Seguro
        is_treated_sis = 1 if insurance_type == "SIS" else 0
        
        # MODELO DE GASTO DE BOLSILLO EN SALUD (OOPE)
        health_need = (
            50.0 +
            chronic_count * 110.0 +
            has_acute_illness_4w * 75.0 +
            had_hospitalization * 380.0 +
            has_elderly * 60.0 +
            has_children_u5 * 45.0 +
            (monthly_total_exp * 0.02)
        ) * d_info["cost_mult"]
        
        # Ruido y shock de salud
        shock_factor = np.random.lognormal(0.0, 0.50)
        gross_health_need = health_need * shock_factor
        
        # Efecto protector del seguro
        if insurance_type == "Sin Seguro":
            oope = gross_health_need * np.random.uniform(0.85, 1.15)
            meds_exp = oope * 0.60
            consult_exp = oope * 0.22
            diag_exp = oope * 0.12
            hosp_exp = oope * 0.06
        elif insurance_type == "SIS":
            copay_ratio = np.random.uniform(0.20, 0.45)
            oope = gross_health_need * copay_ratio
            meds_exp = oope * 0.72
            consult_exp = oope * 0.10
            diag_exp = oope * 0.12
            hosp_exp = oope * 0.06
        elif insurance_type == "EsSalud":
            copay_ratio = np.random.uniform(0.15, 0.35)
            oope = gross_health_need * copay_ratio
            meds_exp = oope * 0.55
            consult_exp = oope * 0.15
            diag_exp = oope * 0.20
            hosp_exp = oope * 0.10
        else:
            copay_ratio = np.random.uniform(0.25, 0.50)
            oope = gross_health_need * copay_ratio
            meds_exp = oope * 0.40
            consult_exp = oope * 0.30
            diag_exp = oope * 0.20
            hosp_exp = oope * 0.10
            
        oope = float(max(5.0, round(oope, 2)))
        meds_exp = float(max(0.0, round(meds_exp, 2)))
        consult_exp = float(max(0.0, round(consult_exp, 2)))
        diag_exp = float(max(0.0, round(diag_exp, 2)))
        hosp_exp = float(max(0.0, round(hosp_exp, 2)))
        
        # INDICADORES OMS DE GASTO CATASTRÓFICO
        che_10 = 1 if (oope / monthly_total_exp) > 0.10 else 0
        che_25 = 1 if (oope / monthly_total_exp) > 0.25 else 0
        che_40_capacity = 1 if (oope / capacity_to_pay) > 0.40 else 0
        
        # Empobrecimiento por Salud
        exp_post_health = monthly_total_exp - oope
        exp_post_health_pc = exp_post_health / hh_size
        impoverished = 1 if (poverty_status == "No Pobre" and exp_post_health_pc < POVERTY_LINE_PER_CAPITA) else 0
        
        records.append({
            "household_id": f"H_{i+1:06d}",
            "department": dept,
            "area": area,
            "is_rural": is_rural,
            "household_size": hh_size,
            "has_children_u5": has_children_u5,
            "num_children_u5": num_children_u5,
            "has_elderly": has_elderly,
            "num_elderly": num_elderly,
            "dependency_ratio": round(dependency_ratio, 2),
            "head_gender": head_gender,
            "is_female_head": is_female_head,
            "head_education": head_education,
            "sisfoh_poverty_score": round(sisfoh_score, 1),
            "poverty_status": poverty_status,
            "monthly_total_exp": round(monthly_total_exp, 2),
            "monthly_income": round(monthly_income, 2),
            "subsistence_food_exp": round(subsistence_exp, 2),
            "capacity_to_pay": round(capacity_to_pay, 2),
            "has_chronic_disease": has_chronic,
            "chronic_disease_count": chronic_count,
            "has_acute_illness_4w": has_acute_illness_4w,
            "had_hospitalization": had_hospitalization,
            "insurance_status": insurance_type,
            "treatment_sis": is_treated_sis,
            "oope_total": oope,
            "oope_medicines": meds_exp,
            "oope_consultations": consult_exp,
            "oope_diagnostics": diag_exp,
            "oope_hospitalization": hosp_exp,
            "oope_share_total_exp": round(oope / monthly_total_exp, 4),
            "oope_share_capacity": round(oope / capacity_to_pay, 4),
            "che_10": che_10,
            "che_25": che_25,
            "che_40_capacity": che_40_capacity,
            "impoverished_by_health": impoverished
        })
        
    df = pd.DataFrame(records)
    
    # Asignar Quintiles de Gasto per cápita (Q1 a Q5)
    df["income_quintile"] = pd.qcut(df["monthly_total_exp"] / df["household_size"], q=5, labels=["Q1 (Más Pobre)", "Q2", "Q3", "Q4", "Q5 (Más Rico)"])
    
    return df


def save_or_load_dataset(filepath: str = "motor-gemelo-digital/data/enaho_synthetic_microdata.csv", n_households: int = 15000) -> pd.DataFrame:
    """Carga el dataset sintético si existe, de lo contrario lo genera y lo guarda."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["income_quintile"] = pd.qcut(df["monthly_total_exp"] / df["household_size"], q=5, labels=["Q1 (Más Pobre)", "Q2", "Q3", "Q4", "Q5 (Más Rico)"])
        return df
    
    df = generate_synthetic_enaho_population(n_households=n_households)
    df.to_csv(filepath, index=False)
    return df


if __name__ == "__main__":
    df = save_or_load_dataset()
    print(f"Dataset generado con {len(df)} registros y {df.shape[1]} columnas.")
    print("Resumen de Gasto Catastrófico Población Base:")
    print(f"- CHE 10%: {df['che_10'].mean()*100:.2f}%")
    print(f"- CHE 25%: {df['che_25'].mean()*100:.2f}%")
    print(f"- CHE 40% Capacidad de Pago: {df['che_40_capacity'].mean()*100:.2f}%")
    print(f"- Empobrecimiento por Salud: {df['impoverished_by_health'].mean()*100:.2f}%")
