"""
Script Principal de Entrenamiento, Tuning de Hiperparámetros y Calibración del Gemelo Digital.
"""

import os
import sys

# Agregar 'src' al path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_generator import save_or_load_dataset
from causal_models import train_and_tune_all_models, DoublyRobustCausalModel, DoubleMachineLearningCausalModel, XLearnerCausalModel


def main():
    print("=================================================================")
    print(" GEMELO DIGITAL DE SALUD PÚBLICA: PIPELINE DE ENTRENAMIENTO & TUNING")
    print("=================================================================")
    
    data_path = os.path.join(BASE_DIR, "data", "enaho_synthetic_microdata.csv")
    models_dir = os.path.join(BASE_DIR, "models")
    
    print("\n1. Cargando / Generando Población Sintética Calibrada ENAHO...")
    df = save_or_load_dataset(filepath=data_path, n_households=15000)
    print(f"   -> Dataset cargado con {len(df)} registros.")
    
    print("\n2. Entrenando y ajustando hiperparámetros de los 3 modelos causales...")
    best_model, benchmark, _ = train_and_tune_all_models(df, save_dir=models_dir)
    
    print("\n3. Verificando persistencia de artefactos:")
    for f in os.listdir(models_dir):
        print(f"   - {f} ({os.path.getsize(os.path.join(models_dir, f))} bytes)")
        
    print("\n¡Pipeline ejecutado exitosamente!")


if __name__ == "__main__":
    main()
