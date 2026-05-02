# -*- coding: utf-8 -*-
"""

@author: MARCELOFGB
"""
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd 

# --- Configuración Inicial y Semilla para la Reproducibilidad ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Número de características que describen el tráfico de red
N_FEATURES = 3 
FEATURE_NAMES = ["packet_size_bytes", "connection_duration_sec", "failed_login_attempts"]

# Rangos típicos para cada característica para la generación de datos y reglas
FEATURE_RANGES = {
    "packet_size_bytes": (100, 2500),
    "connection_duration_sec": (1, 360),
    "failed_login_attempts": (0, 15)
}

# AUMENTADO: Cada "individuo" (conjunto de anticuerpos/reglas) tendrá más reglas
N_RULES_PER_INDIVIDUAL = 10 # Antes 5, ahora 10

# --- 1. Generación de Datos Sintéticos: Simulación de Tráfico de Red ---
def generate_data(n_samples_normal, n_samples_anomaly_type1, n_samples_anomaly_type2):
    data = []
    labels = []

    # Generar datos "NORMALES" (comportamiento "propio")
    for _ in range(n_samples_normal):
        packet_size = random.randint(100, 600)
        connection_duration = random.randint(10, 120)
        failed_login_attempts = 0 
        data.append([packet_size, connection_duration, failed_login_attempts])
        labels.append(0) 

    # Generar datos de "ANOMALÍA TIPO 1" (ej. "Flood de paquetes grandes")
    for _ in range(n_samples_anomaly_type1):
        packet_size = random.randint(1800, 2500) 
        connection_duration = random.randint(1, 30)
        failed_login_attempts = random.randint(0, 1) 
        data.append([packet_size, connection_duration, failed_login_attempts])
        labels.append(1) 

    # Generar datos de "ANOMALÍA TIPO 2" (ej. "Ataque de Fuerza Bruta en Login")
    for _ in range(n_samples_anomaly_type2):
        packet_size = random.randint(200, 500)
        connection_duration = random.randint(5, 60)
        failed_login_attempts = random.randint(5, 15) 
        data.append([packet_size, connection_duration, failed_login_attempts])
        labels.append(1) 

    return np.array(data), np.array(labels)

# Generamos un dataset de ejemplo. Más anomalías para asegurar que haya muestras que detectar.
X, y = generate_data(n_samples_normal=1000, n_samples_anomaly_type1=50, n_samples_anomaly_type2=17) # Un poco más de anomalías

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

print(f"Dataset de Entrenamiento: {len(X_train)} ejemplos (Normal: {np.sum(y_train == 0)}, Anomalía: {np.sum(y_train == 1)})")
print(f"Dataset de Test: {len(X_test)} ejemplos (Normal: {np.sum(y_test == 0)}, Anomalía: {np.sum(y_test == 1)})")

# --- 2. Configuración de DEAP para Programación Evolutiva ---

# Solución para el RuntimeWarning: Eliminar clases existentes si ya han sido creadas
if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
if "Individual" in creator.__dict__:
    del creator.Individual

# Definimos cómo se evalúa un "individuo" (conjunto de reglas): Buscamos maximizar el F1-Score.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Definimos la estructura de un "individuo": será una lista de reglas.
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Función para crear una sola regla aleatoria (un "anticuerpo" inicial)
def create_rule():
    feature_idx = random.randint(0, N_FEATURES - 1) 
    operator_type = random.randint(0, 1) # 0 para '>', 1 para '<'
    
    min_val, max_val = FEATURE_RANGES[FEATURE_NAMES[feature_idx]]
    threshold_value = random.uniform(min_val, max_val)

    return (feature_idx, operator_type, threshold_value)

toolbox.register("attr_rule", create_rule)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rule, N_RULES_PER_INDIVIDUAL)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 3. Función de Evaluación (Fitness): Mide qué tan bueno es un individuo ---
def evaluate_rules(individual, data, labels):
    predictions = []
    
    for i, row in enumerate(data):
        is_anomaly = False
        for rule_tuple in individual: 
            feature_idx, operator_type, threshold_value = rule_tuple
            feature_value = row[feature_idx]
            
            if operator_type == 0: # '>'
                if feature_value > threshold_value:
                    is_anomaly = True
                    break 
            elif operator_type == 1: # '<'
                if feature_value < threshold_value:
                    is_anomaly = True
                    break
        
        predictions.append(1 if is_anomaly else 0) 

    # Calcular F1-score: Si no hay predicciones positivas, f1_score puede devolver 0.0
    # Asegúrate de que este comportamiento sea el deseado. No hay problema si devuelve 0.
    f1 = f1_score(labels, predictions, average='binary', pos_label=1, zero_division=0) # zero_division=0 para evitar warnings
    
    return f1, 

toolbox.register("evaluate", evaluate_rules, data=X_train, labels=y_train)

# --- 4. Operador Genético Principal: Mutación (Para Programación Evolutiva) ---
def mutate_rule_set(individual, rule_mut_prob, param_mut_prob):
    for i in range(len(individual)): 
        if random.random() < rule_mut_prob: 
            original_rule = list(individual[i]) 

            if random.random() < param_mut_prob:
                original_rule[0] = random.randint(0, N_FEATURES - 1)
            
            if random.random() < param_mut_prob:
                original_rule[1] = random.randint(0, 1)
            
            if random.random() < param_mut_prob:
                feature_idx_for_threshold = original_rule[0] 
                min_val, max_val = FEATURE_RANGES[FEATURE_NAMES[feature_idx_for_threshold]]
                original_rule[2] = random.uniform(min_val, max_val)
            
            individual[i] = tuple(original_rule) 
    return individual,

# REDUCIDO: Ajustamos la probabilidad de mutar los parámetros de una regla.
# Esto hace que las mutaciones sean menos drásticas, permitiendo un "refinamiento".
toolbox.register("mutate", mutate_rule_set, rule_mut_prob=0.3, param_mut_prob=0.2) # Antes 0.4, 0.4

# --- 5. Operador de Selección: Selección por Torneo ---
toolbox.register("select", tools.selTournament, tournsize=3)

# --- 6. Parámetros del Algoritmo de Programación Evolutiva ---
# AUMENTADO: Un mayor tamaño de población y más generaciones
POP_SIZE = 100 # Antes 100, ahora 200
NGEN = 40 # Antes 70, ahora 100

MU_PARENTS = POP_SIZE
LAMBDA_OFFSPRING = POP_SIZE * 2 

MUTPB_OFFSPRING = 1.0 

# --- 7. Ejecución del Algoritmo de Programación Evolutiva ---
def run_ep():
    pop = toolbox.population(n=POP_SIZE) 
    hof = tools.HallOfFame(1) 
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU_PARENTS, lambda_=LAMBDA_OFFSPRING, 
                                        cxpb=0.0, mutpb=MUTPB_OFFSPRING, ngen=NGEN, 
                                        stats=stats, halloffame=hof, verbose=False) 

    return pop, logbook, hof

if __name__ == "__main__":
    pop, log, hof = run_ep()

    # --- 8. Visualización de la Evolución del Fitness ---
    gen = log.select("gen")
    avg_fitness = log.select("avg")
    max_fitness = log.select("max")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, avg_fitness, label="Fitness Promedio (F1-Score)")
    plt.plot(gen, max_fitness, label="Mejor Fitness (F1-Score)")
    plt.xlabel("Generación")
    plt.ylabel("F1-Score (Capacidad de Detección)")
    plt.title("Evolución del F1-Score de los 'Anticuerpos' (Reglas)")
    plt.grid(True)
    plt.legend()
    plt.ylim([-0.05, 1.05]) # Forzar el rango Y para que sea de 0 a 1 (o un poco más ancho)
    plt.show()

    # --- 9. Mostrar el Mejor Individuo (Conjunto de Reglas/Anticuerpos) Encontrado ---
    best_individual = hof[0]
    print("\n--- Mejor Conjunto de Reglas ('Anticuerpos') Evolucionado ---")
    operator_map = {0: ">", 1: "<"}
    for i, rule_tuple in enumerate(best_individual):
        feature_idx, operator_type, threshold_value = rule_tuple
        print(f"  Regla {i+1}: SI ({FEATURE_NAMES[feature_idx]} {operator_map[operator_type]} {threshold_value:.2f}) ENTONCES ANOMALÍA")
    print(f"\nFitness del mejor individuo (F1-Score en entrenamiento): {best_individual.fitness.values[0]:.4f}")

    # --- 10. Simulación de Detección de Intrusión con el Mejor Individuo ---
    print("\n--- Simulación de Detección de Intrusión en un Nuevo Escenario (Datos de Test) ---")

    test_predictions = []
    for i, row in enumerate(X_test):
        is_anomaly = False
        for rule_tuple in best_individual:
            feature_idx, operator_type, threshold_value = rule_tuple
            feature_value = row[feature_idx]
            
            if operator_type == 0: # '>'
                if feature_value > threshold_value:
                    is_anomaly = True
                    break
            elif operator_type == 1: # '<'
                if feature_value < threshold_value:
                    is_anomaly = True
                    break
        
        test_predictions.append(1 if is_anomaly else 0)

    report = classification_report(y_test, test_predictions, target_names=["Tráfico Normal", "Tráfico Anómalo"])
    print("\nReporte de Clasificación en Datos de Test:")
    print(report)

    # --- 11. Ejemplos Concretos de Predicciones ---
    print("\n--- Ejemplos de Clasificación de Tráfico de Red ---")
    df_test = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    df_test['Etiqueta Real'] = ['Normal' if l == 0 else 'Anómala' for l in y_test]
    df_test['Predicción'] = ['Normal' if p == 0 else 'Anómala' for p in test_predictions]
    df_test['Resultado'] = np.where(df_test['Etiqueta Real'] == df_test['Predicción'], 'Correcto', 'Incorrecto')

    print("\nPrimeros 10 eventos de tráfico en el conjunto de prueba:")
    print(df_test.head(10).to_string())

    print("\nEventos Anómalos Reales Detectados Correctamente (Verdaderos Positivos):")
    print(df_test[(df_test['Etiqueta Real'] == 'Anómala') & (df_test['Predicción'] == 'Anómala')].head().to_string())

    print("\nEventos Normales Clasificados Erróneamente como Anómalos (Falsos Positivos):")
    print(df_test[(df_test['Etiqueta Real'] == 'Normal') & (df_test['Predicción'] == 'Anómala')].head().to_string())

    print("\nEventos Anómalos Reales NO Detectados (Falsos Negativos):")
    print(df_test[(df_test['Etiqueta Real'] == 'Anómala') & (df_test['Predicción'] == 'Normal')].head().to_string())