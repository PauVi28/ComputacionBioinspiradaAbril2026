# -*- coding: utf-8 -*-
"""


@author: MARCELOFGB
"""
import random
import numpy as np
from deap import base, creator, tools #, algorithms
import matplotlib.pyplot as plt

# --- 1. Definición del Problema: Datos del Diagnóstico Médico ---

# Síntomas (features) - usaremos valores binarios (0: Ausente, 1: Presente) para simplificar
SYMPTOMS = {
    0: 'Fiebre',
    1: 'Tos',
    2: 'Fatiga',
    3: 'Dolor_Garganta',
    4: 'Congestion_Nasal',
    5: 'Dolor_Cabeza',
    6: 'Erupcion_Cutanea'
}
NUM_SYMPTOMS = len(SYMPTOMS)

# Enfermedades (clases a diagnosticar)
DISEASES = {
    0: 'Gripe',
    1: 'Resfriado',
    2: 'Sarampion',
    3: 'Desconocido' # Para casos no diagnosticados
}
NUM_DISEASES = len(DISEASES)

# --- Dataset de Pacientes (Simulado) ---
# Cada paciente es un diccionario de síntomas y el ID de la enfermedad real
# (symptoms_dict, actual_disease_id)
# {symptom_id: value}

TRAINING_DATA = [
    ({'Fiebre':1, 'Tos':1, 'Fatiga':1, 'Dolor_Garganta':0, 'Congestion_Nasal':0, 'Dolor_Cabeza':1, 'Erupcion_Cutanea':0}, 0), # Gripe
    ({'Fiebre':0, 'Tos':1, 'Fatiga':0, 'Dolor_Garganta':1, 'Congestion_Nasal':1, 'Dolor_Cabeza':0, 'Erupcion_Cutanea':0}, 1), # Resfriado
    ({'Fiebre':1, 'Tos':0, 'Fatiga':0, 'Dolor_Garganta':0, 'Congestion_Nasal':0, 'Dolor_Cabeza':0, 'Erupcion_Cutanea':1}, 2), # Sarampion
    ({'Fiebre':1, 'Tos':1, 'Fatiga':0, 'Dolor_Garganta':1, 'Congestion_Nasal':1, 'Dolor_Cabeza':1, 'Erupcion_Cutanea':0}, 0), # Gripe (mezcla)
    ({'Fiebre':0, 'Tos':1, 'Fatiga':0, 'Dolor_Garganta':1, 'Congestion_Nasal':0, 'Dolor_Cabeza':0, 'Erupcion_Cutanea':0}, 1), # Resfriado
    ({'Fiebre':0, 'Tos':0, 'Fatiga':1, 'Dolor_Garganta':0, 'Congestion_Nasal':0, 'Dolor_Cabeza':1, 'Erupcion_Cutanea':0}, 0), # Gripe (atípica)
    ({'Fiebre':1, 'Tos':0, 'Fatiga':0, 'Dolor_Garganta':0, 'Congestion_Nasal':0, 'Dolor_Cabeza':0, 'Erupcion_Cutanea':1}, 2), # Sarampion
    ({'Fiebre':0, 'Tos':0, 'Fatiga':0, 'Dolor_Garganta':0, 'Congestion_Nasal':0, 'Dolor_Cabeza':0, 'Erupcion_Cutanea':0}, 3), # Desconocido (paciente sano)
    ({'Fiebre':1, 'Tos':1, 'Fatiga':1, 'Dolor_Garganta':1, 'Congestion_Nasal':1, 'Dolor_Cabeza':1, 'Erupcion_Cutanea':0}, 0), # Gripe (heavy)
]

# Convertir datos a formato numérico (ID de síntoma -> valor) para el procesamiento de reglas
def prepare_patient_data(symptoms_dict):
    patient_vector = np.zeros(NUM_SYMPTOMS)
    for s_id, s_name in SYMPTOMS.items():
        if s_name in symptoms_dict:
            patient_vector[s_id] = symptoms_dict[s_name]
    return patient_vector

PREPARED_TRAINING_DATA = [(prepare_patient_data(p[0]), p[1]) for p in TRAINING_DATA]

# --- 2. Representación del Individuo: Una Regla ---

# CAMBIO 1: Rule ya no hereda de list, solo gestiona antecedent como un atributo list.
class Rule:
    """
    Representa una única regla "SI... ENTONCES...".
    El antecedente es una lista de tuplas (symptom_id, required_value).
    """
    def __init__(self, antecedent, consequent):
        self.antecedent = list(antecedent) # Asegurarse de que el antecedente se guarde como una lista mutable
        self.consequent = consequent # ID de la enfermedad

    def applies(self, patient_symptoms_vector):
        """
        Verifica si la regla se aplica a un paciente dado sus síntomas.
        patient_symptoms_vector: np.array de valores de síntomas.
        """
        for sym_id, req_val in self.antecedent:
            if patient_symptoms_vector[sym_id] != req_val:
                return False
        return True

    def __str__(self):
        antecedent_str = " AND ".join([f"{SYMPTOMS[s_id]}={val}" for s_id, val in self.antecedent])
        return f"SI ({antecedent_str}) ENTONCES {DISEASES[self.consequent]}"

# --- 3. Configuración de DEAP (se hace antes de inicializar reglas para asegurar el tipo) ---

# 1. Definir los tipos de fitness y individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximizar la precisión
# CAMBIO 2: creator.Individual es ahora nuestra clase 'Rule' personalizada con el atributo fitness.
creator.create("Individual", Rule, fitness=creator.FitnessMax) # Nuestro individuo es una Rule

# --- 3.1. Inicialización del Individuo (Regla) ---

def init_rule():
    """
    Crea una regla aleatoria para la población inicial.
    CAMBIO 3: Ahora devuelve una instancia de creator.Individual.
    """
    # Antecedente: número aleatorio de condiciones (1 a 3)
    num_cond = random.randint(1, 3)
    antecedent = []
    available_symptoms = list(range(NUM_SYMPTOMS))
    random.shuffle(available_symptoms) # Para elegir síntomas únicos
    
    for _ in range(num_cond):
        if not available_symptoms:
            break
        sym_id = available_symptoms.pop()
        req_val = random.randint(0, 1) # Valor binario para el síntoma
        antecedent.append((sym_id, req_val))
    
    # Consecuente: enfermedad aleatoria
    consequent = random.randint(0, NUM_DISEASES - 1)
    
    return creator.Individual(antecedent, consequent) # Instancia de creator.Individual

# --- 4. Operador de Mutación (Exclusivo en PE) ---

def mutRule(individual):
    """
    Operador de mutación para una regla (fenotípica).
    Modifica el antecedente o el consecuente de la regla.
    CAMBIO 4: Asegura que la regla mutada también sea un creator.Individual.
    """
    # Crear una nueva instancia de creator.Individual para la regla mutada
    # Esto es crucial para que el nuevo individuo tenga el atributo 'fitness'
    mutated_rule = creator.Individual(list(individual.antecedent), individual.consequent) 
    
    action = random.choice(['change_consequent', 'add_condition', 'remove_condition', 'change_condition_value', 'change_condition_symptom'])

    if action == 'change_consequent':
        # Mutar el consecuente (enfermedad de diagnóstico)
        new_consequent = random.randint(0, NUM_DISEASES - 1)
        # Bucle para asegurar que mute, solo si hay más de 1 enfermedad posible
        while new_consequent == mutated_rule.consequent and NUM_DISEASES > 1: 
            new_consequent = random.randint(0, NUM_DISEASES - 1)
        mutated_rule.consequent = new_consequent

    elif action == 'add_condition':
        # Añadir una nueva condición al antecedente
        if len(mutated_rule.antecedent) < NUM_SYMPTOMS: # Límite máximo de condiciones
            # Identificar síntomas no usados en el antecedente
            current_symptoms_in_antecedent = {cond[0] for cond in mutated_rule.antecedent}
            possible_symptoms_to_add = [s_id for s_id in range(NUM_SYMPTOMS) if s_id not in current_symptoms_in_antecedent]
            if possible_symptoms_to_add:
                sym_id = random.choice(possible_symptoms_to_add)
                req_val = random.randint(0, 1) # Valor binario para el síntoma
                mutated_rule.antecedent.append((sym_id, req_val))
        
    elif action == 'remove_condition':
        # Eliminar una condición del antecedente
        if len(mutated_rule.antecedent) > 1: # No dejar la regla vacía (antecedente vacío)
            cond_idx = random.randint(0, len(mutated_rule.antecedent) - 1)
            mutated_rule.antecedent.pop(cond_idx)
        # Si solo queda una condición, no la eliminamos para evitar reglas con antecedentes vacíos.

    elif action == 'change_condition_value':
        # Cambiar el valor requerido de una condición existente
        if mutated_rule.antecedent: # Asegurarse de que haya al menos una condición
            cond_idx = random.randint(0, len(mutated_rule.antecedent) - 1)
            sym_id, old_val = mutated_rule.antecedent[cond_idx]
            new_val = 1 - old_val # Flip for binary values (0 to 1, or 1 to 0)
            mutated_rule.antecedent[cond_idx] = (sym_id, new_val)
            
    elif action == 'change_condition_symptom':
        # Cambiar el síntoma de una condición existente (manteniendo su valor)
        if mutated_rule.antecedent:
            cond_idx = random.randint(0, len(mutated_rule.antecedent) - 1)
            old_sym_id, val = mutated_rule.antecedent[cond_idx]
            
            current_symptoms_in_antecedent = {cond[0] for cond in mutated_rule.antecedent}
            # Un síntoma nuevo debe ser diferente del actual Y no estar ya en el antecedente
            possible_new_symptoms = [s_id for s_id in range(NUM_SYMPTOMS) if s_id not in current_symptoms_in_antecedent ]
            if possible_new_symptoms and len(current_symptoms_in_antecedent) < NUM_SYMPTOMS: # Si hay síntomas que no están en el antecedente
                new_sym_id = random.choice(possible_new_symptoms)
                mutated_rule.antecedent[cond_idx] = (new_sym_id, val)

    return mutated_rule, # Retorna una tupla (mutated_rule,)

# --- 5. Función de Aptitud (Evaluación del Conjunto de Reglas) ---

def evaluate_rule_set(rule_set, dataset):
    """
    Evalúa la precisión de un conjunto de reglas (la población) en un dataset.
    Cada regla en la población es un candidato. El FITNESS del individuo será la
    precisión del *conjunto completo de reglas*.
    Este es un paso crucial en el paradigma de RBML con PE.
    """
    correct_diagnoses = 0
    
    for patient_symptoms_vector, actual_disease_id in dataset:
        votes = {d_id: 0 for d_id in range(NUM_DISEASES)}
        fired_rules_count = 0

        # Cada regla en el rule_set intenta aplicarse al paciente
        for rule in rule_set:
            if rule.applies(patient_symptoms_vector):
                # Para asegurarnos de que el consecuente de la regla está dentro del rango de enfermedades
                if 0 <= rule.consequent < NUM_DISEASES:
                    votes[rule.consequent] += 1
                else: # Si una regla tiene un consecuente inválido, no la contamos o se podría penalizar
                    pass 
                fired_rules_count += 1
        
        predicted_disease_id = None
        if fired_rules_count == 0:
            # Si ninguna regla se aplica, diagnosticar como "Desconocido" (ID 3)
            predicted_disease_id = 3 
        else:
            # Votación: la enfermedad con más votos
            max_votes = -1
            tied_diseases_ids = []
            for d_id, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    tied_diseases_ids = [d_id]
                elif count == max_votes:
                    tied_diseases_ids.append(d_id)
            
            if len(tied_diseases_ids) > 1:
                # En caso de empate, diagnosticar como "Desconocido" (ID 3)
                predicted_disease_id = 3
            else:
                predicted_disease_id = tied_diseases_ids[0] # Obtener el único ganador
        
        # Comparar con el diagnóstico real
        if predicted_disease_id == actual_disease_id:
            correct_diagnoses += 1
            
    accuracy = correct_diagnoses / len(dataset) if len(dataset) > 0 else 0
    return accuracy, # DEAP espera una tupla


# --- 6. Configuración de DEAP ---

# Toolbox ya está configurado arriba con creator.create()

toolbox = base.Toolbox()

# Atributos individuales: la función para crear una regla
toolbox.register("individual", init_rule)

# Población: una lista de reglas
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operador de evaluación: evalúa todo el conjunto de reglas
toolbox.register("evaluate", evaluate_rule_set, dataset=PREPARED_TRAINING_DATA)

# Operador de mutación: NO usamos tools.mut.. usamos nuestra función mutRule
toolbox.register("mutate", mutRule)

# Operador de selección: Seleccionamos los mu mejores entre padres e hijos
toolbox.register("select", tools.selBest) # selBest es más directo para mu+lambda

# --- 7. Algoritmo Evolutivo ---

def main():
    random.seed(42) # Para reproducibilidad

    POPULATION_SIZE = 30 # Número de reglas en el conjunto
    NUM_GENERATIONS = 10
    MU = POPULATION_SIZE # Número de padres seleccionados para la siguiente generación
    LAMBDA = POPULATION_SIZE * 3 # Número de hijos generados por mutación

    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Statistics to track the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values[0]) # Acceder al primer elemento de la tupla (accuracy,)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "max"

    print("--- Proceso de Programación Evolutiva (PE) para RBML ---")
    print(f"Población inicial de {POPULATION_SIZE} reglas. Evaluando...")

    # Evaluar la población inicial
    # La evaluación se aplica a la población como un todo para calcular la precisión del conjunto.
    # Todos los individuos en la población obtienen la misma puntuación de aptitud.
    accuracy_initial, = toolbox.evaluate(pop) 
    for ind in pop:
        ind.fitness.values = accuracy_initial, # Asignar la misma aptitud a cada regla en el conjunto

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    # Begin the evolution
    for gen in range(1, NUM_GENERATIONS + 1):
        # Generar hijos mutando individuos de la población actual (padres)
        offspring = []
        for _ in range(LAMBDA): # Generar LAMBDA hijos
            # Seleccionar UN padre aleatorio de la población actual
            parent = random.choice(pop)
            # Clonar el padre para mutarlo y añadirlo a los hijos
            offspring.append(toolbox.mutate(toolbox.clone(parent))[0])

        # Combinar población actual con hijos
        combined_population = pop + offspring
        
        # Evaluar la precisión del conjunto de reglas combinado
        # (Todos los individuos en combined_population obtienen la misma precisión)
        current_accuracy, = toolbox.evaluate(combined_population) 
        for ind in combined_population:
            ind.fitness.values = current_accuracy, # Asignar esta precisión a TODOS los individuos


        # Seleccionar la próxima generación (los MU mejores de la población combinada)
        pop = toolbox.select(combined_population, k=MU) # Selecciona los MU individuos más aptos

        # Log statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=MU+LAMBDA, **record)
        print(logbook.stream)

        # Early stopping if perfect accuracy is reached
        if current_accuracy >= 1.0:
            print(f"Precisión 100% alcanzada en la Generación {gen}")
            break

    print("\n--- Evolución Finalizada ---")
    best_rules_found = pop # El 'pop' final es el conjunto de reglas más apto

    print("\nMejor Conjunto de Reglas Encontrado:")
    for i, rule in enumerate(best_rules_found):
        # Usar str(rule) que llama a Rule.__str__ para una representación legible
        print(f"  Regla {i+1}: {str(rule)}") 
    
    final_accuracy, = toolbox.evaluate(best_rules_found)
    print(f"\nPrecisión Final del Conjunto de Reglas: {final_accuracy:.2%}")

    # --- Visualización ---
    gen = logbook.select("gen")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, max_fitness, label="Mejor Precisión")
    plt.plot(gen, avg_fitness, label="Precisión Promedio")
    plt.xlabel("Generación")
    plt.ylabel("Precisión")
    plt.title("Evolución de la Precisión del Diagnóstico Médico")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()