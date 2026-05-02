# -*- coding: utf-8 -*-
"""

@author: MARCELOFGB
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms

# --- 0. Configuración y Datos Sintéticos ---

# Definición de las categorías y el vocabulario
CATEGORIES = ["ciencia", "fantasia", "historia"]
VOCAB = list(set([
    "libro", "astronomía", "física", "cosmos", "espacio", "datos", "ciencia", "teoria", "experimento",
    "dragón", "magia", "elfo", "orco", "cuento", "espada", "fantasía", "aventura", "reino", "mago",
    "guerra", "imperio", "rey", "cultura", "antiguo", "civilización", "batalla", "histórica", "emperador", "época",
    # Añadimos algunas palabras más para dar más espacio a la diversidad
    "planeta", "estrella", "galaxia", "universo", "historia", "mito", "leyenda", "monstruo", 
    "tesoro", "mapa", "viaje", "templo", "ruinas", "arqueología", "descubrimiento", "invención",
    "observación", "cálculo", "fórmula", "poesía", "cuento", "novela", "ficción", "drama"
]))


# Representación de un documento de texto (lista de palabras)
# Simularemos documentos con palabras clave que sugieren su categoría
DOCUMENTS_RAW = [
    ("libro de astronomía y física del cosmos", "ciencia"),
    ("datos de laboratorio y teoría científica", "ciencia"),
    ("experimento en el espacio", "ciencia"),
    ("planeta, estrella y galaxia del universo, observación y cálculo", "ciencia"),
    ("libro de invención y descubrimiento científico", "ciencia"),

    ("cuento de dragón y magia en el reino", "fantasia"),
    ("elfo y orco en aventura de espada", "fantasia"),
    ("mago y su fantasía de magia", "fantasia"),
    ("mito y leyenda de monstruo y tesoro", "fantasia"),
    ("novela de fantasía con magia y aventura", "fantasia"),

    ("historia de la guerra del imperio antiguo", "historia"),
    ("rey, cultura y civilización histórica", "historia"),
    ("batalla de los emperadores de la época", "historia"),
    ("viaje a ruinas arqueológicas y templos antiguos", "historia"),
    ("drama histórico del rey y el imperio", "historia"),

    ("libro antiguo y teoría matemática", "ciencia"), # Mezcla
    ("espada de mago en la batalla", "fantasia"),   # Mezcla
    ("rey del espacio y el cosmos", "historia"),     # Mezcla
    ("cuento de poesía sobre estrellas", "ciencia"), # Nueva mezcla
    ("ficción histórica de héroes y villanos", "ficción") # Nueva mezcla
]

# Preprocesamiento simple: convertir a minúsculas y tokenizar por espacio
# Guardar como un diccionario de conteo para fácil verificación de palabras
DATA = []
for text, label in DOCUMENTS_RAW:
    words = text.lower().split()
    word_count = {word: words.count(word) for word in set(words)}
    DATA.append({"words": word_count, "label": label})

# Separar datos en entrenamiento y prueba
train_data, test_data = train_test_split(DATA, test_size=0.2, random_state=42)

# Obtener la clase mayoritaria del conjunto de entrenamiento para cuando ninguna regla coincida
train_labels = [d["label"] for d in train_data]
from collections import Counter
majority_class = Counter(train_labels).most_common(1)[0][0]
print(f"Clase mayoritaria en el set de entrenamiento: '{majority_class}'")
print(f"Número de documentos de entrenamiento: {len(train_data)}")
print(f"Número de documentos de prueba: {len(test_data)}")


# --- 1. Representación del Individuo y Reglas ---

# Definimos una "Regla" como una tupla con keywords y category
Rule = namedtuple("Rule", ["keywords", "category"])

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
creator.create("Individual", list, fitness=creator.FitnessMax)


# --- 2. Funciones de Operadores Evolutivos ---

# AJUSTE 1: Aumentar el rango de palabras clave iniciales
def generate_random_rule(vocab, categories, min_keywords=3, max_keywords=5): # Cambiado de 3 a 5
    """Genera una regla aleatoria."""
    num_keywords = random.randint(min_keywords, max_keywords)
    # Asegurarse de que num_keywords no exceda el tamaño del vocabulario
    num_keywords = min(num_keywords, len(vocab)) 
    keywords = random.sample(vocab, num_keywords)
    category = random.choice(categories)
    return Rule(frozenset(keywords), category)

def evaluate_rule_set(individual, dataset, default_category):
    """
    Evalúa el rendimiento de un conjunto de reglas (un individuo) en un dataset.
    La aptitud es la precisión (accuracy).
    """
    if not individual: # Si el individuo no tiene reglas, su aptitud es 0
        return 0.0,

    correct_predictions = 0
    for doc_data in dataset:
        doc_words = doc_data["words"]
        true_label = doc_data["label"]
        predicted_label = default_category # Por defecto, la clase mayoritaria

        # Intentar clasificar con las reglas del individuo
        for rule in individual:
            # Una regla "dispara" si TODAS sus palabras clave están en el documento
            # Es importante verificar que la regla tenga al menos una palabra clave
            if len(rule.keywords) > 0 and all(kw in doc_words for kw in rule.keywords):
                predicted_label = rule.category
                break # La primera regla que coincide gana

        if predicted_label == true_label:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(dataset)
    return accuracy, # La aptitud debe ser una tupla

# AJUSTE 2: Aumentar las probabilidades de mutación
def mutate_rule_set(individual, vocab, categories, indpb, rule_add_prob=0.15, rule_del_prob=0.08):
    """
    Función de mutación para un conjunto de reglas.
    Puede añadir/eliminar una regla, o mutar una regla existente.
    """
    # Mutación de la estructura del conjunto de reglas
    # ¡Importante! Si creamos un nuevo individuo (aunque sea con las mismas reglas), debe ser del tipo Individual
    mutated_individual = creator.Individual(individual) # Convertir a creator.Individual

    if random.random() < rule_add_prob: # Añadir una nueva regla
        mutated_individual.append(generate_random_rule(vocab, categories))
    
    # Asegurarse de que siempre haya al menos una regla, si queremos evitar individuos vacíos
    if mutated_individual and random.random() < rule_del_prob and len(mutated_individual) > 1: # Eliminar una regla (solo si hay más de 1)
        mutated_individual.pop(random.randrange(len(mutated_individual)))

    # Mutación de reglas existentes
    for i in range(len(mutated_individual)):
        if random.random() < indpb: # Probabilidad de mutar una regla específica (indpb se aumenta en el main)
            rule = mutated_individual[i]
            
            mutation_type = random.choice(["change_category", "add_keyword", "remove_keyword", "change_keyword"])
            
            if mutation_type == "change_category":
                new_category = random.choice(categories)
                mutated_individual[i] = Rule(rule.keywords, new_category)
            
            elif mutation_type == "add_keyword": 
                available_words = [w for w in vocab if w not in rule.keywords]
                if available_words:
                    new_kw = random.choice(available_words)
                    new_keywords = frozenset(rule.keywords.union({new_kw}))
                    mutated_individual[i] = Rule(new_keywords, rule.category)
            
            elif mutation_type == "remove_keyword":
                if len(rule.keywords) > 1: # No eliminar todas las palabras clave si es la única
                    kw_to_remove = random.choice(list(rule.keywords))
                    new_keywords = frozenset(set(rule.keywords) - {kw_to_remove})
                    mutated_individual[i] = Rule(new_keywords, rule.category)
                elif len(rule.keywords) == 1: # Si solo hay una palabra, cambiarla en lugar de eliminar
                    mutation_type = "change_keyword" # Forzar a cambiar si solo queda una
            
            elif mutation_type == "change_keyword":
                if len(mutated_individual[i].keywords) > 0: # Accedemos nuevamente para asegurar que la regla actual no fue eliminada/reemplazada por otra mutación anterior en este mismo bucle
                    old_kw = random.choice(list(mutated_individual[i].keywords))
                    available_words = [w for w in vocab if w not in mutated_individual[i].keywords or w == old_kw]
                    if available_words:
                        new_kw = random.choice(available_words)
                        new_keywords = frozenset((set(mutated_individual[i].keywords) - {old_kw}).union({new_kw}))
                        mutated_individual[i] = Rule(new_keywords, rule.category)
                        
    
    # Crear una copia modificable del individuo, asegurándose de que sea de tipo creator.Individual
    temp_individual = creator.Individual(individual)

    # ... (aplicar todas las mutaciones a `temp_individual` ) ...
    
    # Mutación de la estructura del conjunto de reglas
    if random.random() < rule_add_prob: # Añadir una nueva regla
        temp_individual.append(generate_random_rule(vocab, categories))
    
    if temp_individual and random.random() < rule_del_prob and len(temp_individual) > 1: # Eliminar una regla (solo si hay más de 1)
        temp_individual.pop(random.randrange(len(temp_individual)))

    # Iterar y aplicar mutaciones a las reglas dentro de temp_individual
    for i in range(len(temp_individual)):
        if random.random() < indpb:
            rule = temp_individual[i]
            mutation_type = random.choice(["change_category", "add_keyword", "remove_keyword", "change_keyword"])

            if mutation_type == "change_category":
                new_category = random.choice(categories)
                temp_individual[i] = Rule(rule.keywords, new_category)
            
            elif mutation_type == "add_keyword":
                available_words = [w for w in vocab if w not in rule.keywords]
                if available_words:
                    new_kw = random.choice(available_words)
                    new_keywords = frozenset(rule.keywords.union({new_kw}))
                    temp_individual[i] = Rule(new_keywords, rule.category)
            
            elif mutation_type == "remove_keyword":
                if len(rule.keywords) > 1:
                    kw_to_remove = random.choice(list(rule.keywords))
                    new_keywords = frozenset(set(rule.keywords) - {kw_to_remove})
                    temp_individual[i] = Rule(new_keywords, rule.category)
                elif len(rule.keywords) == 1:
                    mutation_type = "change_keyword" # Forzar a cambiar si solo queda una
            
            if mutation_type == "change_keyword": # Re-check mutation_type for chained logic
                if temp_individual[i].keywords: # Ensure there are keywords to change
                    old_kw = random.choice(list(temp_individual[i].keywords))
                    available_words = [w for w in vocab if w not in temp_individual[i].keywords or w == old_kw]
                    if available_words:
                        new_kw = random.choice(available_words)
                        new_keywords = frozenset((set(temp_individual[i].keywords) - {old_kw}).union({new_kw}))
                        temp_individual[i] = Rule(new_keywords, rule.category)
    
    return temp_individual, # Devuelve el nuevo individuo de tipo creator.Individual


def crossover_rule_set(ind1, ind2):
    """
    Cruza de dos conjuntos de reglas.
    Intercambia un número aleatorio de reglas entre los dos individuos.
    """
    # CORRECCIÓN CLAVE: Ambos descendientes deben ser objetos de tipo creator.Individual
    # Si no son creados como creator.Individual, DEAP no podrá acceder a .fitness
    offspring1 = creator.Individual(ind1) # Convertir a creator.Individual
    offspring2 = creator.Individual(ind2) # Convertir a creator.Individual

    if not offspring1 or not offspring2:
        return offspring1, offspring2
    
    len1 = len(offspring1)
    len2 = len(offspring2)
    
    if len1 == 0 or len2 == 0:
        return offspring1, offspring2

    cx_point1_start = random.randint(0, len1)
    cx_point1_end = random.randint(0, len1)
    if cx_point1_start > cx_point1_end: cx_point1_start, cx_point1_end = cx_point1_end, cx_point1_start

    cx_point2_start = random.randint(0, len2)
    cx_point2_end = random.randint(0, len2)
    if cx_point2_start > cx_point2_end: cx_point2_start, cx_point2_end = cx_point2_end, cx_point2_start

    # Intercambio de las porciones de reglas
    temp_segment1 = list(offspring1[cx_point1_start:cx_point1_end]) # Casteo a list para manipular
    temp_segment2 = list(offspring2[cx_point2_start:cx_point2_end]) # Casteo a list para manipular

    # La asignación slices funciona in-place para listas DEAP también
    offspring1[cx_point1_start:cx_point1_end] = temp_segment2
    offspring2[cx_point2_start:cx_point2_end] = temp_segment1

    return offspring1, offspring2 # Ambos deben ser creator.Individual


# --- 3. Registro de Operadores DEAP (Toolbox) ---

toolbox = base.Toolbox()

# Inicialización
toolbox.register("attr_rule", generate_random_rule, VOCAB, CATEGORIES) 
# tools.initRepeat: crea un individuo (lista de reglas) con un número aleatorio de reglas
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rule, n=random.randint(1, 7)) # Permite hasta 7 reglas inicialmente
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores Evolutivos
toolbox.register("evaluate", evaluate_rule_set, dataset=train_data, default_category=majority_class)
toolbox.register("mate", crossover_rule_set)
toolbox.register("mutate", mutate_rule_set, vocab=VOCAB, categories=CATEGORIES, indpb=0.3) # AJUSTE 2: Aumentado indpb de 0.2 a 0.3
toolbox.register("select", tools.selTournament, tournsize=3)


# --- 4. Algoritmo Evolutivo Principal ---

def main():
    # AJUSTE 3: Parámetros del Algoritmo Evolutivo
    POPULATION_SIZE = 100 # Aumentado de 100 a 200
    GENERATIONS = 25    # Aumentado de 50 a 100
    CX_PROB = 0.8        # Probabilidad de cruce (puede mantenerse)
    MUT_PROB = 0.4       # Probabilidad de mutación (Aumentado de 0.3 a 0.4)
    
    # Inicialización de la población
    population = toolbox.population(n=POPULATION_SIZE)
    
    # Hall of Fame para guardar al mejor individuo de todas las generaciones
    hof = tools.HallOfFame(1)
    
    # Estadísticas para el seguimiento de la evolución
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Se crea el Logbook internamente en DEAP o se pasa como objeto preexistente.
    # Si quieres iniciar el logbook con un header, lo haces antes de la llamada.
    logbook = tools.Logbook() 
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    print("Inicio del proceso evolutivo...")
    print(f"Población inicial: {POPULATION_SIZE} individuos.")
    
    # El algoritmo eaSimple maneja los pasos de evaluación, selección, cruce, mutación y reemplazo
    # 'logbook' no es un argumento de entrada, sino una de las variables devueltas.
    population, logbook = algorithms.eaSimple(population, toolbox, 
                                              cxpb=CX_PROB, mutpb=MUT_PROB, 
                                              ngen=GENERATIONS, 
                                              stats=stats, 
                                              halloffame=hof, 
                                              verbose=True) 
    
    print("\nFin del proceso evolutivo.")
    
    # --- 5. Resultados y Visualización ---

    best_individual = hof[0]
    print("\n--- Mejor Conjunto de Reglas Encontrado (entrenamiento) ---")
    # Mostrar todas las reglas del mejor individuo
    if best_individual:
        for i, rule in enumerate(best_individual):
            print(f"Regla {i+1}: SI todas estas palabras están presentes: {list(rule.keywords)} ENTONCES es categoría: '{rule.category}'")
        print(f"Número total de reglas: {len(best_individual)}")
        print(f"Aptitud (Accuracy en entrenamiento): {best_individual.fitness.values[0]:.4f}")
    else:
        print("No se encontró un conjunto de reglas apto.")

    # Evaluar el mejor individuo en el conjunto de prueba
    if best_individual:
        test_accuracy, = evaluate_rule_set(best_individual, test_data, majority_class)
        print(f"Aptitud (Accuracy en PRUEBA): {test_accuracy:.4f}")
    else:
        test_accuracy = 0.0 # O algún otro valor por defecto si no hay reglas

    # Extraer estadísticas para la gráfica
    gen = logbook.select("gen")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, max_fitness, label="Máxima Aptitud")
    plt.plot(gen, avg_fitness, label="Aptitud Media")
    plt.xlabel("Generación")
    plt.ylabel("Aptitud (Accuracy)")
    plt.title("Evolución de la Aptitud a lo largo de las Generaciones")
    plt.grid(True)
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Accuracy en Prueba: {test_accuracy:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Clasificación de un Ejemplo Nuevo ---")
    new_doc_text = "mi libro sobre los dragones en la historia es genial"
    # Preprocesar el nuevo documento de la misma manera que los datos de entrenamiento
    new_doc_words = {word: 1 for word in new_doc_text.lower().split()}
    
    predicted_label = majority_class # Asignar la clase mayoritaria por defecto
    if best_individual:
        for rule in best_individual:
            if len(rule.keywords) > 0 and all(kw in new_doc_words for kw in rule.keywords):
                predicted_label = rule.category
                break
    print(f"Documento: '{new_doc_text}'")
    print(f"Palabras en documento: {list(new_doc_words.keys())}")
    print(f"Clasificado como: '{predicted_label}'")

if __name__ == "__main__":
    main()