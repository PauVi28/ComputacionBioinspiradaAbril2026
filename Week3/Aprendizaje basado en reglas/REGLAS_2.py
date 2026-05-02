# -*- coding: utf-8 -*-
"""

@author: MARCELOFGB
"""
import random
from deap import base, creator, tools, algorithms
import statistics # <--- ¡IMPORTANTE! Importar statistics

# --- 1. Definición del Contexto (Palabras Clave y Diagnósticos) ---
# Un vocabulario de posibles síntomas (palabras clave)
KEYWORDS = [
  'fiebre', 'tos', 'dolor cabeza', 'nauseas', 'vomitos', 'dolor garganta',
  'fatiga', 'escalofrios', 'dolor muscular', 'erupcion', 'congestion',
  'mareos', 'dolor pecho', 'dificultad respirar', 'ganglios inflamados'
]

# Posibles diagnósticos (clases)
CLASSES = [
  'Influenza', 'Resfriado Comun', 'Migraña', 'Gastroenteritis',
  'Alergia', 'Bronquitis', 'Varicela'
]

# --- 2. Conjunto de Datos de Entrenamiento (Simulado) ---
# Formato: (texto_documento, diagnostico_real)
# Los textos deben contener palabras clave y estar en minúsculas para coincidir fácilmente.
DATASET = [
  ("paciente con fiebre, tos, fatiga y dolor muscular", 'Influenza'),
  ("congestion nasal, tos y dolor garganta leve", 'Resfriado Comun'),
  ("dolor cabeza intenso, nauseas y mareos", 'Migraña'),
  ("vomitos, nauseas y dolor de estomago", 'Gastroenteritis'),
  ("erupcion en la piel con picazon y fiebre leve", 'Varicela'),
  ("fiebre alta, tos persistente y dificultad para respirar", 'Bronquitis'),
  ("estornudos, ojos llorosos y congestion despues de estar en el jardin", 'Alergia'),
  ("dolor de cabeza, algo de fiebre y escalofrios", 'Influenza'),
  ("solo tos y congestion leve", 'Resfriado Comun'),
  ("vomitos frecuentes y nauseas severas", 'Gastroenteritis'),
  ("dolor de garganta, algo de tos y fatiga", 'Resfriado Comun'),
  ("fiebre, dolor muscular y ganglios inflamados", 'Influenza'),
  ("tos seca y dificultad para respirar", 'Bronquitis'),
  ("dolor de cabeza fuerte, sensibilidad a la luz", 'Migraña'),
]

# --- 3. Configuración de DEAP ---

# Definir el tipo de aptitud (maximizar, ya que es precisión)
# y el tipo de individuo (una lista de reglas)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Inicializar un Toolbox
toolbox = base.Toolbox()

# --- 4. Generadores para la Inicialización de Individuos ---

# Generador de una lista de palabras clave para un antecedente (una regla)
def generate_antecedent(min_keywords=3, max_keywords=5):
  num_keywords = random.randint(min_keywords, max_keywords)
  # Evitar palabras clave duplicadas en el antecedente de una misma regla
  return random.sample(KEYWORDS, min(num_keywords, len(KEYWORDS)))

# Generador de un consecuente (una clase/diagnóstico)
def generate_consequent():
  return random.choice(CLASSES)

# Generador de una regla (antecedente, consecuente)
def generate_rule():
  return (generate_antecedent(), generate_consequent())

# Generador de un individuo (una lista de reglas)
# Un individuo tendrá entre 1 y 5 reglas inicialmente
def generate_individual(min_rules=1, max_rules=5):
  num_rules = random.randint(min_rules, max_rules)
  return [generate_rule() for _ in range(num_rules)]

toolbox.register("attr_rule", generate_rule)
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 5. Función de Aptitud (Evaluation) ---
def evaluate_ruleset(individual, dataset):
  correct_predictions = 0
  total_documents = len(dataset)

  if not individual: # Si el individuo no tiene reglas, la precisión es 0
    return (0.0,)

  for doc_text, actual_diagnosis in dataset:
    predicted_diagnosis = None
    doc_text_lower = doc_text.lower() # Normalizar el texto del documento

    # Iterar a través de las reglas para encontrar una coincidencia
    for antecedent_keywords, consequent_class in individual:
      # Comprobar si TODAS las palabras clave del antecedente están en el documento
      all_keywords_present = True
      for keyword in antecedent_keywords:
        if keyword not in doc_text_lower: # Simple substring check
          all_keywords_present = False
          break
      
      if all_keywords_present:
        predicted_diagnosis = consequent_class
        break # La primera regla que coincide gana

    if predicted_diagnosis == actual_diagnosis:
      correct_predictions += 1
  
  accuracy = correct_predictions / total_documents
  return (accuracy,) # DEAP espera una tupla para el fitness

toolbox.register("evaluate", evaluate_ruleset, dataset=DATASET)


# --- 6. Operadores Genéticos Personalizados ---

def mutate_individual(individual, indpb_rule=0.2, indpb_keyword=0.2, indpb_class=0.3, indpb_add_remove_rule=0.1):
  """
  Mutación para un individuo (conjunto de reglas).
  Puede:
   1. Añadir/quitar una regla (indpb_add_remove_rule)
   2. Mutar una regla existente (indpb_rule)
    a. Mutar el antecedente: añadir/quitar una palabra clave (indpb_keyword)
    b. Mutar el consecuente: cambiar el diagnóstico (indpb_class)
  """
  
  # 1. Mutación de adición/eliminación de regla
  if random.random() < indpb_add_remove_rule:
    if len(individual) > 1 and random.random() < 0.5: # Quitar una regla
      index_to_remove = random.randrange(len(individual))
      individual.pop(index_to_remove)
    else: # Añadir una regla
      individual.append(generate_rule())
  
  # 2. Mutación de reglas existentes
  for i in range(len(individual)):
    if random.random() < indpb_rule: # Decidir si mutar esta regla específica
      antecedent, consequent = individual[i]

      # Mutar el antecedente (palabras clave)
      if random.random() < indpb_keyword:
        if antecedent and random.random() < 0.5: # Quitar una palabra clave
          if len(antecedent) > 1: # Asegurarse de que no quede vacío
            antecedent.pop(random.randrange(len(antecedent)))
        else: # Añadir una palabra clave
          # Selecciona una palabra clave que no esté ya en el antecedente
          possible_adds = list(set(KEYWORDS) - set(antecedent))
          if possible_adds:
            antecedent.append(random.choice(possible_adds))
        antecedent = sorted(list(set(antecedent))) # Eliminar duplicados y ordenar para consistencia

      # Mutar el consecuente (clase/diagnóstico)
      if random.random() < indpb_class:
        new_consequent = random.choice(CLASSES)
        while new_consequent == consequent and len(CLASSES) > 1: # Asegurar que sea diferente si es posible
          new_consequent = random.choice(CLASSES)
        consequent = new_consequent
      
      individual[i] = (antecedent, consequent)
  
  # Asegurarse de que el individuo no esté vacío (si se eliminaron todas las reglas)
  if not individual:
    individual.append(generate_rule())

  return individual,


def crossover_individual(ind1, ind2, indpb_rule=0.5):
  """
  Cruce tipo "one-point crossover" a nivel de reglas entre dos individuos.
  Intercambia un segmento de reglas entre los dos individuos.
  """
  size1 = len(ind1)
  size2 = len(ind2)

  if size1 < 2 or size2 < 2: # No hay suficiente material para cruzar
    return ind1, ind2

  cxpoint1 = random.randint(1, size1 - 1)
  cxpoint2 = random.randint(1, size2 - 1)

  # Intercambiar las "colas" de las listas de reglas
  ind1[cxpoint1:], ind2[cxpoint2:] = ind2[cxpoint2:], ind1[cxpoint1:]

  return ind1, ind2


toolbox.register("mutate", mutate_individual)
toolbox.register("mate", crossover_individual)
toolbox.register("select", tools.selTournament, tournsize=3)


# --- 7. Bucle Principal de la Evolución ---
def main():
  random.seed(42) # Para reproducibilidad

  population_size = 60
  num_generations = 50
  cx_prob = 0.0  # Probabilidad de cruce
  mut_prob = 0.8  # Probabilidad de mutación

  # Crear la población inicial
  pop = toolbox.population(n=population_size)

  # Objeto para almacenar las mejores soluciones encontradas
  hof = tools.HallOfFame(1)

  # Estadísticas
  # CORRECCIÓN CLAVE: Acceder al valor numérico de la aptitud 'ind.fitness.values[0]'
  stats = tools.Statistics(lambda ind: ind.fitness.values[0]) 
  stats.register("avg", statistics.mean)
  stats.register("std", statistics.stdev)
  stats.register("min", min)
  stats.register("max", max)

  print("Iniciando evolución...")
  # El algoritmo eaSimple es un algoritmo genético clásico de tipo "generacional"
  # que trabaja con una población de tamaño fijo y reemplaza la generación completa
  # en cada paso.
  pop, log = algorithms.eaSimple(pop, toolbox, cx_prob, mut_prob, num_generations, 
                                stats=stats, halloffame=hof, verbose=True)

  print("\n--- Evolución Terminada ---")
  print(f"Mejor individuo encontrado (precisión: {hof[0].fitness.values[0]:.2f}):")

  for i, rule in enumerate(hof[0]):
    antecedent_str = " Y ".join(rule[0])
    print(f"  Regla {i+1}: SI (documento contiene '{antecedent_str}') ENTONCES '{rule[1]}'")
  
  print("\n--- ¡Probando el mejor individuo en el conjunto de datos de entrenamiento! ---")
  final_accuracy = evaluate_ruleset(hof[0], DATASET)[0]
  print(f"Precisión final del mejor individuo en los datos de entrenamiento: {final_accuracy:.2f}")

  # Ejemplo de cómo usar el mejor individuo para una nueva 'predicción'
  print("\n--- Ejemplos de Predicción ---")
  test_docs = [
    "Un paciente se queja de fiebre, tos y dolor de garganta.",
    "Sufre de dolor de cabeza agudo y nauseas",
    "Erupción en la piel y picazón intensa",
    "Tos seca, fatiga, pero sin fiebre", # Podría no ser clasificado o clasif. mal
    "El sujeto presenta dificultad para respirar y tos",
    "Solo tiene dolor muscular leve" # Muy genérico, difícil de clasificar
  ]

  def predict_with_rules(document_text, ruleset):
    doc_text_lower = document_text.lower()
    
    for antecedent_keywords, consequent_class in ruleset:
      all_keywords_present = True
      for keyword in antecedent_keywords:
        if keyword not in doc_text_lower:
          all_keywords_present = False
          break
      if all_keywords_present:
        return consequent_class
    return "Desconocido" # Si ninguna regla coincide

  for doc in test_docs:
    prediction = predict_with_rules(doc, hof[0])
    print(f"Documento: '{doc}'\n  Predicción: '{prediction}'\n")

if __name__ == "__main__":
  main()