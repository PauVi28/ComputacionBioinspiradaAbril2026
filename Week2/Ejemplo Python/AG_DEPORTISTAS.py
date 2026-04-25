# -*- coding: utf-8 -*-
"""


@author: MARCELOFGB
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# --- Configuración de Parámetros del Algoritmo Genético ---
POPULATION_SIZE = 1000     # Número de individuos en cada generación
CHROMOSOME_LENGTH = 15      # Número de genes por cromosoma (características del deportista)
GENE_MIN_VALUE = 0          # Valor mínimo que puede tener un gen
GENE_MAX_VALUE = 100        # Valor máximo que puede tener un gen
GENERATIONS = 10           # Número máximo de generaciones a evolucionar
MUTATION_RATE = 0.5        # Probabilidad de que un gen mute
CROSSOVER_RATE = 0.8        # Probabilidad de que ocurra un cruce entre padres
ELITISM_COUNT = 2           # Número de los mejores individuos que pasan directamente a la siguiente generación
TOURNAMENT_SIZE = 10         # Tamaño del grupo para la selección por torneo

# Nombres de los genes para una mejor legibilidad en los resultados
GENE_NAMES = [
    "Velocidad", "Fuerza", "Resistencia", "Agilidad", "Salto",
    "Flexibilidad", "Equilibrio", "Reacción", "Potencia", "VO2 Máx",
    "Recuperación", "Concentración", "Resiliencia Lesiones",
    "Táctica", "Disciplina"
]

# --- Definición del Perfil Ideal del Deportista (Función Objetivo) ---
def get_target_profile(training_stage, month, weight_category, age_group):
    """
    Define el perfil ideal de características para un deportista
    basado en la etapa de entrenamiento, el mes, la categoría de peso y el grupo de edad.
    Este es el 'objetivo' que el algoritmo genético intentará alcanzar.
    """
    # Establece un perfil base general (valores intermedios)
    target = [50] * CHROMOSOME_LENGTH

    # Modificaciones del perfil según la Etapa de Entrenamiento
    if training_stage == 'pre-season':
        # En pretemporada, se busca construir una base física sólida.
        target[0] = 70  # Velocidad
        target[1] = 80  # Fuerza
        target[2] = 75  # Resistencia
        target[8] = 75  # Potencia
        target[9] = 70  # VO2 Máx
        target[12] = 60 # Resiliencia a Lesiones (se trabaja en prevención)
        target[14] = 80 # Disciplina (importante para sentar bases)
    elif training_stage == 'in-season':
        # En temporada, se prioriza el pico de rendimiento y habilidades específicas.
        target[0] = 90  # Velocidad (pico)
        target[1] = 85  # Fuerza (mantenimiento)
        target[2] = 90  # Resistencia (pico)
        target[3] = 90  # Agilidad (pico)
        target[4] = 88  # Salto (pico)
        target[8] = 90  # Potencia (pico)
        target[9] = 92  # VO2 Máx (pico)
        target[11] = 95 # Concentración (crítica para la competición)
        target[13] = 85 # Táctica (aplicación en competición)
    elif training_stage == 'off-season':
        # En la post-temporada, el enfoque es la recuperación y prevención de lesiones, con menor intensidad.
        target[2] = 40  # Resistencia (descanso activo)
        target[10] = 80 # Recuperación (énfasis)
        target[12] = 90 # Resiliencia a Lesiones (recuperación y fortalecimiento)
        target[14] = 70 # Disciplina (mantener hábitos básicos)

    # Modificaciones del perfil según el Mes del Año
    # Ejemplo: Si el mes es julio o agosto, se asume un pico de competición
    if month == 7 or month == 8:
        for i in [0, 1, 2, 3, 4, 8, 9, 11, 13]: # Atributos de rendimiento máximo
            target[i] = min(GENE_MAX_VALUE, target[i] + 5) # Pequeño impulso para la competición

    # Modificaciones del perfil según la Categoría de Peso
    if weight_category == 'light':
        target[0] = min(GENE_MAX_VALUE, target[0] + 10) # Mayor velocidad
        target[3] = min(GENE_MAX_VALUE, target[3] + 10) # Mayor agilidad
        target[1] = max(GENE_MIN_VALUE, target[1] - 10) # Menos fuerza bruta esperada
    elif weight_category == 'heavy':
        target[1] = min(GENE_MAX_VALUE, target[1] + 10) # Mayor fuerza
        target[8] = min(GENE_MAX_VALUE, target[8] + 10) # Mayor potencia
        target[0] = max(GENE_MIN_VALUE, target[0] - 10) # Menos velocidad esperada

    # Modificaciones del perfil según el Grupo de Edad
    if age_group == 'junior':
        target[14] = min(GENE_MAX_VALUE, target[14] + 15) # Enfoque en disciplina y fundamentos
        target[12] = min(GENE_MAX_VALUE, target[12] + 10) # Resiliencia a lesiones (aprender límites del cuerpo)
        target[11] = max(GENE_MIN_VALUE, target[11] - 5) # Concentración puede ser naturalmente menor
    elif age_group == 'veteran':
        target[10] = min(GENE_MAX_VALUE, target[10] + 15) # Recuperación prioritaria
        target[13] = min(GENE_MAX_VALUE, target[13] + 15) # Mayor acumen estratégico
        target[2] = max(GENE_MIN_VALUE, target[2] - 10) # Resistencia puede declinar naturalmente

    # Asegura que todos los valores estén dentro de los límites definidos (0-100)
    target = [max(GENE_MIN_VALUE, min(GENE_MAX_VALUE, int(g))) for g in target]
    return target

# --- Funciones Esenciales del Algoritmo Genético ---

# 1. Función de Aptitud (Fitness Function)
def calculate_fitness(individual, target_profile):
    """
    Calcula la aptitud de un individuo.
    La aptitud se basa en la inversa de la suma de las diferencias absolutas entre
    los genes del individuo y el perfil objetivo.
    Cuanto menor sea la diferencia total, mayor será la aptitud (más cercano al ideal).
    Se añade un 1 en el denominador para evitar división por cero y asegurar que la aptitud
    sea 1 para una coincidencia perfecta.
    """
    differences = [abs(individual[i] - target_profile[i]) for i in range(CHROMOSOME_LENGTH)]
    total_difference = sum(differences) # Suma total de las diferencias
    return 1 / (1 + total_difference) # La aptitud es 1/(1 + diferencia total)

# 2. Inicialización de la Población
def create_individual():
    """Crea un individuo (cromosoma) con 15 genes, cada uno con un valor aleatorio."""
    return [random.randint(GENE_MIN_VALUE, GENE_MAX_VALUE) for _ in range(CHROMOSOME_LENGTH)]

def create_initial_population(size):
    """Crea una población inicial de un tamaño dado."""
    return [create_individual() for _ in range(size)]

# 3. Selección (Selection)
def select_parents(population, fitnesses, num_parents):
    """
    Selecciona padres utilizando la estrategia de selección por torneo.
    Se crea un "torneo" seleccionando aleatoriamente 'TOURNAMENT_SIZE' individuos,
    y el individuo con la mejor aptitud de ese grupo es elegido como padre.
    Este proceso se repite hasta obtener el número deseado de padres.
    """
    parents = []
    # Emparejar individuos con sus aptitudes para facilitar la selección.
    population_with_fitness = list(zip(population, fitnesses))

    for _ in range(num_parents):
        # Selecciona aleatoriamente competidores para el torneo
        tournament_contenders = random.sample(population_with_fitness, TOURNAMENT_SIZE)
        # Elige al competidor con la mejor aptitud como ganador del torneo
        winner_individual, _ = max(tournament_contenders, key=lambda x: x[1])
        parents.append(winner_individual)
    return parents

# 4. Reproducción (Cruce / Crossover)
def crossover(parent1, parent2):
    """
    Realiza el cruce de un punto entre dos padres para crear dos hijos.
    Si se cumple la tasa de cruce, se elige un punto aleatorio y se intercambian
    los segmentos de los cromosomas de los padres. De lo contrario, los hijos
    son copias idénticas de sus padres.
    """
    if random.random() < CROSSOVER_RATE:
        # Elige un punto de cruce aleatorio, evitando los extremos (0 y CHROMOSOME_LENGTH)
        crossover_point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        # Si no hay cruce, los hijos son copias de los padres
        return parent1, parent2

# 5. Mutación (Mutation)
def mutate(individual):
    """
    Aplica mutación a un individuo con una probabilidad dada para cada gen.
    Si un gen muta, su valor se cambia a uno nuevo aleatorio dentro del rango permitido.
    Esto ayuda a explorar nuevas soluciones y evitar mínimos locales.
    """
    mutated_individual = individual[:] # Crear una copia para no modificar el original
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < MUTATION_RATE:
            # Reemplaza el gen con un nuevo valor aleatorio
            mutated_individual[i] = random.randint(GENE_MIN_VALUE, GENE_MAX_VALUE)
    return mutated_individual

# --- Bucle Principal del Algoritmo Genético ---
def run_genetic_algorithm(target_profile_params):
    """
    Ejecuta el algoritmo genético completo siguiendo el diagrama de flujo.
    """
    print(f"\n--- Iniciando Algoritmo Genético para las condiciones: {target_profile_params} ---")

    # Extraer parámetros para definir el perfil objetivo
    training_stage = target_profile_params['training_stage']
    month = target_profile_params['month']
    weight_category = target_profile_params['weight_category']
    age_group = target_profile_params['age_group']

    # Obtener el perfil ideal de características para las condiciones dadas
    target_profile = get_target_profile(training_stage, month, weight_category, age_group)
    print("\n--- Perfil Objetivo Ideal para estas condiciones ---")
    for i, val in enumerate(target_profile):
        print(f"{GENE_NAMES[i]}: {val}")
    print("-" * 40)

    # Listas para almacenar el historial de aptitud para las gráficas
    best_fitness_history = []  # Mejor aptitud en cada generación
    avg_fitness_history = []   # Aptitud promedio en cada generación

    # 1. Población inicial
    population = create_initial_population(POPULATION_SIZE)
    print(f"Población inicial de {POPULATION_SIZE} individuos creada.")

    for generation in range(GENERATIONS):
        # 2. Evaluación de la población
        fitnesses = [calculate_fitness(ind, target_profile) for ind in population]

        current_best_fitness = max(fitnesses)
        current_avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(current_avg_fitness)

        # Criterio de Parada
        # Aquí, el criterio es un número máximo de generaciones O que se alcance una aptitud muy alta.
        if current_best_fitness >= 0.99 or generation == GENERATIONS - 1:
            print(f"\n--- Generación {generation + 1}/{GENERATIONS} ---")
            print(f"Mejor aptitud: {current_best_fitness:.4f}, Aptitud promedio: {current_avg_fitness:.4f}")
            print("Criterio de parada alcanzado (máx. generaciones o aptitud casi perfecta).")
            break # Salir del bucle

        # Si NO se cumple el Criterio de Parada, continuamos con la evolución:
        # Imprimir progreso cada 10 generaciones o al inicio
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"Generación {generation + 1}/{GENERATIONS}: Mejor aptitud = {current_best_fitness:.4f}, Aptitud promedio = {current_avg_fitness:.4f}")

        # 3. Selección
        # Primero, aplicar elitismo: los mejores individuos pasan directamente
        elites = sorted(list(zip(population, fitnesses)), key=lambda x: x[1], reverse=True)[:ELITISM_COUNT]
        elites_individuals = [elite[0] for elite in elites]

        # Seleccionar padres para la reproducción del resto de la población
        # Necesitamos generar (POPULATION_SIZE - ELITISM_COUNT) nuevos individuos
        num_to_breed = POPULATION_SIZE - ELITISM_COUNT
        # La selección se hace de la población actual (incluyendo posibles elites, aunque no directamente)
        parents = select_parents(population, fitnesses, num_to_breed)

        # 4. Reproducción y 5. Mutación (para crear la Nueva Generación)
        next_population = []
        # Añadir los individuos élite a la nueva población primero
        next_population.extend(elites_individuals)

        # Generar el resto de la nueva población a través de cruce y mutación
        # Se generan pares de hijos hasta que la nueva población alcance el tamaño deseado.
        while len(next_population) < POPULATION_SIZE:
            # Elegir dos padres aleatorios de la lista de padres seleccionados
            p1, p2 = random.sample(parents, 2)
            child1, child2 = crossover(p1, p2) # Realizar cruce
            mutated_child1 = mutate(child1)     # Mutar al primer hijo
            mutated_child2 = mutate(child2)     # Mutar al segundo hijo
            next_population.append(mutated_child1)
            # Asegurarse de no exceder el tamaño de la población si solo queda espacio para un hijo
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutated_child2)

        # La nueva población reemplaza a la antigua
        population = next_population

    # 6. Población Final y Mejor Individuo
    # Re-evaluar la población final para obtener el mejor individuo al final de la ejecución
    final_fitnesses = [calculate_fitness(ind, target_profile) for ind in population]
    best_final_individual_idx = np.argmax(final_fitnesses) # Índice del individuo con mejor aptitud
    best_final_individual = population[best_final_individual_idx]
    best_final_fitness = final_fitnesses[best_final_individual_idx]

    print("\n--- ¡Algoritmo Finalizado! ---")
    print("\n--- Mejor Individuo Encontrado en la Población Final ---")
    for i, val in enumerate(best_final_individual):
        print(f"{GENE_NAMES[i]}: {val} (Target: {target_profile[i]})")
    print(f"Aptitud del mejor individuo: {best_final_fitness:.4f} (Máx. posible es 1.0)")

    # Calcular la suma de diferencias para una interpretación más clara de la 'distancia'
    diffs = [abs(best_final_individual[i] - target_profile[i]) for i in range(CHROMOSOME_LENGTH)]
    print(f"Suma de diferencias absolutas al perfil objetivo: {sum(diffs)}")

    # Retornar datos para las gráficas
    return best_fitness_history, avg_fitness_history, best_final_individual, target_profile

# --- Ejecución del Algoritmo y Visualización de Resultados ---

# Define un escenario específico para el deportista
# Puedes cambiar estos parámetros para ver cómo varía el perfil ideal y los resultados del AG
simulation_params_1 = {
    'training_stage': 'off-season',
    'month': 7,
    'weight_category': 'medium',
    'age_group': 'junior'
}

# Ejecutar el algoritmo genético para el escenario definido
best_fit_hist_1, avg_fit_hist_1, best_ind_final_1, target_profile_used_1 = run_genetic_algorithm(simulation_params_1)

# --- Generación de Gráficas Didácticas ---

# Gráfica 1: Evolución de la Aptitud a lo largo de las Generaciones
plt.figure(figsize=(12, 6))
plt.plot(best_fit_hist_1, label='Mejor Aptitud por Generación', color='green')
plt.plot(avg_fit_hist_1, label='Aptitud Promedio por Generación', color='orange', linestyle='--')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.title('Evolución de la Aptitud del Deportista a lo largo de las Generaciones')
plt.legend()
plt.grid(True)
plt.show()

# Gráfica 2: Comparación de Características: Mejor Individuo Encontrado vs. Perfil Objetivo Ideal
plt.figure(figsize=(14, 7))
bar_width = 0.35
index = np.arange(CHROMOSOME_LENGTH) # Posiciones para las barras

plt.bar(index, best_ind_final_1, bar_width, label='Mejor Individuo Final', color='skyblue')
plt.bar(index + bar_width, target_profile_used_1, bar_width, label='Perfil Objetivo Ideal', color='lightcoral')

plt.xlabel('Característica del Deportista')
plt.ylabel('Valor (0-100)')
plt.title('Comparación: Características del Mejor Individuo Encontrado vs. Perfil Objetivo Ideal')
plt.xticks(index + bar_width / 2, GENE_NAMES, rotation=45, ha='right') # Nombres de los genes en el eje X
plt.legend()
plt.tight_layout() # Ajusta el diseño para evitar solapamientos
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("\n--- Fin de la simulación ---")