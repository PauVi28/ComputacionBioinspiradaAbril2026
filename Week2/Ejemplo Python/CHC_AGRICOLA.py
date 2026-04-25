import numpy as np
import random
from collections import namedtuple

# --- 1. Definición del Problema y Función de Fitness ---

# Límites para los factores (Luz, Agua, Nutrientes, Temperatura, pH)
FACTOR_LOWER_BOUNDS = np.array([200.0, 1.0, 50.0, 18.0, 5.0])
FACTOR_UPPER_BOUNDS = np.array([1000.0, 10.0, 200.0, 35.0, 8.0])
NUM_FACTORS = len(FACTOR_LOWER_BOUNDS)

# Valores óptimos para la función de fitness (crecimiento)
IDEAL_FACTORS = np.array([700.0, 5.0, 120.0, 28.0, 6.5])

# Sensibilidad (menor sigma = más sensible a la desviación)
# Ajustar estos valores para ver cómo afecta el crecimiento
FACTOR_SENSITIVITY = np.array([
    100.0,  # Luz
    1.0,    # Agua
    20.0,   # Nutrientes
    3.0,    # Temperatura
    0.5     # pH
])

def calculate_growth_fitness(factors):
    """
    Simula el crecimiento del producto agrícola.
    Basado en la proximidad a los factores ideales.
    Cuanto más cerca, mayor el valor.
    """
    if not (np.all(factors >= FACTOR_LOWER_BOUNDS) and np.all(factors <= FACTOR_UPPER_BOUNDS)):
        return -1.0e9 # Penalización si los factores están fuera de rango

    # Calcular la penalización por desviación de los ideales
    deviations = (factors - IDEAL_FACTORS) / FACTOR_SENSITIVITY
    
    # Usar una función gaussiana inversa para el crecimiento
    # exp(-x^2) donde x es la desviación normalizada
    growth_score = np.prod(np.exp(-(deviations**2)))
    
    return growth_score

# Definición del individuo
Individual = namedtuple('Individual', ['genes', 'fitness'])

# --- 2. Parámetros del Algoritmo CHC ---

POPULATION_SIZE = 500
MAX_GENERATIONS = 50
MAX_STAGNATION_GENERATIONS = 50 # Criterio de parada por estancamiento

# Parámetros específicos de CHC
D_INITIAL = NUM_FACTORS # Umbral inicial de divergencia (número de genes diferentes)
D_DECREASE_STEP = 1     # Cuánto decrece D cuando hay estancamiento
D_DECREASE_GENERATIONS = 10 # Cada cuántas generaciones sin mejora D se reduce
EPSILON_DIVERGENCE = 0.5 # Minima diferencia para considerar un gen "divergente"

# --- 3. Funciones Auxiliares del CHC ---

def initialize_population(pop_size, num_factors, lower_bounds, upper_bounds):
    """Crea una población inicial de individuos aleatorios."""
    population = []
    for _ in range(pop_size):
        genes = np.random.uniform(lower_bounds, upper_bounds, num_factors)
        
        # Opcional: redondear para factores enteros, aunque aquí usamos float
        # genes[0] = round(genes[0]) # Luz
        # genes[2] = round(genes[2]) # Nutrientes
        
        fitness = calculate_growth_fitness(genes)
        population.append(Individual(genes=genes, fitness=fitness))
    return population

def count_significantly_different_genes(ind1, ind2, epsilon):
    """
    Calcula cuántos genes (factores) difieren significativamente entre dos individuos.
    Usado para el umbral de divergencia D.
    """
    diff_count = np.sum(np.abs(ind1.genes - ind2.genes) > epsilon)
    return diff_count

def chc_crossover(parent1, parent2, num_factors, lower_bounds, upper_bounds):
    """
    Implementa una forma de cruce "Half Uniform Crossover" (HUX) adaptado para CHC
    en datos continuos, asegurando que los hijos sean "aproximadamente mitad diferentes".
    """
    child1_genes = np.copy(parent1.genes)
    child2_genes = np.copy(parent2.genes)

    # Determinar qué genes intercambiar para ser "half different"
    # Elige la mitad de los genes aleatoriamente para intercambiar valores.
    # Esto es una simplificación del HUX para valores continuos.
    # Un HUX canónico se aplica a bits. Aquí lo simulamos con genes flotantes.
    
    num_exchanges = num_factors // 2 # Intercambiar aproximadamente la mitad de los genes
    genes_to_exchange_indices = random.sample(range(num_factors), num_exchanges)

    for i in genes_to_exchange_indices:
        child1_genes[i], child2_genes[i] = child2_genes[i], child1_genes[i]
    
    # Asegurar que los genes estén dentro de los límites después del cruce
    child1_genes = np.clip(child1_genes, lower_bounds, upper_bounds)
    child2_genes = np.clip(child2_genes, lower_bounds, upper_bounds)

    return Individual(genes=child1_genes, fitness=calculate_growth_fitness(child1_genes)), \
           Individual(genes=child2_genes, fitness=calculate_growth_fitness(child2_genes))

def chc_hypermutation_restart(population, best_global_individual, lower_bounds, upper_bounds, pop_size):
    """
    Realiza una hiper-mutación (reinicio) de la población manteniendo el mejor individuo.
    """
    print("\n--- ¡Hiper-mutación (Reinicio de Población) activada! ---")
    new_population = [best_global_individual] # Conserva el mejor

    # Genera el resto de la población de forma aleatoria (alta diversidad)
    for _ in range(pop_size - 1):
        genes = np.random.uniform(lower_bounds, upper_bounds, NUM_FACTORS)
        
        # Opcional: redondear
        # genes[0] = round(genes[0])
        # genes[2] = round(genes[2])
        
        fitness = calculate_growth_fitness(genes)
        new_population.append(Individual(genes=genes, fitness=fitness))
        
    return new_population

# --- 4. Algoritmo Principal CHC ---

def chc_genetic_algorithm():
    population = initialize_population(POPULATION_SIZE, NUM_FACTORS, FACTOR_LOWER_BOUNDS, FACTOR_UPPER_BOUNDS)

    best_global_individual = max(population, key=lambda ind: ind.fitness)
    
    current_d_threshold = D_INITIAL
    stagnation_counter = 0 # Cuenta generaciones sin mejora del mejor global
    d_decrease_stagnation_counter = 0 # Cuenta generaciones sin mejora para reducir D

    print("--- Iniciando Algoritmo Genético CHC ---")
    print(f"Mejor inicial: {best_global_individual.fitness:.4f} con factores {best_global_individual.genes}")

    for generation in range(MAX_GENERATIONS):
        # 1. Emparejamiento y Cruce
        offspring = []
        random.shuffle(population) # Mezclar para emparejamientos aleatorios

        for i in range(0, POPULATION_SIZE, 2):
            if i + 1 < POPULATION_SIZE:
                parent1 = population[i]
                parent2 = population[i+1]

                # CHEQUEO DE DIVERGENCIA (CORAZÓN DEL CHC)
                diversity = count_significantly_different_genes(parent1, parent2, EPSILON_DIVERGENCE)
                
                if diversity >= current_d_threshold:
                    child1, child2 = chc_crossover(parent1, parent2, NUM_FACTORS, FACTOR_LOWER_BOUNDS, FACTOR_UPPER_BOUNDS)
                    offspring.extend([child1, child2])
                # else: Si no cumplen el umbral D, no se cruzan (prevención de incesto).

        # 2. Selección de Sobrevivientes (Elitismo)
        # Combinar padres y la descendencia generada
        combined_population = sorted(population + offspring, key=lambda ind: ind.fitness, reverse=True)
        
        # Seleccionar los mejores POPULATION_SIZE individuos
        new_population = combined_population[:POPULATION_SIZE]
        population = new_population

        current_best_gen_individual = population[0] # El mejor de la generación actual

        # --- Manejo de la Convergencia y el Umbral D ---
        
        improved_this_generation = False
        if current_best_gen_individual.fitness > best_global_individual.fitness:
            best_global_individual = current_best_gen_individual
            improved_this_generation = True
            stagnation_counter = 0 # Reiniciar contador de estancamiento global
            d_decrease_stagnation_counter = 0 # Reiniciar contador para reducción de D
        else:
            stagnation_counter += 1
            d_decrease_stagnation_counter += 1

        # Reducir D si hay estancamiento en la mejora para D-threshold
        if d_decrease_stagnation_counter >= D_DECREASE_GENERATIONS:
            current_d_threshold -= D_DECREASE_STEP
            current_d_threshold = max(0, current_d_threshold) # Asegura que D no sea negativo
            d_decrease_stagnation_counter = 0 # Resetear contador después de reducir D
            print(f"  --> Generación {generation+1}: Mejor global no mejora, D reducido a {current_d_threshold}")


        # HIIPER-MUTACIÓN (REINICIO) si D llega a 0
        if current_d_threshold <= 0:
            population = chc_hypermutation_restart(population, best_global_individual, FACTOR_LOWER_BOUNDS, FACTOR_UPPER_BOUNDS, POPULATION_SIZE)
            current_d_threshold = D_INITIAL # Reiniciar D
            stagnation_counter = 0 # Reiniciar contador de estancamiento global
            d_decrease_stagnation_counter = 0 # Reiniciar contador para reducción de D

        # --- Impresión de Progreso ---
        print(f"Generación {generation+1}: Mejor Fitness = {current_best_gen_individual.fitness:.4f}, Mejor Global = {best_global_individual.fitness:.4f}, D={int(current_d_threshold)}")
        
        # --- Criterios de Parada ---
        if stagnation_counter >= MAX_STAGNATION_GENERATIONS:
            print(f"\n--- Criterio de Parada: Estancamiento del Mejor Fitness Global tras {MAX_STAGNATION_GENERATIONS} generaciones ---")
            break
        
        if generation == MAX_GENERATIONS - 1:
            print(f"\n--- Criterio de Parada: Alcanzado el máximo de {MAX_GENERATIONS} generaciones ---")

    print("\n--- Optimización Finalizada ---")
    print(f"Mejor combinación de factores encontrada: {best_global_individual.genes}")
    print(f"Crecimiento estimado (Fitness): {best_global_individual.fitness:.4f}")
    
    # Imprimir los factores con formato
    factor_names = ["Luz (Lux)", "Agua (L/día)", "Nutrientes (ppm)", "Temperatura (°C)", "pH del Suelo"]
    print("\nDetalles de la mejor combinación:")
    for i, factor_name in enumerate(factor_names):
        print(f"- {factor_name}: {best_global_individual.genes[i]:.2f}")

    print("\nValores óptimos (teóricos) para comparación:")
    for i, factor_name in enumerate(factor_names):
        print(f"- {factor_name}: {IDEAL_FACTORS[i]:.2f}")

# --- Ejecución del Algoritmo ---
if __name__ == "__main__":
    chc_genetic_algorithm()