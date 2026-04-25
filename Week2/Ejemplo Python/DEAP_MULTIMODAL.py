import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms
import random

# --- 1. Definición del Problema Multimodal ---

# Puntos centrales de nuestros picos (x, y)
PEAK_CENTERS = [
    (2.0, 2.0),
    (-2.0, -2.0),
    (2.0, -2.0),
    (-2.0, 2.0),
    (0.0, 0.0) # Un pico extra en el centro, quizás un poco más bajo para variar
]

# Alturas de los picos (correspondientes a PEAK_CENTERS)
PEAK_AMPLITUDES = [100, 100, 100, 100, 80]

# Desviación estándar de los picos (controla la anchura)
PEAK_SIGMAS = [0.5, 0.8, 0.5, 0.8, 0.7] # Todos los picos son bastante estrechos


# Función objetivo multimodal
def multimodal_fitness(individual):
    x, y = individual[0], individual[1]
    value = 0.0
    for i in range(len(PEAK_CENTERS)):
        cx, cy = PEAK_CENTERS[i]
        amplitude = PEAK_AMPLITUDES[i]
        sigma = PEAK_SIGMAS[i]
        # Fórmula de la campana Gaussiana 2D
        value += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return value, # La coma es importante para las tuplas de fitness

# Límites del espacio de búsqueda
BOUND_LOW, BOUND_UP = -5.0, 5.0

# --- 2. Configuración de DEAP ---

# 1. Definir los tipos de fitness e individuos
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximizar la función
creator.create("Individual", list, fitness=creator.FitnessMax)

# 2. Inicialización de la caja de herramientas
toolbox = base.Toolbox()

# Atributo genético: un número flotante aleatorio dentro de los límites
toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)

# Estructura del individuo: una lista de 2 flotantes (x, y)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)

# Estructura de la población
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores genéticos
toolbox.register("evaluate", multimodal_fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5) # Cruce BLX-alpha (bueno para números reales)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.1) # Mutación Gaussiana

# La selección se hará de forma personalizada en el algoritmo de crowding

# --- 3. Implementación del Algoritmo de Deterministic Crowding ---

def eaDeterministicCrowding(population, toolbox, cxpb, mutpb, ngen, stats=None,
                            halloffame=None, verbose=__debug__):
    """Algoritmo de Deterministic Crowding."""

    logbook = tools.Logbook()
    # CORRECCIÓN AQUÍ: Usar stats.fields en lugar de stats.header
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) 

    # Evaluar la población inicial
    fits = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fits):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # Bucle principal de generaciones
    for gen in range(1, ngen + 1):
        # Clonar la población para la siguiente generación
        # Esto es crucial para el crowding, ya que los reemplazos se hacen in-place pero después de procesar los padres
        offspring = []
        
        # Mezclar la población para emparejamientos aleatorios
        random.shuffle(population) 

        # Iterar en pares para generar y reemplazar
        for i in range(0, len(population), 2):
            parent1 = toolbox.clone(population[i])
            parent2 = toolbox.clone(population[i+1])

            # Cruce
            child1, child2 = toolbox.mate(parent1, parent2)
            # Asegurarse de que los individuos estén dentro de los límites después del cruce/mutación
            child1[:] = [np.clip(val, BOUND_LOW, BOUND_UP) for val in child1]
            child2[:] = [np.clip(val, BOUND_LOW, BOUND_UP) for val in child2]

            del child1.fitness.values # Clear fitness so it's re-evaluated
            del child2.fitness.values

            # Mutación
            child1 = toolbox.mutate(child1)[0]
            child2 = toolbox.mutate(child2)[0]
            # Asegurarse de que los individuos estén dentro de los límites después del cruce/mutación
            child1[:] = [np.clip(val, BOUND_LOW, BOUND_UP) for val in child1]
            child2[:] = [np.clip(val, BOUND_LOW, BOUND_UP) for val in child2]
            
            # Evaluar a los hijos
            child1.fitness.values = toolbox.evaluate(child1)
            child2.fitness.values = toolbox.evaluate(child2)

            # Lógica de reemplazo de Deterministic Crowding
            # Calcular distancias genotípicas (euclidiana en este caso 2D)
            dist_p1_c1 = np.linalg.norm(np.array(parent1) - np.array(child1))
            dist_p2_c2 = np.linalg.norm(np.array(parent2) - np.array(child2))
            dist_p1_c2 = np.linalg.norm(np.array(parent1) - np.array(child2))
            dist_p2_c1 = np.linalg.norm(np.array(parent2) - np.array(child1))

            # Determinar qué emparejamiento minimiza la suma de distancias
            if (dist_p1_c1 + dist_p2_c2) <= (dist_p1_c2 + dist_p2_c1):
                # Caso 1: (p1, c1) y (p2, c2)
                if child1.fitness.values > parent1.fitness.values:
                    offspring.append(child1)
                else:
                    offspring.append(parent1)
                
                if child2.fitness.values > parent2.fitness.values:
                    offspring.append(child2)
                else:
                    offspring.append(parent2)
            else:
                # Caso 2: (p1, c2) y (p2, c1)
                if child2.fitness.values > parent1.fitness.values:
                    offspring.append(child2)
                else:
                    offspring.append(parent1)

                if child1.fitness.values > parent2.fitness.values:
                    offspring.append(child1)
                else:
                    offspring.append(parent2)
        
        population[:] = offspring # Reemplaza la población con la nueva generación

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

# --- 4. Parámetros de la Ejecución y Ejecución Principal ---

if __name__ == "__main__":
    random.seed(42) # Para reproducibilidad

    POPULATION_SIZE = 200 # Tamaño de la población, debe ser suficiente para cubrir todos los nichos
    GENERATIONS = 300     # Número de generaciones
    CXPB = 0.9            # Probabilidad de cruce
    MUTPB = 0.1           # Probabilidad de mutación

    pop = toolbox.population(n=POPULATION_SIZE)

    # Estadísticas para monitorear el progreso
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Iniciando algoritmo de Deterministic Crowding...")
    final_population, log = eaDeterministicCrowding(pop, toolbox, CXPB, MUTPB,
                                              GENERATIONS, stats, verbose=True)
    print("Algoritmo finalizado.")

    # --- 5. Análisis y Visualización de Resultados ---

    # Extraer las coordenadas de los individuos finales
    x_coords = [ind[0] for ind in final_population]
    y_coords = [ind[1] for ind in final_population]
    z_coords = [ind.fitness.values[0] for ind in final_population]

    # Crear la malla para la visualización de la función objetivo
    x = np.linspace(BOUND_LOW, BOUND_UP, 100)
    y = np.linspace(BOUND_LOW, BOUND_UP, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calcular Z para toda la malla
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = multimodal_fitness([X[i, j], Y[i, j]])[0]


    # Plot 3D de la función objetivo y los puntos encontrados
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie de la función objetivo
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, rstride=1, cstride=1)

    # Puntos de la población final
    ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o', s=50, label='Individuos de la Población Final', depthshade=False)

    # Marcar los centros de los picos reales
    peak_x = [p[0] for p in PEAK_CENTERS]
    peak_y = [p[1] for p in PEAK_CENTERS]
    peak_z = [multimodal_fitness([px, py])[0] for px, py in PEAK_CENTERS]
    ax.scatter(peak_x, peak_y, peak_z, c='blue', marker='X', s=200, label='Centros de Picos Reales', edgecolors='black', linewidth=1)

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Valor de Fitness (Z)')
    ax.set_title('Optimización Multimodal con Deterministic Crowding')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Opcional: Mostrar los puntos únicos "encontrados" agrupados
    # Una forma simple de agrupar cercanos (no es clustering formal)
    found_optima_display = []
    TOLERANCE = 0.5 # Distancia máxima para considerar puntos como el mismo óptimo

    # Para evitar añadir duplicados muy cercanos
    processed_indices = set()
    for i, ind1 in enumerate(final_population):
        if i in processed_indices:
            continue
        
        cluster = [ind1]
        for j, ind2 in enumerate(final_population):
            if i == j or j in processed_indices:
                continue
            if np.linalg.norm(np.array(ind1) - np.array(ind2)) < TOLERANCE:
                cluster.append(ind2)
                processed_indices.add(j)
        
        # Tomar el mejor individuo del cluster como representante
        best_in_cluster = max(cluster, key=lambda ind: ind.fitness.values[0])
        found_optima_display.append(best_in_cluster)
        processed_indices.add(i) # Añadir el original también

    # Ordenar los óptimos encontrados por fitness descendente para una mejor visualización
    found_optima_display.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
    
    print(f"\nNúmero de picos reales: {len(PEAK_CENTERS)}")
    print(f"Puntos distintos encontrados por el algoritmo (aproximado): {len(found_optima_display)}")
    print("\nCoordenadas de los puntos 'óptimos' encontrados (representantes de clusters):")
    for i, ind in enumerate(found_optima_display):
        print(f"  Óptimo {i+1}: X={ind[0]:.2f}, Y={ind[1]:.2f}, Fitness={ind.fitness.values[0]:.2f}")

    # Visualización 2D de la distribución de la población
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Valor de Fitness')
    plt.scatter(x_coords, y_coords, c='red', marker='o', s=20, alpha=0.6, label='Individuos de la Población Final')
    plt.scatter(peak_x, peak_y, c='blue', marker='X', s=150, label='Centros de Picos Reales', edgecolors='black', linewidth=1)
    
    # También mostrar los puntos agrupados
    grouped_x = [ind[0] for ind in found_optima_display]
    grouped_y = [ind[1] for ind in found_optima_display]
    plt.scatter(grouped_x, grouped_y, c='purple', marker='*', s=300, label='Óptimos Agrupados', edgecolors='white', linewidth=2, zorder=5)

    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Distribución de la Población Final en el Espacio de Búsqueda')
    plt.legend()
    plt.grid(True)
    plt.show()
