# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 01:40:46 2025

@author: MARCELOFGB
"""
import random
import string
import statistics # Asegúrate de que statistics esté importado

from deap import base, creator, tools#, algorithms

# --- 1. Configuración del Algoritmo Genético ---

# Parámetros de la contraseña
PASSWORD_LENGTH = 8 # Longitud de la contraseña a generar
ALL_CHARS = string.ascii_letters + string.digits + string.punctuation
# Excluimos algunos caracteres que podrían causar problemas o ser confusos
ALL_CHARS = ''.join(c for c in ALL_CHARS if c not in '<>{}[]()/?\\|\'"`') # Filtra caracteres problemáticos

# Parámetros del GA
POPULATION_SIZE = 100
NGEN = 50         # Número máximo de generaciones
CXPB = 0.7         # Probabilidad de cruce
MUTPB = 0.52        # Probabilidad de mutación

# Criterio de parada por falta de mejora
MAX_NO_IMPROVEMENT_GENS = 50 # Número de generaciones sin mejora para detenerse

# --- 2. Definición del Problema (DEAP Creator) ---

# Queremos maximizar la aptitud, por eso 'weights=(1.0,)'
# Chequeo para evitar RuntimeWarning al ejecutar múltiples veces
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 3. Configuración de la Caja de Herramientas (Toolbox) ---

toolbox = base.Toolbox()

# Atributo: elegir un carácter aleatorio de ALL_CHARS
toolbox.register("attr_char", random.choice, ALL_CHARS)

# Individuo: una lista de caracteres de longitud PASSWORD_LENGTH
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_char, PASSWORD_LENGTH)

# Población: una lista de individuos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación (Fitness Function)
def evaluatePassword(individual):
    """
    Evalúa la aptitud de la contraseña generada.
    Recompensa por la variedad de tipos de caracteres y la unicidad.
    """
    password_str = "".join(individual)

    has_upper = any(c.isupper() for c in password_str)
    has_lower = any(c.islower() for c in password_str)
    has_digit = any(c.isdigit() for c in password_str)
    # CORRECCIÓN: Usar string.punctuation para verificar símbolos de forma más fiable
    has_symbol = any(c in string.punctuation for c in password_str)

    # Calcular la diversidad de caracteres
    unique_chars_count = len(set(individual))

    score = 0
    # Puntos por tener al menos un tipo de caracter
    if has_upper:
        score += 10
    if has_lower:
        score += 10
    if has_digit:
        score += 10
    if has_symbol:
        score += 10

    # Puntos adicionales por la diversidad de caracteres (escalado)
    # Una mayor diversidad significa menos repeticiones obvias
    score += unique_chars_count * 0.5 # Cada caracter único suma 0.5 puntos

    return (score,) # DEAP espera una tupla, incluso para un solo valor de aptitud

toolbox.register("evaluate", evaluatePassword)

# Operadores genéticos
toolbox.register("select", tools.selTournament, tournsize=3) # Selección por torneo
toolbox.register("mate", tools.cxTwoPoint) # Cruce de dos puntos

# Mutación: un carácter en el individuo puede cambiar a un carácter aleatorio
def mutatePassword(individual, char_set, indpb):
    """
    Muta un individuo cambiando aleatoriamente sus caracteres.
    indpb: probabilidad de que un carácter individual mute.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(char_set)
    return individual, # DEAP espera una tupla de individuos mutados

toolbox.register("mutate", mutatePassword, char_set=ALL_CHARS, indpb=0.1) # Probabilidad de mutación individual por carácter

# --- 4. Ejecución del Algoritmo Genético ---

def main():
    #random.seed(42) # Semilla para reproducibilidad

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1) # Para guardar el mejor individuo encontrado
    
    # CORRECCIÓN: La lambda debe extraer el valor numérico (no la tupla)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", statistics.mean)
    stats.register("std", statistics.stdev)
    stats.register("min", min)
    stats.register("max", max)

    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max"

    print("Iniciando algoritmo genético para generar contraseña...")
    print(f"Longitud deseada: {PASSWORD_LENGTH}")
    print(f"Generaciones máximas: {NGEN}")
    print(f"Generaciones sin mejora para parar: {MAX_NO_IMPROVEMENT_GENS}\n")

    # --- Lógica del criterio de parada personalizado ---
    best_fitness_overall = -float('inf')
    generations_without_improvement = 0

    # Evaluar la población inicial
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Actualizar HallOfFame y logbook para la generación 0 después de la evaluación inicial
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)
    
    if hof: # Si hay un mejor individuo en el HallOfFame (debería haberlo)
        current_best_fitness = hof[0].fitness.values[0]
        best_fitness_overall = current_best_fitness # Inicializa con el mejor de la gen 0
        print(f"Mejor contraseña inicial: {''.join(hof[0])} (Fitness: {hof[0].fitness.values[0]:.2f})")


    # Bucle principal de generaciones
    for gen in range(1, NGEN + 1):
        # Seleccionar la próxima generación de individuos
        offspring = toolbox.select(pop, len(pop))
        # Clonar los individuos seleccionados (importante para no modificar la población actual)
        offspring = list(map(toolbox.clone, offspring))

        # Aplicar Cruce y Mutación en la descendencia
        # Recorre la descendencia de a pares para el cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # Eliminar la aptitud de los hijos si ha ocurrido cruce, para que DEAP los re-evalúe
                del child1.fitness.values
                del child2.fitness.values

        # Recorre cada individuo para la mutación
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant) # MutatePassword ya devuelve una tupla, DEAP lo maneja
                # Eliminar la aptitud si ha ocurrido mutación
                del mutant.fitness.values

        # Evaluar los individuos con fitness inválido (aquellos modificados por cruce o mutación)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Reemplazar la población actual con la descendencia
        pop[:] = offspring

        # Actualizar HallOfFame con la nueva población
        hof.update(pop)

        # Grabar estadísticas
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # --- Criterio de parada por falta de mejora ---
        if hof: # Asegurarse de que hof no está vacío (aunque para hof(1) nunca lo estará después de gen 0)
            current_best_fitness = hof[0].fitness.values[0]
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                generations_without_improvement = 0
                print(f"  --> Mejora detectada en Gen {gen}! Mejor actual: {''.join(hof[0])} (Fitness: {hof[0].fitness.values[0]:.2f})")
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= MAX_NO_IMPROVEMENT_GENS:
                print("\n--- CRITERIO DE PARADA ALCANZADO ---")
                print(f"No hubo mejora en la aptitud durante {MAX_NO_IMPROVEMENT_GENS} generaciones.")
                break
    else: # Se ejecuta si el bucle no fue interrumpido por un 'break'
        print("\n--- FINALIZACIÓN POR MÁXIMO DE GENERACIONES ---")
        print(f"Se alcanzaron {NGEN} generaciones.")


    # --- Resultados Finales ---
    print("\n--- Resultados ---")
    print(f"Mejor individuo encontrado: {''.join(hof[0])}")
    print(f"Aptitud del mejor individuo: {hof[0].fitness.values[0]:.2f}")
    print(f"Longitud de la contraseña: {len(hof[0])}")

    # Verificar las propiedades de la contraseña final
    final_password_str = "".join(hof[0])
    print("\nPropiedades de la contraseña sugerida:")
    print(f"- Tiene mayúsculas: {any(c.isupper() for c in final_password_str)}")
    print(f"- Tiene minúsculas: {any(c.islower() for c in final_password_str)}")
    print(f"- Tiene dígitos: {any(c.isdigit() for c in final_password_str)}")
    print(f"- Tiene símbolos: {any(c in string.punctuation for c in final_password_str)}")
    print(f"- Caracteres únicos: {len(set(final_password_str))}")
    print(f"Conjunto de caracteres usados: {set(final_password_str)}")

    return pop, stats, hof, logbook

if __name__ == "__main__":
    pop, stats, hof, logbook = main()