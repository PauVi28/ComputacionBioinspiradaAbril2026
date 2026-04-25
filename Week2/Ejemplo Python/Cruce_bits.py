# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 17:20:27 2025

@author: MARCELOFGB
"""

import random

# --- Configuración del Ejemplo 1 ---
CHROMOSOME_LENGTH_EX1 = 16
POPULATION_SIZE_EX1 = 100

print("="*80)
print("  EJEMPLO 1 (Módificado): Cromosoma de 16 Bits y Métodos de Cruzamiento")
print("="*80)

# --- Funciones de Generación de Población (del ejemplo anterior) ---
def generate_individual_ex1():
    """Genera un cromosoma de 16 bits aleatorios."""
    chromosome = [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH_EX1)]
    return "".join(map(str, chromosome))

def generate_initial_population_ex1(size):
    """Genera una población de individuos de 16 bits."""
    population = []
    for _ in range(size):
        population.append(generate_individual_ex1())
    return population

# Generar una población inicial para tener padres de ejemplo
initial_population_ex1 = generate_initial_population_ex1(POPULATION_SIZE_EX1)

print("\n--- Vista de la Población Generada (Muestra) ---")
print(f"Población total generada: {len(initial_population_ex1)} individuos.")
print("Individuos:")
for i in range(100):
    print(f"  Individuo {i+1:3d}: [{initial_population_ex1[i]}]")
print("-" * 80)


# =========================================================================
# === MÉTODOS DE CRUZAMIENTO Y SU VISUALIZACIÓN ===========================
# =========================================================================

def display_crossover_header(title):
    print("\n" + "="*80)
    print(f"  >>> {title} <<<")
    print("="*80)

def visualize_chromosomes(parent1, parent2, point1=None, point2=None, mask=None, child1=None, child2=None):
    """
    Función auxiliar para visualizar los cromosomas y puntos de cruce.
    Adapta la visualización según el tipo de cruce.
    """
    length = len(parent1)
    
    print("\n  Padre 1: " + " ".join(parent1))
    print("  Padre 2: " + " ".join(parent2))

    points_str = " " * 11 # Alineación inicial

    # Dibujar los puntos de cruce
    if point1 is not None and point2 is None: # Un punto
        points_str += " " * (2 * point1) + "^" + " " * (2 * (length - point1 -1)) + " (Punto de Cruce)"
        
    elif point1 is not None and point2 is not None: # Dos puntos
        if point1 < point2: # Asegurar orden
            p1, p2 = point1, point2
        else:
            p1, p2 = point2, point1

        points_str += " " * (2 * p1) + "^" + " " * (2 * (p2 - p1 -1)) + " ^" + " " * (2 * (length - p2 -1)) + " (Puntos de Cruce)"
    
    elif mask is not None: # Uniforme
        # Para el cruce uniforme, la "línea" de puntos es la máscara
        points_str += " " * (1) + " ".join(mask) + " (Máscara de Cruce)"
    
    print(points_str)

    # Dibujar la línea de cruce
    if point1 is not None and point2 is None:
        line_p1 = " ".join(parent1[:point1]) + " | " + " ".join(parent1[point1:])
        line_p2 = " ".join(parent2[:point1]) + " | " + " ".join(parent2[point1:])
        print("  " + line_p1)
        print("  " + line_p2)
    elif point1 is not None and point2 is not None:
        if point1 < point2:
            p1, p2 = point1, point2
        else:
            p1, p2 = point2, point1
        
        line_p1 = " ".join(parent1[:p1]) + " | " + " ".join(parent1[p1:p2]) + " | " + " ".join(parent1[p2:])
        line_p2 = " ".join(parent2[:p1]) + " | " + " ".join(parent2[p1:p2]) + " | " + " ".join(parent2[p2:])
        print("  " + line_p1)
        print("  " + line_p2)
    else: # Uniforme o sin puntos de cruce explícitos en esta etapa
        print("  " + " ".join(parent1))
        print("  " + " ".join(parent2))


    if child1 and child2:
        print("\n  Hijo 1:  " + " ".join(child1))
        print("  Hijo 2:  " + " ".join(child2))
        print("--------------------------------------------------")


# --- 1. Cruzamiento de un Punto (One-Point Crossover) ---
def one_point_crossover(parent1, parent2):
    """
    Realiza el cruzamiento de un punto entre dos cromosomas.
    Establece un punto de intercambio aleatorio.
    """
    length = len(parent1)
    # El punto de cruce no puede ser ni el principio ni el final
    crossover_point = random.randint(1, length - 1) 

    child1_part1 = parent1[:crossover_point]
    child1_part2 = parent2[crossover_point:]
    child1 = child1_part1 + child1_part2

    child2_part1 = parent2[:crossover_point]
    child2_part2 = parent1[crossover_point:]
    child2 = child2_part1 + child2_part2
    
    display_crossover_header("Cruzamiento de UN Punto")
    print(f"Punto de cruce elegido: {crossover_point}")
    visualize_chromosomes(parent1, parent2, point1=crossover_point, child1=child1, child2=child2)
    
    return child1, child2

# --- 2. Cruzamiento de Dos Puntos (Two-Point Crossover) ---
def two_point_crossover(parent1, parent2):
    """
    Realiza el cruzamiento de dos puntos entre dos cromosomas.
    Establece dos puntos de intercambio aleatorios.
    """
    length = len(parent1)
    
    # Elegir dos puntos de cruce distintos y ordenarlos
    points = sorted(random.sample(range(1, length), 2))
    crossover_point1 = points[0]
    crossover_point2 = points[1]

    # Partes para Hijo 1
    child1_part1 = parent1[:crossover_point1]
    child1_part2 = parent2[crossover_point1:crossover_point2]
    child1_part3 = parent1[crossover_point2:]
    child1 = child1_part1 + child1_part2 + child1_part3

    # Partes para Hijo 2 (intercambiado con Hijo 1)
    child2_part1 = parent2[:crossover_point1]
    child2_part2 = parent1[crossover_point1:crossover_point2]
    child2_part3 = parent2[crossover_point2:]
    child2 = child2_part1 + child2_part2 + child2_part3

    display_crossover_header("Cruzamiento de DOS Puntos")
    print(f"Puntos de cruce elegidos: {crossover_point1} y {crossover_point2}")
    visualize_chromosomes(parent1, parent2, point1=crossover_point1, point2=crossover_point2, child1=child1, child2=child2)
    
    return child1, child2

# --- 3. Cruzamiento Uniforme (Uniform Crossover) ---
def uniform_crossover(parent1, parent2):
    """
    Realiza el cruzamiento uniforme. Cada gen se elige aleatoriamente
    de uno de los padres con un 50% de probabilidad.
    """
    length = len(parent1)
    child1_list = []
    child2_list = []
    mask = [] # Para visualizar qué padre contribuye
    
    for i in range(length):
        if random.random() < 0.5: # 50% de probabilidad
            child1_list.append(parent1[i])
            child2_list.append(parent2[i])
            mask.append('1') # Representa que Parent1 contribuyó a Child1 (y P2 a C2)
        else:
            child1_list.append(parent2[i])
            child2_list.append(parent1[i])
            mask.append('2') # Representa que Parent2 contribuyó a Child1 (y P1 a C2)
    
    child1 = "".join(child1_list)
    child2 = "".join(child2_list)

    display_crossover_header("Cruzamiento UNIFORME")
    # visualizamos la máscara para entender el proceso gen a gen
    visualize_chromosomes(parent1, parent2, mask=mask, child1=child1, child2=child2)

    return child1, child2

# =========================================================================
# === DEMOSTRACIÓN DE LOS MÉTODOS DE CRUZAMIENTO =========================
# =========================================================================

print("\n--- DEMOSTRACIÓN DE LOS OPERADORES DE CRUZAMIENTO ---")

# Elegir dos padres aleatorios de la población inicial para la demostración
parent_A = initial_population_ex1[random.randint(0, POPULATION_SIZE_EX1 - 1)]
parent_B = initial_population_ex1[random.randint(0, POPULATION_SIZE_EX1 - 1)]

# Asegurarse de que los padres sean diferentes para una mejor demostración
while parent_A == parent_B:
    parent_B = initial_population_ex1[random.randint(0, POPULATION_SIZE_EX1 - 1)]

print(f"\nPadre elegido A: [{parent_A}]")
print(f"Padre elegido B: [{parent_B}]")
print("\n")

# Demostración Cruzamiento de Un Punto
c1_p1, c2_p1 = one_point_crossover(parent_A, parent_B)

# Demostración Cruzamiento de Dos Puntos
c1_p2, c2_p2 = two_point_crossover(parent_A, parent_B)

# Demostración Cruzamiento Uniforme
# Para una mejor visualización del cruce uniforme, se usan padres con patrones claros
parent_uniform_A = "0000000000000000"
parent_uniform_B = "1111111111111111"
print("\nUtilizando padres didácticos para Cruce Uniforme:")
print(f"  Padre A (Uniforme): [{parent_uniform_A}]")
print(f"  Padre B (Uniforme): [{parent_uniform_B}]")
c1_uniform, c2_uniform = uniform_crossover(parent_uniform_A, parent_uniform_B)

print("\nDemostración de cruzamientos finalizada.")
print("="*80 + "\n\n")
