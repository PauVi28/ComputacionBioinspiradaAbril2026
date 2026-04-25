# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 16:33:48 2025

@author: MARCELOFGB
"""

import random

# --- Configuración del Ejemplo 1 ---
CHROMOSOME_LENGTH_EX1 = 10
POPULATION_SIZE_EX1 = 500

print("="*60)
print("  EJEMPLO 1: Cromosoma de 16 Bits, 100 Individuos")
print("="*60)

# --- 1. Proceso de Génesis de un Individuo ---
print("\n--- Génesis Detallada de UN Individuo (para entender el proceso) ---")
print(f"Cromosoma tiene {CHROMOSOME_LENGTH_EX1} bits. Cada bit se elige aleatoriamente (0 o 1).")

def generate_individual_ex1():
    """Genera un cromosoma de 16 bits aleatorios."""
    chromosome = []
    print("\nGenerando un nuevo individuo:")
    for i in range(CHROMOSOME_LENGTH_EX1):
        gene_value = random.randint(0, 1) # 0 o 1
        chromosome.append(gene_value)
        print(f"  Gen {i+1:2d}: Se elige aleatoriamente {gene_value}")
    return "".join(map(str, chromosome)) # Convertir la lista de enteros a una cadena de bits

# Generamos y mostramos el proceso de un individuo
sample_individual_ex1 = generate_individual_ex1()
print(f"\nIndividuo generado: [{sample_individual_ex1}]")
print("-" * 50)

# --- 2. Generación de la Población Inicial ---
print(f"\n--- Generación de la Población Inicial (Total: {POPULATION_SIZE_EX1} Individuos) ---")

def generate_initial_population_ex1(size):
    """Genera una población de individuos de 16 bits."""
    population = []
    for i in range(size):
        # Para la población, no mostramos el detalle de cada bit para que no sea muy largo
        chromosome = "".join(map(str, [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH_EX1)]))
        population.append(chromosome)
    return population

initial_population_ex1 = generate_initial_population_ex1(POPULATION_SIZE_EX1)

# --- 3. Visualización de la Población Generada ---
print("\n--- Vista de la Población Generada (Muestra) ---")
print(f"Población total generada: {len(initial_population_ex1)} individuos.")

# Mostrar los primeros 5 y los últimos 5 individuos para una muestra
print("\nPrimeros 5 individuos:")
for i in range(5):
    print(f"  Individuo {i+1:3d}: [{initial_population_ex1[i]}]")

print("\n...") # Indicador de que hay más individuos

print(f"\nÚltimos 5 individuos (de {POPULATION_SIZE_EX1}):")
for i in range(POPULATION_SIZE_EX1 - 5, POPULATION_SIZE_EX1):
    print(f"  Individuo {i+1:3d}: [{initial_population_ex1[i]}]")

print("\nLa población muestra una gran diversidad de patrones binarios generados aleatoriamente.")
print("="*60 + "\n\n")