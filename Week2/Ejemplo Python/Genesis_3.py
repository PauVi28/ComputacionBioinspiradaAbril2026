# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 17:08:49 2025

@author: MARCELOFGB
"""

import random
import string # Para acceder a las letras mayúsculas

# --- Configuración del Ejemplo 3 ---
CHROMOSOME_LENGTH_EX3 = 10
POPULATION_SIZE_EX3 = 500
ALPHABET_EX3 = string.ascii_uppercase # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

print("="*60)
print("  EJEMPLO 3: Cromosoma de 10 Caracteres (A-Z), 100 Individuos")
print("="*60)

# --- 1. Proceso de Génesis de un Individuo ---
print("\n--- Génesis Detallada de UN Individuo (para entender el proceso) ---")
print(f"Cromosoma tiene {CHROMOSOME_LENGTH_EX3} caracteres. Cada gen es una letra aleatoria de {len(ALPHABET_EX3)}.")

def generate_individual_ex3():
    """Genera un cromosoma de 10 caracteres aleatorios (A-Z)."""
    chromosome = []
    print("\nGenerando un nuevo individuo:")
    for i in range(CHROMOSOME_LENGTH_EX3):
        gene_value = random.choice(ALPHABET_EX3) # Elegir una letra aleatoria
        chromosome.append(gene_value)
        print(f"  Gen {i+1:2d}: Se elige aleatoriamente '{gene_value}'")
    return "".join(chromosome) # Convertir la lista de caracteres a una cadena

# Generamos y mostramos el proceso de un individuo
sample_individual_ex3 = generate_individual_ex3()
print(f"\nIndividuo generado: [{sample_individual_ex3}]")
print("-" * 50)

# --- 2. Generación de la Población Inicial ---
print(f"\n--- Generación de la Población Inicial (Total: {POPULATION_SIZE_EX3} Individuos) ---")

def generate_initial_population_ex3(size):
    """Genera una población de individuos de 10 caracteres."""
    population = []
    for i in range(size):
        # Para la población, no mostramos el detalle de cada caracter
        chromosome = "".join([random.choice(ALPHABET_EX3) for _ in range(CHROMOSOME_LENGTH_EX3)])
        population.append(chromosome)
    return population

initial_population_ex3 = generate_initial_population_ex3(POPULATION_SIZE_EX3)

# --- 3. Visualización de la Población Generada ---
print("\n--- Vista de la Población Generada (Muestra) ---")
print(f"Población total generada: {len(initial_population_ex3)} individuos.")

# Mostrar los primeros 5 y los últimos 5 individuos para una muestra
print("\nPrimeros 5 individuos:")
for i in range(5):
    print(f"  Individuo {i+1:3d}: [{initial_population_ex3[i]}]")

print("\n...") # Indicador de que hay más individuos

print(f"\nÚltimos 5 individuos (de {POPULATION_SIZE_EX3}):")
for i in range(POPULATION_SIZE_EX3 - 5, POPULATION_SIZE_EX3):
    print(f"  Individuo {i+1:3d}: [{initial_population_ex3[i]}]")

print("\nLa población muestra una gran diversidad de secuencias de caracteres generadas aleatoriamente.")
print("="*60 + "\n\n")