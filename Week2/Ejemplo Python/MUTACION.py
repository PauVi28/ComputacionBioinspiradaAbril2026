# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 17:31:51 2025

@author: MARCELOFGB
"""

import random
import string # Para acceder a las letras mayúsculas

# --- Configuración del Ejemplo 3 ---
CHROMOSOME_LENGTH_EX3 = 10
POPULATION_SIZE_EX3 = 100
ALPHABET_EX3 = string.ascii_uppercase # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# --- Configuración de la Mutación ---
MUTATION_RATE_PER_GENE = 0.1 # 10% de probabilidad de que cada gen mute

print("="*60)
print("  EJEMPLO 3 (Modificado): Cromosoma de 10 Caracteres y MUTACIÓN")
print("="*60)

# --- Funciones de Generación de Población (del ejemplo anterior) ---
def generate_individual_ex3():
    """Genera un cromosoma de 10 caracteres aleatorios (A-Z)."""
    chromosome = [random.choice(ALPHABET_EX3) for _ in range(CHROMOSOME_LENGTH_EX3)]
    return "".join(chromosome)

def generate_initial_population_ex3(size):
    """Genera una población de individuos de 10 caracteres."""
    population = []
    for _ in range(size):
        population.append(generate_individual_ex3())
    return population

# Generar una población inicial para tener individuos de ejemplo
initial_population_ex3 = generate_initial_population_ex3(POPULATION_SIZE_EX3)

print("\n--- Vista de la Población Generada (Muestra) ---")
print(f"Población total generada: {len(initial_population_ex3)} individuos.")
print("Primeros 3 individuos:")
for i in range(3):
    print(f"  Individuo {i+1:3d}: [{initial_population_ex3[i]}]")
print("-" * 50)


# =========================================================================
# === MÉTODO DE MUTACIÓN Y SU VISUALIZACIÓN ===============================
# =========================================================================

def print_mutation_header(title):
    print("\n" + "="*50)
    print(f"  >>> {title} <<<")
    print("="*50)

def visualize_mutation(original_chromosome, mutated_chromosome, mutation_indices):
    """
    Función para visualizar el proceso de mutación.
    Muestra el cromosoma original, los puntos de mutación y el cromosoma mutado.
    """
    length = len(original_chromosome)
    
    print("\n  Cromosoma Original:  " + " ".join(original_chromosome))
    
    # Crear un indicador para las posiciones mutadas
    indicator_line = [" "] * (length * 2 - 1) # Espacios suficientes para los caracteres y los espacios entre ellos
    for idx in mutation_indices:
        indicator_line[idx * 2] = "^" # Marcar la posición del gen
    
    if mutation_indices:
        print("                     " + "".join(indicator_line) + " <-- Posición(es) de mutación")
    else:
        print("                     (No se produjeron mutaciones en esta ejecución)")

    print("  Cromosoma Mutado:    " + " ".join(mutated_chromosome))
    print("--------------------------------------------------")


def mutate_chromosome(chromosome_str, mutation_rate, alphabet):
    """
    Aplica la mutación a un cromosoma dado.
    Para cada gen, decide si muta basándose en la tasa de mutación.
    Si muta, el gen cambia a un carácter aleatorio diferente del original.
    """
    chromosome_list = list(chromosome_str) # Convertir a lista para poder modificar los caracteres
    mutated_indices = []

    for i in range(len(chromosome_list)):
        if random.random() < mutation_rate:
            original_gene = chromosome_list[i]
            new_gene = random.choice(alphabet)
            # Asegurarse de que el nuevo gen sea diferente al original para que sea visualmente claro
            while new_gene == original_gene:
                new_gene = random.choice(alphabet)
            
            chromosome_list[i] = new_gene
            mutated_indices.append(i) # Guardar el índice del gen mutado

    return "".join(chromosome_list), mutated_indices

# =========================================================================
# === DEMOSTRACIÓN DE LA MUTACIÓN =========================================
# =========================================================================

print_mutation_header("DEMOSTRACIÓN DEL PROCESO DE MUTACIÓN")
print(f"Tasa de mutación por gen: {MUTATION_RATE_PER_GENE * 100}%")

num_demos = 5 # Realizaremos 5 demostraciones para ver diferentes resultados

for i in range(num_demos):
    print(f"\n--- Demostración de Mutación {i+1} ---")
    
    # Elegir un individuo aleatorio de la población inicial para mutar
    original_ind = initial_population_ex3[random.randint(0, POPULATION_SIZE_EX3 - 1)]
    
    print(f"Individuo seleccionado para mutación: [{original_ind}]")
    
    mutated_ind, indices = mutate_chromosome(original_ind, MUTATION_RATE_PER_GENE, ALPHABET_EX3)
    
    visualize_mutation(original_ind, mutated_ind, indices)

    if not indices:
        print("    -> No se produjeron mutaciones en esta ejecución debido a la aleatoriedad.")
        print("       (La tasa de mutación es una probabilidad por gen)")

print("\nDemostración de mutación finalizada.")
print("="*60 + "\n\n")