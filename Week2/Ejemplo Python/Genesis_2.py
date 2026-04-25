# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 17:04:35 2025

@author: MARCELOFGB
"""

import random

# --- Configuración del Ejemplo 2 ---
NUM_TREES_PER_CHROMOSOME_EX2 = 4
POPULATION_SIZE_EX2 = 50
MAX_TREE_DEPTH = 3 # Profundidad máxima para controlar el tamaño de los árboles

# Conjuntos de funciones (nodos internos) y terminales (nodos hoja)
FUNCTION_SET = {'+', '-', '*', '/'} # Asumimos todas son binarias para simplificar
TERMINAL_SET = {'x', 'y', '2', '5', '0.5'}

print("="*60)
print("  EJEMPLO 2: Cromosoma de 4 Árboles (Programación Genética)")
print("="*60)

# --- Clase para representar Nodos del Árbol ---
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children is not None else []

    def __repr__(self):
        # Representación simplificada para mostrar la estructura en el cromosoma
        if not self.children:
            return str(self.value)
        return f"({self.value} {' '.join(str(child) for child in self.children)})"

    def to_string(self): # Para una representación más "matemática"
        if not self.children:
            return str(self.value)
        
        op = self.value
        left = self.children[0].to_string()
        right = self.children[1].to_string()
        
        # Añadir paréntesis para claridad en el orden de operaciones
        if op in ['*', '/']:
            if self.children[0].value in ['+', '-']:
                left = f"({left})"
            if self.children[1].value in ['+', '-']:
                right = f"({right})"
        return f"{left} {op} {right}"


def build_random_tree(current_depth):
    """
    Construye un árbol aleatorio recursivamente.
    Usa una mezcla de funciones y terminales dependiendo de la profundidad.
    """
    # Si la profundidad actual es máxima O el 70% de las veces, generar un terminal
    if current_depth >= MAX_TREE_DEPTH or random.random() < 0.7:
        return Node(random.choice(list(TERMINAL_SET)))
    else:
        # Generar una función
        func = random.choice(list(FUNCTION_SET))
        node = Node(func)
        # Para simplificar, asumimos todas las funciones son binarias (2 hijos)
        node.children.append(build_random_tree(current_depth + 1))
        node.children.append(build_random_tree(current_depth + 1))
        return node

# --- Función para Imprimir un Árbol en ASCII Art (didáctico) ---
def print_tree_ascii(node, indent="", is_last=True):
    """Imprime un árbol en formato ASCII art."""
    if node is None:
        return

    marker = "└── " if is_last else "├── "
    print(indent + marker + str(node.value))
    
    new_indent = indent + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        print_tree_ascii(child, new_indent, i == len(node.children) - 1)


# --- 1. Proceso de Génesis de un Individuo ---
print("\n--- Génesis Detallada de UN Individuo (para entender el proceso) ---")
print(f"Cromosoma tiene {NUM_TREES_PER_CHROMOSOME_EX2} árboles. Cada árbol es un 'gen'.")
print("Cada árbol se construye recursivamente con funciones y terminales aleatorios.")

def generate_individual_ex2():
    """Genera un cromosoma compuesto por 4 árboles aleatorios."""
    chromosome_trees = []
    print("\nGenerando un nuevo individuo (cromosoma con 4 árboles):")
    for i in range(NUM_TREES_PER_CHROMOSOME_EX2):
        print(f"\n  Gen {i+1} (Árbol {i+1}):")
        tree = build_random_tree(0) # Iniciar la construcción del árbol desde profundidad 0
        chromosome_trees.append(tree)
        print("  Estructura del árbol generado:")
        print_tree_ascii(tree)
        print(f"  Expresión (aproximada): {tree.to_string()}")
    return chromosome_trees

sample_individual_ex2 = generate_individual_ex2()
print("\nIndividual generado completo (representación simplificada de los árboles):")
print([tree.to_string() for tree in sample_individual_ex2])
print("-" * 50)

# --- 2. Generación de la Población Inicial ---
print(f"\n--- Generación de la Población Inicial (Total: {POPULATION_SIZE_EX2} Individuos) ---")

def generate_initial_population_ex2(size):
    """Genera una población de individuos, cada uno con 4 árboles."""
    population = []
    for i in range(size):
        chromosome = [build_random_tree(0) for _ in range(NUM_TREES_PER_CHROMOSOME_EX2)]
        population.append(chromosome)
    return population

initial_population_ex2 = generate_initial_population_ex2(POPULATION_SIZE_EX2)

# --- 3. Visualización de la Población Generada ---
print("\n--- Vista de la Población Generada (Muestra) ---")
print(f"Población total generada: {len(initial_population_ex2)} individuos.")

# Mostrar los primeros 3 y los últimos 2 individuos para una muestra
print("\nPrimeros 10 individuos:")
for i in range(10):
    print(f"  Individuo {i+1:2d}:")
    for j, tree in enumerate(initial_population_ex2[i]):
        print(f"    Gen {j+1} (Tree): {tree.to_string()}")
    print("-" * 20)

print("\n...") # Indicador de que hay más individuos

print(f"\nÚltimos 2 individuos (de {POPULATION_SIZE_EX2}):")
for i in range(POPULATION_SIZE_EX2 - 2, POPULATION_SIZE_EX2):
    print(f"  Individuo {i+1:2d}:")
    for j, tree in enumerate(initial_population_ex2[i]):
        print(f"    Gen {j+1} (Tree): {tree.to_string()}")
    print("-" * 20)

print("\nCada individuo es una colección de expresiones o 'mini-programas' diferentes, generados aleatoriamente.")
print("Cada árbol en sí mismo es una expresión diferente, lo que garantiza una gran diversidad inicial.")
print("="*60 + "\n\n")