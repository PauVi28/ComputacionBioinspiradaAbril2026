# -*- coding: utf-8 -*-
"""
Created on Fri May  9 00:55:08 2025

@author: MARCELOFGB
"""
import sympy
import random
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x # Importa 'x' como un símbolo de SymPy
from sympy.utilities.lambdify import lambdify
import copy

# --- Parámetros del Algoritmo Genético ---
POPULATION_SIZE = 100
MAX_GENERATIONS = 10
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
MAX_INITIAL_DEPTH = 3  # Profundidad máxima para los árboles de expresión iniciales
TOURNAMENT_SIZE = 5    # Tamaño del torneo para la selección
ELITISM_COUNT = 2      # Cuántos mejores individuos pasan directamente a la siguiente generación

# --- Definición del problema: Encontrar f(x) = x^2 + x + 1 ---
# Símbolo de SymPy que usaremos
X_SYM = x # Ya importado de sympy.abc

# Función objetivo como expresión SymPy
TARGET_EXPR_SYM = X_SYM**2 + X_SYM + 1

# Convertimos la expresión SymPy a una función Python para evaluación rápida
# Usamos 'numpy' para operaciones vectorizadas si es necesario
TARGET_FUNC_LAMBDA = lambdify(X_SYM, TARGET_EXPR_SYM, 'numpy')

# Puntos de datos para evaluar el fitness
# Usaremos un rango de valores de x para ver qué tan bien se ajusta la fórmula
X_POINTS = np.linspace(-5, 5, 20) # 20 puntos entre -5 y 5
Y_TARGET = TARGET_FUNC_LAMBDA(X_POINTS)

# --- Componentes de la Programación Genética ---

# 1. Representación de Individuos (Programas/Fórmulas)
# Usaremos directamente expresiones de SymPy.

# Nodos terminales (variables o constantes)
TERMINALS = [X_SYM] + [sympy.Integer(i) for i in range(-2, 3)] # x, -2, -1, 0, 1, 2

# Nodos de función (operadores)
# (función_sympy, aridad)

FUNCTIONS = [
    (sympy.Add, 2, "add"),
    (sympy.Mul, 2, "mul"),
    (lambda arg1, arg2: arg1 - arg2, 2, "sub"),  # <-- CORRECCIÓN AQUÍ
    # (sympy.Pow, 2) # Pow puede ser más complicado de manejar inicialmente por dominios
    # Para simplificar, podríamos tener una función unaria como 'square'
    (lambda val: val**2, 1, "sqr"), # Función unaria para el cuadrado
    (lambda val: val, 1, "id") # Identidad, útil para rellenar
]


# Función para generar una expresión SymPy aleatoria (árbol)
def generate_random_expression(max_depth, current_depth=0):
    """Genera una expresión SymPy aleatoria hasta una profundidad máxima."""
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.4): # Probabilidad de elegir un terminal antes de max_depth
        return random.choice(TERMINALS)

    func_data = random.choice(FUNCTIONS)
    func, arity = func_data[0], func_data[1]

    args = [generate_random_expression(max_depth, current_depth + 1) for _ in range(arity)]

    # Manejo especial para Pow si se usa directamente con aridad 2
    # if func == sympy.Pow:
    #     # Asegurarse que el exponente sea algo razonable, ej. un entero pequeño
    #     args[1] = sympy.Integer(random.choice([1, 2])) # Exponente 1 o 2

    return func(*args)

# 2. Función de Fitness
def calculate_fitness(individual_expr):
    """
    Calcula el fitness de un individuo (expresión SymPy).
    Menor error es mejor fitness.
    """
    try:
        # Convertir la expresión SymPy a una función Python evaluable
        # Nota: 'evalf()' puede ser necesario si hay floats en la expresión
        # Para seguridad, usamos parse_mathematica que es más restrictivo que eval()
        # O mejor, lambdify que es lo más seguro y eficiente
        func_lambda = lambdify(X_SYM, individual_expr, 'numpy')
        y_predicted = func_lambda(X_POINTS)

        # Manejar casos donde y_predicted no es un array (e.g. si la expresión es una constante)
        if not isinstance(y_predicted, np.ndarray):
            y_predicted = np.full_like(Y_TARGET, y_predicted)
        
        # Calcular el Error Cuadrático Medio (MSE)
        # Suma de errores al cuadrado, para evitar que números muy grandes dominen.
        # Queremos minimizar este error.
        error = np.sum((y_predicted - Y_TARGET)**2)

        # Evitar división por cero o valores infinitos
        if np.isinf(error) or np.isnan(error) or error > 1e10: # Penalizar errores muy grandes
            return 1e12 # Un valor de error muy alto (mal fitness)
        
        # Devolvemos el error. Un valor más bajo es mejor.
        # Sumamos una pequeña constante para evitar fitness de 0 que podría causar problemas
        # si luego se usa para divisiones (ej. 1/error)
        return error + 1e-6 

    except (TypeError, SyntaxError, OverflowError, ValueError, ZeroDivisionError, AttributeError) as e:
        # Si la expresión no es válida o causa un error durante la evaluación
        print(f"Error evaluando {individual_expr}: {e}")
        return 1e12 # Un valor de error muy alto (mal fitness)

# 3. Selección (Torneo)
def tournament_selection(population_with_fitness, k=TOURNAMENT_SIZE):
    """Selecciona un individuo usando selección por torneo."""
    selected_tournament = random.sample(population_with_fitness, k)
    # Ordenar por fitness (el primer elemento de la tupla es el fitness)
    # Recordar que menor fitness (error) es mejor
    winner = min(selected_tournament, key=lambda item: item[0])
    return winner[1] # Devolver la expresión (individuo)

# 4. Cruce (Crossover)
def get_all_subexpressions(expr):
    """Devuelve una lista de todas las subexpresiones (nodos) de una expresión SymPy."""
    subs = [expr]
    if hasattr(expr, 'args'):
        for arg in expr.args:
            subs.extend(get_all_subexpressions(arg))
    return list(set(subs)) # Usar set para eliminar duplicados si los hubiera

def crossover(parent1_expr, parent2_expr):
    """
    Realiza el cruce de subárboles entre dos expresiones SymPy.
    Devuelve dos hijos.
    """
    child1_expr, child2_expr = copy.deepcopy(parent1_expr), copy.deepcopy(parent2_expr)

    # Intentar varias veces encontrar puntos de cruce válidos
    for _ in range(5): # Intentar hasta 5 veces
        try:
            # Seleccionar un punto de cruce (subexpresión) en cada padre
            # No seleccionar la raíz completa para evitar cruces triviales si son átomos
            subexprs1 = [s for s in get_all_subexpressions(child1_expr) if s.args or s == child1_expr]
            subexprs2 = [s for s in get_all_subexpressions(child2_expr) if s.args or s == child2_expr]

            if not subexprs1 or not subexprs2: # Si no hay subexpresiones (ej. son solo átomos)
                return child1_expr, child2_expr # Devolver padres sin cambios

            point1 = random.choice(subexprs1)
            point2 = random.choice(subexprs2)
            
            # Crear hijos intercambiando los subárboles
            # xreplace es la forma de SymPy de sustituir subexpresiones
            temp_child1 = child1_expr.xreplace({point1: point2})
            temp_child2 = child2_expr.xreplace({point2: point1})

            # Verificar si el cruce fue exitoso (a veces xreplace no cambia si point1 o point2 son muy genéricos)
            if temp_child1 != child1_expr and temp_child2 != child2_expr:
                 return temp_child1, temp_child2

        except Exception as e:
            print(f"Error en cruce: {e}, padres: {parent1_expr}, {parent2_expr}")
            pass # Intentar de nuevo o devolver padres sin cambios

    return child1_expr, child2_expr # Devolver padres si el cruce falla repetidamente

# 5. Mutación
def mutate(individual_expr, max_depth=MAX_INITIAL_DEPTH):
    """
    Muta una expresión SymPy.
    Puede cambiar un terminal, un operador o reemplazar un subárbol.
    """
    mutated_expr = copy.deepcopy(individual_expr)
    
    # Obtener todas las subexpresiones
    subexprs = get_all_subexpressions(mutated_expr)
    if not subexprs:
        return generate_random_expression(max_depth=1) # Si es un átomo, generar uno nuevo simple

    point_to_mutate = random.choice(subexprs)

    # Opción 1: Reemplazar el nodo seleccionado por una nueva subexpresión aleatoria
    # Esto es una forma común y simple de mutación de subárbol
    random_subtree = generate_random_expression(max_depth=random.randint(0,max_depth-1)) # Profundidad aleatoria para el nuevo subárbol
    
    if point_to_mutate == mutated_expr : # Si se va a mutar la raíz
        mutated_expr = random_subtree
    else:
        try:
            mutated_expr = mutated_expr.xreplace({point_to_mutate: random_subtree})
        except Exception as e:
            print(f"Error en mutación (xreplace): {e}, expr: {mutated_expr}, point: {point_to_mutate}")
            # Si falla xreplace (raro si point_to_mutate es de get_all_subexpressions), generar uno nuevo
            return generate_random_expression(max_depth)

    # Podríamos añadir otras formas de mutación:
    # - Cambiar un terminal por otro terminal.
    # - Cambiar una función por otra función (de aridad compatible).
    # - Si es una constante, perturbarla ligeramente.

    return mutated_expr

# --- Algoritmo Principal ---
def genetic_programming():
    print("Iniciando Programación Genética...")
    print(f"Función objetivo: {TARGET_EXPR_SYM}")
    print(f"Evaluando en X_POINTS: {X_POINTS[:5]}... (total {len(X_POINTS)} puntos)")

    # 1. Inicializar Población
    population = [generate_random_expression(MAX_INITIAL_DEPTH) for _ in range(POPULATION_SIZE)]
    
    best_fitness_overall = float('inf')
    best_expr_overall = None
    history_best_fitness = []

    for gen in range(MAX_GENERATIONS):
        # 2. Evaluación de Fitness
        population_with_fitness = []
        for expr in population:
            fitness = calculate_fitness(expr)
            population_with_fitness.append((fitness, expr))

        # Ordenar por fitness (menor error es mejor)
        population_with_fitness.sort(key=lambda item: item[0])
        
        current_best_fitness = population_with_fitness[0][0]
        current_best_expr = population_with_fitness[0][1]
        history_best_fitness.append(current_best_fitness)

        if current_best_fitness < best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_expr_overall = current_best_expr
            print(f"Gen {gen}: Nuevo Mejor Fitness = {best_fitness_overall:.4e}, Expr = {sympy.simplify(best_expr_overall)}")
        elif gen % 10 == 0: # Imprimir progreso cada 10 generaciones
             print(f"Gen {gen}: Mejor Fitness Actual = {current_best_fitness:.4e}, Expr = {sympy.simplify(current_best_expr)}")


        # Criterio de parada (si encontramos una solución suficientemente buena)
        if best_fitness_overall < 1e-5: # Umbral de error muy bajo
            print("Solución óptima (o muy cercana) encontrada.")
            break
        
        # 3. Selección y Creación de Nueva Generación
        new_population = []

        # Elitismo: los mejores individuos pasan directamente
        for i in range(ELITISM_COUNT):
            new_population.append(population_with_fitness[i][1])

        # Llenar el resto de la nueva población
        while len(new_population) < POPULATION_SIZE:
            # Selección
            parent1 = tournament_selection(population_with_fitness)
            parent2 = tournament_selection(population_with_fitness)

            # Cruce
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2 # Sin cruce, pasan los padres

            # Mutación
            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        
        population = new_population

    print("\n--- Proceso Finalizado ---")
    print(f"Mejor expresión encontrada: {sympy.simplify(best_expr_overall)}")
    print(f"Con fitness (error): {best_fitness_overall:.4e}")

    # Simplificar la expresión final para una mejor visualización
    simplified_best_expr = sympy.simplify(best_expr_overall)
    print(f"Expresión simplificada: {simplified_best_expr}")

    # Graficar resultados
    plt.figure(figsize=(12, 6))

    # Gráfico de Fitness a lo largo de las generaciones
    plt.subplot(1, 2, 1)
    plt.plot(history_best_fitness)
    plt.title("Mejor Fitness (Error) por Generación")
    plt.xlabel("Generación")
    plt.ylabel("Error Cuadrático Medio (Menor es Mejor)")
    plt.grid(True)
    plt.yscale('log') # Escala logarítmica si los errores varían mucho

    # Gráfico de la función objetivo vs. la función encontrada
    plt.subplot(1, 2, 2)
    plt.plot(X_POINTS, Y_TARGET, 'b-', label=f"Objetivo: ${sympy.latex(TARGET_EXPR_SYM)}$")
    
    try:
        best_func_lambda = lambdify(X_SYM, simplified_best_expr, 'numpy')
        y_evolved = best_func_lambda(X_POINTS)
        if not isinstance(y_evolved, np.ndarray): # Si es una constante
            y_evolved = np.full_like(Y_TARGET, y_evolved)
        plt.plot(X_POINTS, y_evolved, 'r--', label=f"Evolucionada: ${sympy.latex(simplified_best_expr)}$")
    except Exception as e:
        print(f"No se pudo graficar la función evolucionada: {e}")
        plt.plot([], [], 'r--', label=f"Evolucionada (Error al graficar):\n${sympy.latex(simplified_best_expr)}$")


    plt.title("Función Objetivo vs. Función Evolucionada")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return simplified_best_expr, best_fitness_overall

# --- Ejecutar el algoritmo ---
if __name__ == "__main__":
    evolved_expression, final_fitness = genetic_programming()