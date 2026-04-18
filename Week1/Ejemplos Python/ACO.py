# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 22:46:33 2025

@author: MARCELOFGB
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
#import math

# --- 1. Definición del Grafo y Ciudades ---
# Nombres de las ciudades
CITY_NAMES = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"
]
NUM_CITIES = len(CITY_NAMES)

# Coordenadas arbitrarias para visualización (para un diseño más estético)
CITY_POSITIONS = {
    "C1":  (2, 0),    # Destino (abajo centro)
    "C2":  (-1, 2),   # Centro izquierda
    "C3":  (5, 4),    # Derecha superior
    "C4":  (-2, 4),   # Izquierda superior
    "C5":  (-3, 6),   # Esquina superior izquierda
    "C6":  (-5, 5),   # Extremo izquierdo
    "C7":  (2, 6),    # Origen (arriba centro - alineado verticalmente con C1)
    "C8":  (4, 0),    # Esquina inferior derecha
    "C9":  (6, 2),    # Extremo derecho
    "C10": (0, 1)     # Abajo izquierda (cerca de C1 pero separado)
}
# Matriz de costos de aristas (distancias/tiempo de viaje). np.inf indica que no hay conexión directa.
# Simétrico, ya que se asume que el costo de A a B es el mismo que de B a A.
COSTS_MATRIX = np.array([
    # C1    C2     C3     C4      C5      C6      C7      C8    C9      C10
    [0,     120,   80,    np.inf, np.inf, np.inf, 63, 50,    60,    np.inf], # C1
    [120,   0,     90,    70,     np.inf, 60, np.inf, np.inf,np.inf, 40],     # C2
    [80,    90,    0,     np.inf, 60, np.inf, np.inf, np.inf,30,    50],     # C3
    [np.inf,70,    np.inf,0,      60,     20,     58, np.inf,np.inf, np.inf], # C4
    [np.inf,np.inf,np.inf,60,     0,      30,     np.inf, 60,np.inf, 70], # C5
    [np.inf,42,np.inf,20,     30,     0,      50, np.inf,30, np.inf], # C6
    [np.inf,np.inf,np.inf,50, np.inf, np.inf, 0,      np.inf,85, np.inf], # C7
    [50,    np.inf,np.inf,90, np.inf, 100, np.inf, 0,     20,    np.inf], # C8
    [60,    np.inf,30,    30, np.inf, 38, np.inf, 20,    0,     np.inf], # C9
    [np.inf,40,    50,    np.inf, np.inf, np.inf, np.inf, np.inf,np.inf, 0]      # C10
])

# Costos asociados a los nodos (ej. peajes, tiempo de espera en la ciudad, etc.)
# Estos costos se añaden al llegar a la ciudad (excluyendo la ciudad de origen).
# Son un factor que las hormigas también intentan minimizar.
NODE_COSTS = np.array([
    10,  # C1
    15,  # C2
    8,   # C3
    7,   # C4
    5,   # C5
    6,   # C6
    20,  # C7
    4,   # C8
    9,   # C9
    12   # C10
])

# --- 2. Parámetros del Algoritmo ACO ---
NUM_ANTS = 50          # Número de hormigas
MAX_ITERATIONS = 500   # Número máximo de iteraciones
RHO = 0.1              # Tasa de evaporación de feromona (rho, ρ)
ALPHA = 1.0            # Influencia de la feromona (alfa, α)
BETA = 2.0             # Influencia de la heurística (beta, β)
Q = 100.0              # Cantidad de feromona total a depositar
INITIAL_PHEROMONE = 1.0 # Cantidad inicial de feromona en todas las aristas

# --- 3. Funciones Auxiliares ---

def calculate_total_path_cost(path, costs_matrix, node_costs):
    """
    Calcula el costo total de una ruta, incluyendo costos de aristas y nodos.
    El costo de un nodo se asocia al 'llegar' o 'pasar por' ese nodo.
    """
    total_cost = 0
    if len(path) <= 1:
        # Si la ruta tiene 0 o 1 nodo, el costo es 0 (no hay viaje)
        return total_cost

    # Agrega el costo de cada arista y el costo del nodo_destino de esa arista
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i+1]
        edge_cost = costs_matrix[from_node][to_node]
        if edge_cost == np.inf:
            return np.inf # Ruta inválida si hay una conexión inexistente
        
        total_cost += edge_cost
        total_cost += node_costs[to_node] # Suma el costo del nodo a donde se llega

    return total_cost

def initialize_pheromones(num_cities, initial_pheromone):
    """Inicializa la matriz de feromonas con un valor uniforme."""
    return np.full((num_cities, num_cities), initial_pheromone)

def calculate_heuristics(costs_matrix, node_costs):
    """
    Calcula la matriz de heurísticas (ηij), que es el inverso del costo total de transición.
    El costo de transición es el costo de la arista + el costo del nodo de destino.
    """
    heuristics_matrix = np.zeros_like(costs_matrix, dtype=float)
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            if costs_matrix[i][j] != np.inf and costs_matrix[i][j] > 0:
                # La heurística es inversamente proporcional al costo de la arista + el costo del nodo destino
                heuristics_matrix[i][j] = 1.0 / (costs_matrix[i][j] + node_costs[j])
            elif i == j: # No hay costo heurístico para transitar a sí mismo
                heuristics_matrix[i][j] = 0.0
            else: # Conexión inexistente
                heuristics_matrix[i][j] = 0.0
    return heuristics_matrix

def select_next_city(current_city_idx, visited_cities, pheromones, heuristics, alpha, beta, num_cities):
    """
    Selecciona la siguiente ciudad a visitar basada en la regla de decisión probabilística.
    """
    probabilities = []
    possible_next_cities = []
    
    # Calcular la visibilidad (numerador) para las ciudades no visitadas
    for city_idx in range(num_cities):
        if city_idx not in visited_cities:
            pheromone = pheromones[current_city_idx][city_idx]
            heuristic = heuristics[current_city_idx][city_idx]
            
            # Si no hay conexión o heurística es 0, la probabilidad es 0.
            # Evita divisiones por cero o logaritmos de cero.
            if heuristic == 0:
                pi_j = 0.0
            else:
                pi_j = (pheromone**alpha) * (heuristic**beta)
            
            probabilities.append(pi_j)
            possible_next_cities.append(city_idx)
            
    total_prob_sum = sum(probabilities)

    if total_prob_sum == 0:
        # Si no hay ciudades posibles a visitar (todas visitadas o no hay conexión),
        # la hormiga está "atascada".
        return -1 # Indica que no puede avanzar

    # Normalizar las probabilidades
    probabilities = [p / total_prob_sum for p in probabilities]
    
    # Elegir la siguiente ciudad basado en las probabilidades
    next_city_idx = random.choices(possible_next_cities, weights=probabilities, k=1)[0]
    return next_city_idx

# --- 4. Algoritmo ACO Principal ---

def solve_aco_routing(start_city_name, end_city_name, costs_matrix, node_costs):
    """
    Implementa el algoritmo ACO para encontrar la ruta óptima entre dos ciudades.
    """
    start_city_idx = CITY_NAMES.index(start_city_name)
    end_city_idx = CITY_NAMES.index(end_city_name)

    if start_city_idx == end_city_idx:
        return [start_city_idx], 0.0, "Las ciudades de origen y destino son las mismas."

    pheromones = initialize_pheromones(NUM_CITIES, INITIAL_PHEROMONE)
    heuristics = calculate_heuristics(costs_matrix, node_costs)

    best_global_path = None
    best_global_cost = np.inf

    print(f"\nIniciando búsqueda de ruta de {start_city_name} a {end_city_name} con ACO...")

    for iteration in range(MAX_ITERATIONS):
        paths_this_iteration = []
        costs_this_iteration = []

        for ant_id in range(NUM_ANTS):
            current_path = [start_city_idx]
            visited_cities = {start_city_idx} # Usar un set para búsquedas rápidas
            current_city = start_city_idx

            while current_city != end_city_idx:
                next_city = select_next_city(current_city, visited_cities, pheromones, heuristics, ALPHA, BETA, NUM_CITIES)
                
                if next_city == -1: # La hormiga está atascada
                    current_path = [] # Marcar como ruta inválida
                    break
                
                current_path.append(next_city)
                visited_cities.add(next_city)
                current_city = next_city
                
                # Opcional: limitar la longitud de la ruta para evitar ciclos infinitos si no hay camino o el grafo es denso
                if len(current_path) > NUM_CITIES * 2: # Por ejemplo, el doble de ciudades
                    current_path = []
                    break

            if current_path and current_path[-1] == end_city_idx: # Solo si la hormiga llegó al destino
                cost = calculate_total_path_cost(current_path, costs_matrix, node_costs)
                if cost != np.inf: # Si la ruta es válida
                    paths_this_iteration.append(current_path)
                    costs_this_iteration.append(cost)

                    if cost < best_global_cost:
                        best_global_cost = cost
                        best_global_path = list(current_path) # Copia la lista
                        # print(f"Iteración {iteration+1}: Nueva mejor ruta encontrada: {best_global_path}, Costo: {best_global_cost:.2f}")

        # --- Actualización de Feromonas ---
        # 1. Evaporación
        pheromones = (1 - RHO) * pheromones

        # 2. Deposición de feromonas
        for i in range(len(paths_this_iteration)):
            path = paths_this_iteration[i]
            cost = costs_this_iteration[i]
            if cost > 0: # Evitar división por cero
                delta_pheromone = Q / cost
                for j in range(len(path) - 1):
                    pheromones[path[j]][path[j+1]] += delta_pheromone # Ant-Cycle strategy

        # Estrategia Elitist Ant System: La mejor ruta global refuerza su camino
        if best_global_path is not None and best_global_cost != np.inf and best_global_cost > 0:
            elite_delta_pheromone = Q / best_global_cost
            for j in range(len(best_global_path) - 1):
                pheromones[best_global_path[j]][best_global_path[j+1]] += elite_delta_pheromone * 2 # Un boost extra
        
        # Opcional: Limitar las feromonas para evitar la convergencia prematura o descontrolada
        # pheromones = np.clip(pheromones, 0.01, 1000) # Ejemplo de clipping

        if iteration % 50 == 0:
            print(f"Iteración {iteration+1}/{MAX_ITERATIONS}. Mejor costo actual: {best_global_cost if best_global_cost != np.inf else 'infinito'}")

    return best_global_path, best_global_cost, "Búsqueda finalizada."

# --- 5. Funciones de Visualización ---
def plot_graph(city_names, city_positions, costs_matrix, best_path=None, best_cost=None, start_node=None, end_node=None):
    """
    Dibuja el grafo de ciudades y resalta la mejor ruta encontrada.
    """
    G = nx.DiGraph() # Usamos Digraph si las conexiones son direccionales, o Graph si son bidireccionales.
                     # Para este caso, aunque los costos son simétricos, las feromonas pueden ser asimétricas,
                     # por lo que DiGraph es más apropiado para ACO.

    for i, city_name in enumerate(city_names):
        G.add_node(i, label=city_name)

    edge_labels = {}
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            cost = costs_matrix[i][j]
            if cost != np.inf and i != j:
                G.add_edge(i, j, weight=cost) # Añadir las aristas                
                edge_labels[(i,j)] = f"{cost:.0f}" # Mostrar costo en la arista
                

    pos = {i: city_positions[city_names[i]] for i in range(NUM_CITIES)}

    
    
    plt.figure(figsize=(10, 8))
    
    # Dibujar nodos
    node_colors = ['skyblue' for _ in range(NUM_CITIES)]
    if start_node is not None:
        node_colors[start_node] = 'lightgreen' # Origen
    if end_node is not None:
        node_colors[end_node] = 'salmon' # Destino
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
     
    # Dibujar etiquetas de nodos
    nx.draw_networkx_labels(G, pos, {i: city_names[i] for i in range(NUM_CITIES)}, font_size=9, font_weight='bold')

    # Dibujar aristas inactivas (todas las posibles conexiones)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', style='dashed')
    
    # Dibujar etiquetas de aristas (costos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    
    # Resaltar la mejor ruta si existe
    if best_path and len(best_path) > 1:
        path_edges = []
        for i in range(len(best_path) - 1):
            path_edges.append((best_path[i], best_path[i+1]))
        
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.5, arrowsize=20)
        
        # Mostrar el costo sobre la ruta resaltada
        path_edge_labels = { (best_path[i], best_path[i+1]): f"{costs_matrix[best_path[i]][best_path[i+1]]:.0f}" for i in range(len(best_path) - 1) }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=path_edge_labels, font_color='blue', font_size=10)

    plt.title(f"Ruta óptima entre ciudades (ACO)\nCosto Total: {best_cost:.2f}" if best_cost is not None else "Grafo de ciudades")
    plt.axis('off')
    plt.show()

# --- 6. Ejecución Principal ---
if __name__ == "__main__":
    
    # Mapeo de nombres de ciudades a sus índices
    city_to_idx = {name: i for i, name in enumerate(CITY_NAMES)}

    # Pedir ciudad de origen y destino al usuario
    print("Ciudades disponibles:")
    for i, name in enumerate(CITY_NAMES):
        print(f"  {i+1}. {name}")

    while True:
        try:
            origin_name = input("\nIngrese el nombre de la ciudad de origen: ").strip().title()
            if origin_name not in CITY_NAMES:
                raise ValueError("Ciudad no válida.")
            
            destination_name = input("Ingrese el nombre de la ciudad de destino: ").strip().title()
            if destination_name not in CITY_NAMES:
                raise ValueError("Ciudad no válida.")

            if origin_name == destination_name:
                print("La ciudad de origen y destino no pueden ser la misma. Intente de nuevo.")
                continue

            
            break
        except ValueError as e:
            print(f"Error: {e}. Por favor, ingrese un nombre de ciudad válido de la lista.")

    # Ejecutar el algoritmo ACO
    best_path_indices, total_cost, message = solve_aco_routing(origin_name, destination_name, COSTS_MATRIX, NODE_COSTS)

    print("\n--- Resultados del Algoritmo ACO ---")
    print(f"Ruta buscada: {origin_name} -> {destination_name}")
    print(f"Mensaje: {message}")

    if best_path_indices is not None and total_cost != np.inf:
        best_path_names = [CITY_NAMES[idx] for idx in best_path_indices]
        print(f"Mejor ruta encontrada: {' -> '.join(best_path_names)}")
        print(f"Costo total de la mejor ruta: {total_cost:.2f}")

        # Preparar para la visualización
        start_node_idx = city_to_idx[origin_name]
        end_node_idx = city_to_idx[destination_name]
        
        plot_graph(CITY_NAMES, CITY_POSITIONS, COSTS_MATRIX, best_path_indices, total_cost, start_node_idx, end_node_idx)
    else:
        print("No se pudo encontrar una ruta válida entre las ciudades especificadas.")
        # Opcional: mostrar solo el grafo sin una ruta resaltada si no se encontró ninguna.
        plot_graph(CITY_NAMES, CITY_POSITIONS, COSTS_MATRIX, None, None, city_to_idx[origin_name], city_to_idx[destination_name])