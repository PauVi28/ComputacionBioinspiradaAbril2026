# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 01:12:20 2025

@author: MARCELOFGB
"""

import numpy as np
import random
import copy # Para copiar objetos complejos como las soluciones

# --- 1. Definición del Problema: Datos de Proveedores ---
# Diccionario donde la clave es el ítem y el valor es una lista de diccionarios
# Cada diccionario representa un proveedor para ese ítem.
PROVEEDORES_DATA = {
    "Tornillos M8": [
        {"nombre": "Tornillería Rápida", "costo": 0.10, "calidad": 4, "entrega": 3},
        {"nombre": "Sujetadores S.A.", "costo": 0.12, "calidad": 5, "entrega": 2},
        {"nombre": "MegaFix", "costo": 0.09, "calidad": 3, "entrega": 4},
        {"nombre": "Ferretería Central", "costo": 0.11, "calidad": 4, "entrega": 3},
    ],
    "Placas de Circuito": [
        {"nombre": "TechCircuit", "costo": 25.00, "calidad": 5, "entrega": 7},
        {"nombre": "PCB Express", "costo": 22.00, "calidad": 4, "entrega": 5},
        {"nombre": "GlobalBoards", "costo": 28.00, "calidad": 5, "entrega": 6},
    ],
    "Sensores de Temperatura": [
        {"nombre": "SensoriTec", "costo": 15.00, "calidad": 4, "entrega": 4},
        {"nombre": "ThermoSolutions", "costo": 14.50, "calidad": 5, "entrega": 3},
        {"nombre": "Automatismos XY", "costo": 16.00, "calidad": 3, "entrega": 5},
        {"nombre": "Metrología Digital", "costo": 14.80, "calidad": 4, "entrega": 3},
    ],
    "Baterías Litio": [
        {"nombre": "EnergyPack", "costo": 8.00, "calidad": 4, "entrega": 2},
        {"nombre": "PowerCells", "costo": 7.50, "calidad": 5, "entrega": 3},
        {"nombre": "CeldaDuradera", "costo": 8.20, "calidad": 3, "entrega": 2},
    ]
}

# Lista ordenada de los ítems que necesitamos adquirir.
# El orden define la "dimensión" de nuestra solución.
ITEMS_A_ADQUIRIR = list(PROVEEDORES_DATA.keys())

# --- 2. Parámetros del Algoritmo ABC ---
COLONY_SIZE = 20           # Número total de abejas empleadas (también es el número de fuentes de néctar)
MAX_ITERATIONS = 150       # Número máximo de ciclos de optimización
LIMIT = 7                  # Número de veces que una fuente puede no mejorar antes de ser abandonada por un explorador
FACTOR_PENALIZACION_CALIDAD = 10  # Multiplicador para penalizar baja calidad (ej: 1 punto de calidad faltante = 10 unidades de costo)
FACTOR_PENALIZACION_TIEMPO = 5    # Multiplicador para penalizar tiempo de entrega (ej: 1 día extra = 5 unidades de costo)

class ABCSupplierOptimizer:
    def __init__(self, proveedores_data, items_a_adquirir, colony_size, max_iterations, limit,
                 factor_calidad, factor_tiempo):
        self.proveedores_data = proveedores_data
        self.items_a_adquirir = items_a_adquirir
        self.num_items = len(items_a_adquirir)
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.limit = limit
        self.factor_calidad = factor_calidad
        self.factor_tiempo = factor_tiempo

        self.solutions = []  # Lista de fuentes (abejas empleadas con sus fuentes)
        self.best_global_solution = None
        self.best_global_objective_score = float('inf') # Queremos minimizar, así que inicializamos con infinito

        # Almacena el número de proveedores por cada ítem para facilitar la generación de soluciones
        self.num_suppliers_per_item = [
            len(proveedores_data[item]) for item in items_a_adquirir
        ]

        self._initialize_colony()

    def _calculate_objective_score(self, solution_indices):
        """
        Calcula el puntaje de penalización total para una combinación de proveedores.
        Un puntaje más bajo es mejor.
        Args:
            solution_indices (list): Una lista de índices, donde solution_indices[i]
                                      es el índice del proveedor elegido para items_a_adquirir[i].
        Returns:
            float: El puntaje de penalización total.
        """
        total_cost = 0.0
        total_quality = 0.0
        total_delivery_time = 0.0

        for i, item_name in enumerate(self.items_a_adquirir):
            supplier_idx = solution_indices[i]
            supplier = self.proveedores_data[item_name][supplier_idx]
            total_cost += supplier["costo"]
            total_quality += supplier["calidad"]
            total_delivery_time += supplier["entrega"]

        avg_quality = total_quality / self.num_items
        avg_delivery = total_delivery_time / self.num_items

        # Queremos minimizar el costo, maximizar la calidad (penalizar si es baja), minimizar el tiempo
        # Asumimos que la calidad ideal es 5. La penalización es (5 - calidad_promedio)
        objective_score = (
            total_cost +
            (5 - avg_quality) * self.factor_calidad +
            avg_delivery * self.factor_tiempo
        )
        return objective_score

    def _get_abc_fitness(self, objective_score):
        """
        Convierte el puntaje de penalización (objetivo a minimizar) a un fitness (a maximizar)
        para el algoritmo ABC.
        """
        # Evitar división por cero o números muy grandes.
        # Si el puntaje es negativo (lo que no debería pasar con nuestra función), lo corregimos.
        if objective_score >= 0:
            return 1 / (1 + objective_score)
        else: # En caso de puntajes negativos muy bajos (problema hipotético)
            return 1 + abs(objective_score) # Cuanto más negativo, mayor fitness

    def _initialize_colony(self):
        """
        Inicializa aleatoriamente las fuentes de néctar (soluciones) para las abejas empleadas.
        Cada solución es una combinación de proveedores.
        """
        print("Inicializando la colonia de abejas...")
        for _ in range(self.colony_size):
            # Una solución es una lista de índices de proveedor, uno por cada ítem
            solution_indices = [
                random.randrange(self.num_suppliers_per_item[i])
                for i in range(self.num_items)
            ]
            objective_score = self._calculate_objective_score(solution_indices)
            fitness = self._get_abc_fitness(objective_score)

            self.solutions.append({
                "indices": solution_indices,
                "objective_score": objective_score,
                "fitness": fitness,
                "trial_count": 0 # Conteo de intentos sin mejora
            })

            # Actualizar la mejor solución global encontrada hasta ahora
            if objective_score < self.best_global_objective_score:
                self.best_global_objective_score = objective_score
                self.best_global_solution = copy.deepcopy(solution_indices)

        print(f"Colonia inicializada con {self.colony_size} abejas.")
        print(f"Mejor solución inicial: {round(self.best_global_objective_score, 2)}")

    def _generate_neighbor(self, current_solution_indices):
        """
        Genera una solución vecina (abeja buscando alrededor de su fuente).
        Cambia aleatoriamente *un* proveedor para *un* ítem.
        """
        neighbor_indices = list(current_solution_indices) # Copia la solución actual
        
        # Elige aleatoriamente un ítem para cambiar su proveedor
        item_to_change_idx = random.randrange(self.num_items)
        
        # Obtiene el número total de proveedores para ese ítem
        num_suppliers_for_item = self.num_suppliers_per_item[item_to_change_idx]
        
        if num_suppliers_for_item <= 1: # Si solo hay un proveedor para este item, no se puede cambiar.
             return neighbor_indices # No hay un verdadero vecino distinto
        
        # Elige un nuevo proveedor aleatorio para ese ítem, asegurándose de que no sea el mismo
        new_supplier_idx = random.randrange(num_suppliers_for_item)
        while new_supplier_idx == neighbor_indices[item_to_change_idx]:
            new_supplier_idx = random.randrange(num_suppliers_for_item)
            
        neighbor_indices[item_to_change_idx] = new_supplier_idx
        return neighbor_indices

    def _run_employed_bees_phase(self):
        """
        Fase de las abejas empleadas: Cada abeja explora el vecindario de su fuente.
        """
        for i in range(self.colony_size):
            current_bee = self.solutions[i]
            
            # Genera una solución vecina
            new_solution_indices = self._generate_neighbor(current_bee["indices"])
            new_objective_score = self._calculate_objective_score(new_solution_indices)
            new_fitness = self._get_abc_fitness(new_objective_score)

            # Compara y actualiza si la nueva solución es mejor (mayor fitness, menor score objetivo)
            if new_fitness > current_bee["fitness"]: # O new_objective_score < current_bee["objective_score"]
                current_bee["indices"] = new_solution_indices
                current_bee["objective_score"] = new_objective_score
                current_bee["fitness"] = new_fitness
                current_bee["trial_count"] = 0  # Reinicia el contador de intentos
            else:
                current_bee["trial_count"] += 1 # Incrementa el contador de intentos
            
            # Actualizar la mejor solución global
            if new_objective_score < self.best_global_objective_score:
                self.best_global_objective_score = new_objective_score
                self.best_global_solution = copy.deepcopy(new_solution_indices)

    def _run_onlooker_bees_phase(self):
        """
        Fase de las abejas observadoras: Seleccionan fuentes basándose en la probabilidad (fitness)
        y luego exploran sus vecindarios.
        """
        # Calcular las probabilidades de selección para cada fuente
        total_fitness = sum(s["fitness"] for s in self.solutions)
        if total_fitness == 0: # Evitar división por cero si todas las fitness son 0
            probabilities = [1.0 / self.colony_size] * self.colony_size
        else:
            probabilities = [s["fitness"] / total_fitness for s in self.solutions]

        for _ in range(self.colony_size): # El mismo número de observadoras que de empleadas
            # Seleccionar una fuente basada en las probabilidades (método de la ruleta)
            chosen_source_idx = np.random.choice(range(self.colony_size), p=probabilities)
            chosen_bee = self.solutions[chosen_source_idx]

            # Generar una solución vecina para la fuente elegida
            new_solution_indices = self._generate_neighbor(chosen_bee["indices"])
            new_objective_score = self._calculate_objective_score(new_solution_indices)
            new_fitness = self._get_abc_fitness(new_objective_score)

            # Compara y actualiza si la nueva solución es mejor
            if new_fitness > chosen_bee["fitness"]:
                chosen_bee["indices"] = new_solution_indices
                chosen_bee["objective_score"] = new_objective_score
                chosen_bee["fitness"] = new_fitness
                chosen_bee["trial_count"] = 0
            else:
                chosen_bee["trial_count"] += 1
            
            # Actualizar la mejor solución global
            if new_objective_score < self.best_global_objective_score:
                self.best_global_objective_score = new_objective_score
                self.best_global_solution = copy.deepcopy(new_solution_indices)


    def _run_scout_bees_phase(self):
        """
        Fase de las abejas exploradoras: Identifican fuentes abandonadas (no mejoradas)
        y las reemplazan con nuevas fuentes aleatorias.
        """
        for i in range(self.colony_size):
            if self.solutions[i]["trial_count"] >= self.limit:
                print(f"  -> Fuente {i} agotada. Abeja exploradora en acción...")
                # Generar una nueva fuente aleatoria para esta abeja
                new_solution_indices = [
                    random.randrange(self.num_suppliers_per_item[j])
                    for j in range(self.num_items)
                ]
                new_objective_score = self._calculate_objective_score(new_solution_indices)
                new_fitness = self._get_abc_fitness(new_objective_score)

                self.solutions[i] = {
                    "indices": new_solution_indices,
                    "objective_score": new_objective_score,
                    "fitness": new_fitness,
                    "trial_count": 0  # Reiniciar el contador para la nueva fuente
                }
                # Es posible que la nueva fuente aleatoria sea mejor que la mejor global
                if new_objective_score < self.best_global_objective_score:
                    self.best_global_objective_score = new_objective_score
                    self.best_global_solution = copy.deepcopy(new_solution_indices)


    def optimize(self):
        """
        Ejecuta el algoritmo ABC a través de las fases definidas.
        """
        print("\nComenzando la optimización con Algoritmo ABC...")
        for iteration in range(self.max_iterations):
            # 1. Fase de Abejas Empleadas
            self._run_employed_bees_phase()

            # 2. Fase de Abejas Observadoras
            self._run_onlooker_bees_phase()

            # 3. Fase de Abejas Exploradoras
            self._run_scout_bees_phase()

            # Reportar el progreso
            print(f"Iteración {iteration + 1}/{self.max_iterations}: Mejor puntaje actual = {round(self.best_global_objective_score, 2)}")
            
            # Opcional: Criterio de parada si la mejora es mínima

        print("\nOptimización finalizada.")
        return self.best_global_solution, self.best_global_objective_score

    def get_solution_details(self, solution_indices):
        """
        Formatea y devuelve los detalles de una solución de proveedor.
        """
        details = []
        total_cost = 0.0
        total_quality = 0.0
        total_delivery_time = 0.0

        for i, item_name in enumerate(self.items_a_adquirir):
            supplier_idx = solution_indices[i]
            supplier = self.proveedores_data[item_name][supplier_idx]
            
            details.append(f"  - Ítem: {item_name}")
            details.append(f"    Proveedor: {supplier['nombre']}")
            details.append(f"    Costo: ${supplier['costo']:.2f}")
            details.append(f"    Calidad: {supplier['calidad']}/5")
            details.append(f"    Entrega: {supplier['entrega']} días")
            details.append("-" * 30)

            total_cost += supplier['costo']
            total_quality += supplier['calidad']
            total_delivery_time += supplier['entrega']
            
        avg_quality = total_quality / self.num_items
        avg_delivery = total_delivery_time / self.num_items
        
        # Calcular el puntaje objetivo final (lo mismo que en _calculate_objective_score)
        objective_score = (
            total_cost +
            (5 - avg_quality) * self.factor_calidad +
            avg_delivery * self.factor_tiempo
        )

        details.append("\nResumen de la Solución:")
        details.append(f"  Costo Total Bruto: ${total_cost:.2f}")
        details.append(f"  Calidad Promedio: {avg_quality:.2f}/5")
        details.append(f"  Tiempo de Entrega Promedio: {avg_delivery:.2f} días")
        details.append(f"  Puntaje de Penalización Total (Objetivo): {objective_score:.2f}")

        return "\n".join(details)


# --- Ejecución del Algoritmo ---
if __name__ == "__main__":
    optimizer = ABCSupplierOptimizer(
        PROVEEDORES_DATA,
        ITEMS_A_ADQUIRIR,
        COLONY_SIZE,
        MAX_ITERATIONS,
        LIMIT,
        FACTOR_PENALIZACION_CALIDAD,
        FACTOR_PENALIZACION_TIEMPO
    )

    best_solution_indices, best_score = optimizer.optimize()

    print("\n" + "="*50)
    print("RESULTADO DE LA OPTIMIZACIÓN ABC")
    print("="*50)
    print(f"\nMejor puntaje de penalización encontrado: {best_score:.2f}")
    print("\nDetalles de la mejor combinación de proveedores:")
    print(optimizer.get_solution_details(best_solution_indices))

    # Ejemplo de cómo sería la solución real (índices transformados a nombres)
    print("\nCombinación de proveedores (solo nombres):")
    for i, item_name in enumerate(ITEMS_A_ADQUIRIR):
        supplier_idx = best_solution_indices[i]
        supplier_name = PROVEEDORES_DATA[item_name][supplier_idx]["nombre"]
        print(f"  - {item_name}: {supplier_name}")