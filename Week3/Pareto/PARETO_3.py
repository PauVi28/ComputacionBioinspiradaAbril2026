# -*- coding: utf-8 -*-
"""

@author: MARCELOFGB
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination

# --- 1. Definición del Problema Multi-Objetivo ---
class MaizAguaProblema(Problem):

    def __init__(self, area_total=1000):
        super().__init__(n_var=2,  # x1: densidad de siembra, x2: fertilizante
                         n_obj=2,  # F1: Producción de maíz, F2: Uso de agua
                         n_constr=0, # No hay restricciones de igualdad/desigualdad más allá de los límites
                         xl=np.array([0.0, 0.0]), # Límites inferiores para x1 (densidad), x2 (fertilizante)
                         xu=np.array([5.0, 1.0])) # Límites superiores

        self.area_total = area_total

    def _evaluate(self, X, out, *args, **kwargs):
        # Extraer las variables de decisión para cada solución en X
        x1 = X[:, 0]  # Densidad de siembra (plantas/m2)
        x2 = X[:, 1]  # Fertilizante (kg/m2)

        # --- Calcular las Funciones Objetivo ---

        # 1. Producción de Maíz (F1) - Queremos MAXIMIZAR
        # kg de maíz por metro cuadrado
        produccion_por_m2 = 50 * x1 + 30 * x2 - 5 * x1**2 - 2 * x2**2 
        # Asegurarse de que la producción no sea negativa (por si las funciones dan valores poco realistas en los límites)
        produccion_por_m2[produccion_por_m2 < 0] = 0 
        
        # Producción total en el área
        produccion_total = self.area_total * produccion_por_m2
        
        # Pymoo MINIMIZA por defecto. Para MAXIMIZAR la producción, MINIMIZAMOS su negativo.
        f1 = -produccion_total 

        # 2. Uso de Agua (F2) - Queremos MINIMIZAR
        # Litros de agua por metro cuadrado
        agua_por_m2 = 10 * x1 + 5 * x2 + x1**2 + 0.5 * x2**2
        
        # Uso total de agua en el área
        f2 = self.area_total * agua_por_m2

        # Asignar los valores calculados a la salida de pymoo
        out["F"] = np.column_stack([f1, f2])

# --- 2. Configuración y Ejecución del Algoritmo de Optimización ---

print("Inicializando el problema...")
problem = MaizAguaProblema(area_total=1000)

# Usamos NSGA-II, un algoritmo popular para optimización multi-objetivo
print("Configurando el algoritmo NSGA2...")
algorithm = NSGA2(pop_size=100) # Tamaño de la población de soluciones en cada generación

# Definimos una condición de terminación: 200 generaciones
print("Definiendo la terminación (200 generaciones)...")
termination = get_termination("n_gen", 200)

# Ejecutamos la optimización
print("Ejecutando la optimización (esto puede tomar un momento)...")
res = minimize(problem,
               algorithm,
               termination,
               seed=1,        # Para resultados reproducibles
               verbose=False) # Para no imprimir el progreso en la consola

print("\nOptimización completada.")

# --- 3. Resultados y Visualización ---

# Extraer el Frente de Pareto (las soluciones no dominadas en el espacio objetivo)
# Recordar que f1 es el negativo de la producción, así que lo invertimos para la visualización
frente_pareto_produccion = -res.F[:, 0]
frente_pareto_agua = res.F[:, 1]

# Extraer las variables de decisión correspondientes a las soluciones del Frente de Pareto
soluciones_pareto_x1 = res.X[:, 0]
soluciones_pareto_x2 = res.X[:, 1]


print("\n--- Resultados del Frente de Pareto ---")
print(f"Número de soluciones en el Frente de Pareto: {len(frente_pareto_produccion)}")

# Mostrar algunas de las soluciones encontradas
print("\nEjemplos de soluciones en el Frente de Pareto:")
print(f"{'x1 (plantas/m2)':<18} {'x2 (kg fert/m2)':<18} {'Producción Total (kg)':<25} {'Agua Total (Litros)':<25}")
print("-" * 90)
for i in range(min(10, len(frente_pareto_produccion))):
    print(f"{soluciones_pareto_x1[i]:<18.4f} {soluciones_pareto_x2[i]:<18.4f} {frente_pareto_produccion[i]:<25.2f} {frente_pareto_agua[i]:<25.2f}")
if len(frente_pareto_produccion) > 10:
    print("...")


# --- Gráfica del Frente de Pareto (Espacio Objetivo) ---
plt.figure(figsize=(10, 7))
scatter_plot = Scatter(title="Frente de Pareto: Producción de Maíz vs. Uso de Agua",
                       labels=['Producción de Maíz (kg)', 'Uso de Agua (Litros)'])
scatter_plot.add(res.F, s=50, facecolor="none", edgecolor="blue") # Todos los puntos encontrados
plt.scatter(frente_pareto_produccion, frente_pareto_agua, s=60, facecolor="red", edgecolor="red", label='Frente de Pareto') # Solo el frente
plt.xlabel("Producción de Maíz (kg)")
plt.ylabel("Uso de Agua (Litros)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().invert_xaxis() # Es más intuitivo si la producción (MAX) va de izq a derecha
plt.legend()
plt.show()


# --- Gráfica de las Variables de Decisión (Espacio de Diseño) ---
plt.figure(figsize=(9, 6))
scatter_design_space = Scatter(title="Espacio de Diseño de Soluciones de Pareto",
                               labels=['Densidad de Siembra (plantas/m²)', 'Fertilizante (kg/m²)'])
scatter_design_space.add(res.X, s=50, facecolor="none", edgecolor="green")
plt.scatter(soluciones_pareto_x1, soluciones_pareto_x2, s=60, facecolor="purple", edgecolor="purple", label='Soluciones de Pareto')
plt.xlabel("Densidad de Siembra (x1 - plantas/m²)")
plt.ylabel("Fertilizante (x2 - kg/m²)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# --- Análisis Adicional (Opcional) ---
# Puedes analizar cómo cambian x1 y x2 a lo largo del frente de Pareto
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(frente_pareto_produccion, soluciones_pareto_x1, c='blue', alpha=0.7)
plt.xlabel("Producción de Maíz (kg)")
plt.ylabel("Densidad de Siembra (x1)")
plt.title("X1 vs. Producción")
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().invert_xaxis() # Para que la producción vaya de izq a derecha

plt.subplot(1, 2, 2)
plt.scatter(frente_pareto_produccion, soluciones_pareto_x2, c='green', alpha=0.7)
plt.xlabel("Producción de Maíz (kg)")
plt.ylabel("Fertilizante (x2)")
plt.title("X2 vs. Producción")
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().invert_xaxis() # Para que la producción vaya de izq a derecha

plt.tight_layout()
plt.show()