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

# Paso 1: Definir la clase del Problema de Optimización Multiobjetivo
class MechanicalDesignProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=2,  # x1: Índice de Material, x2: Espesor del Componente
            n_obj=2,  # Objetivo 1: -Resistencia a la Fatiga (para minimizar), Objetivo 2: Costo de Producción (para minimizar)
            n_constr=0, # No hay restricciones explícitas aparte de los límites de las variables
            xl=np.array([0.1, 2.0]),  # Límite inferior para x1 y x2
            xu=np.array([2.0, 5.0])   # Límite superior para x1 y x2
        )

    # Este método evalúa las funciones objetivo para una dada set de variables
    def _evaluate(self, X, out, *args, **kwargs):
        x1 = X[:, 0]  # Extrae la columna para el Índice de Material
        x2 = X[:, 1]  # Extrae la columna para el Espesor del Componente

        # Objetivo 1: Maximizar la Resistencia a la Fatiga -> Minimizar -Resistencia a la Fatiga
        # Asumimos que la resistencia aumenta con un mejor material y mayor espesor
        fatigue_strength = 100 * x1 * x2
        f1 = -fatigue_strength  # pymoo necesita minimizar, así que negamos el objetivo de maximización

        # Objetivo 2: Minimizar el Costo de Producción
        # Asumimos que el costo aumenta cuadráticamente con la calidad del material y linealmente con el espesor
        production_cost = 10 * x1**2 + 5 * x2
        f2 = production_cost

        # pymoo espera los valores de los objetivos en una matriz llamada "F"
        out["F"] = np.column_stack([f1, f2])

# Paso 2: Crear una instancia del problema
problem = MechanicalDesignProblem()

# Paso 3: Definir el algoritmo de optimización (NSGA-II)
# pop_size: Tamaño de la población en cada generación
# elimianate_duplicates: Asegura que no haya individuos idénticos en la población
algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)

# Paso 4: Definir el criterio de terminación
# En este caso, el algoritmo se detendrá después de 200 generaciones.
termination = get_termination("n_gen", 200)

# Paso 5: Ejecutar la optimización
print("Iniciando la optimización multiobjetivo (NSGA-II)...")
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,      # Semilla para reproducibilidad de los resultados
    verbose=False, # No mostrar el progreso detallado de la optimización
    save_history=True # Guardar el historial de poblaciones (útil para análisis avanzados)
)
print("Optimización finalizada.")

# Paso 6: Extraer los resultados
# res.X contiene las variables de decisión (x1, x2) de la población final
X = res.X

# res.F contiene los valores de los objetivos (-f1, f2) de la población final
F = res.F

# --- Corrección para la visualización ---

# Importante: Para la visualización, revertimos el primer objetivo a su forma original (Resistencia a la Fatiga positiva)
F_plot = F.copy()
F_plot[:, 0] = -F_plot[:, 0] # Convertir -Resistencia a la Fatiga a Resistencia a la Fatiga positiva

# Visualizar el Frente de Pareto (Espacio de Objetivos)
# CORRECCIÓN: Usar 'axis_labels' para las etiquetas de los ejes
plot = Scatter(
    title="Frente de Pareto (Espacio de Objetivos)",
    axis_labels=["Resistencia a la Fatiga (Maximizar)", "Costo de Producción (Minimizar)"], # <--- CORREGIDO AQUÍ
    # El argumento 'labels' es para la leyenda si se añaden múltiples series de datos
    # labels=["Frente de Pareto"] # Esto es para la leyenda, Plot lo infiere si solo hay una serie
)
plot.add(F_plot, s=50, facecolors='none', edgecolors='blue', alpha=0.8) # Añadir los puntos del frente de Pareto
plot.do() # Esto dibuja la gráfica base

# Configuración adicional de matplotlib para mejorar la legibilidad
# Ya no es necesario plt.xlabel y plt.ylabel si se usa axis_labels en Scatter
# plt.xlabel("Resistencia a la Fatiga (Maximizar)")
# plt.ylabel("Costo de Producción (Minimizar)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Visualizar los Diseños Óptimos (Espacio de Decisión)
# CORRECCIÓN: Usar 'axis_labels' para las etiquetas de los ejes
plot_X = Scatter(
    title="Diseños Óptimos (Espacio de Decisión)",
    axis_labels=["Índice de Material (x1)", "Espesor del Componente (x2)"], # <--- CORREGIDO AQUÍ
    # labels=["Diseños Óptimos"] # Lo mismo, lo infiere si solo hay una serie
)
plot_X.add(X, s=50, facecolors='none', edgecolors='red', alpha=0.8) # Añadir los puntos en el espacio de decisión
plot_X.do() # Dibujar la gráfica base

# Configuración adicional de matplotlib
# Ya no es necesario plt.xlabel y plt.ylabel
plt.xlabel("Índice de Material (x1)")
plt.ylabel("Espesor del Componente (x2)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# Paso 8: Interpretación de los Resultados
print("\n--- Interpretación del Frente de Pareto ---")
print("El Frente de Pareto muestra el compromiso (trade-off) entre la Resistencia a la Fatiga y el Costo de Producción.")
print("Cada punto en la gráfica del Frente de Pareto representa un diseño de pieza donde no se puede mejorar un objetivo sin empeorar el otro.")
print("\nEjemplos de Diseños óptimos en el Frente de Pareto:")

# Ordenar los resultados por Resistencia a la Fatiga para ver la relación claramente
# Esto nos ayuda a mostrar ejemplos que van de "menor resistencia/costo" a "mayor resistencia/costo"
sorted_indices = np.argsort(F_plot[:, 0]) # Ordenar por Resistencia a la Fatiga (ya positiva)
F_sorted = F_plot[sorted_indices]
X_sorted = X[sorted_indices]

print(f"{'Resistencia Fatiga':<20} {'Costo Producción':<20} {'Índice Material (x1)':<20} {'Espesor (x2)':<20}")
print("-" * 80)

# Mostrar algunos puntos del "extremo inferior" (baja resistencia, bajo costo)
for i in range(min(5, len(F_sorted))):
    print(f"{F_sorted[i, 0]:<20.2f} {F_sorted[i, 1]:<20.2f} {X_sorted[i, 0]:<20.2f} {X_sorted[i, 1]:<20.2f}")

print("...")

# Mostrar algunos puntos del "extremo superior" (alta resistencia, alto costo)
for i in range(max(0, len(F_sorted) - 5), len(F_sorted)):
    print(f"{F_sorted[i, 0]:<20.2f} {F_sorted[i, 1]:<20.2f} {X_sorted[i, 0]:<20.2f} {X_sorted[i, 1]:<20.2f}")

print("\nConclusión:")
print("Un ingeniero puede elegir un punto del frente de Pareto basándose en las prioridades del proyecto.")
print("- Si el presupuesto es muy ajustado, se elegiría un punto hacia el extremo inferior-izquierda (menor resistencia, menor costo).")
print("- Si la durabilidad es crítica, se elegiría un punto hacia el extremo superior-derecha (mayor resistencia, mayor costo).")
print("- Los puntos intermedios ofrecen un equilibrio.")

plt.show() # Asegúrate de que las ventanas de los gráficos se cierren solo cuando se llame a show()