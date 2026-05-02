# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 00:25:22 2025

@author: MARCELOFGB
"""
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
#from pymoo.visualization.scatter import Scatter # Importamos aunque usaremos Matplotlib directamente para mayor control.

# 1. Definición del Problema Multi-Objetivo
# Heredamos de pymoo.core.problem.Problem
class RutaCamionProblem(Problem):

    def __init__(self):
        super().__init__(n_var=1, # Una variable de decisión 'x'
                         n_obj=2, # Dos objetivos: minimizar tiempo, minimizar costo_carga_insuficiente
                         xl=np.array([0.0]), # Límite inferior para 'x'
                         xu=np.array([4.0])) # Límite superior para 'x'

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Método para evaluar los objetivos para un conjunto dado de variables de decisión X.
        X es una matriz donde cada fila es una solución candidata (un valor de 'x' en este caso).
        """
        # X[:, 0] accede a la primera columna (y única) de X, que son nuestros valores de 'x'.

        # Objetivo 1: Minimizar el tiempo de viaje
        # Cuanto menor sea 'x', menor el tiempo.
        f1 = X[:, 0]**2

        # Objetivo 2: Minimizar el costo por carga insuficiente (equivale a maximizar la carga útil)
        # Queremos que 'x' esté cerca de 2 para maximizar la carga.
        f2 = (X[:, 0] - 2)**2

        # Asignar los valores de los objetivos a la salida
        # out["F"] debe ser una matriz numpy de (n_soluciones, n_objetivos)
        out["F"] = np.column_stack([f1, f2])

# 2. Configuración y Ejecución del Algoritmo de Optimización
problem = RutaCamionProblem()

# Usamos NSGA-II, un algoritmo popular para optimización multiobjetivo
algorithm = NSGA2(
    pop_size=100, # Número de soluciones en cada generación
    n_offsprings=10, # Número de descendientes generados en cada generación
    eliminate_duplicates=True # Eliminar soluciones duplicadas de la población
)

# Criterio de terminación: 200 generaciones
termination = ('n_gen', 200) 

# Ejecutar el algoritmo de optimización
print("Iniciando la optimización multi-objetivo...")
res = minimize(problem,
               algorithm,
               termination,
               seed=1, # Para reproducibilidad de los resultados
               verbose=True) # Mostrar el progreso de la optimización

# Extraer los resultados del frente de Pareto
X_pareto = res.X # Variables de decisión (valores de 'x') de las soluciones en el frente de Pareto
F_pareto = res.F # Valores de los objetivos (tiempo, costo_carga_insuficiente) en el frente de Pareto

print("\nOptimización completada.")
print(f"Número de soluciones encontradas en el Frente de Pareto: {len(F_pareto)}")

# 3. Visualización del Espacio de Búsqueda y el Frente de Pareto

# Generar datos para todo el espacio de búsqueda en el dominio de los objetivos
# Esto nos permite ver todos los puntos posibles que el camión podría generar
# evaluando diferentes 'factores de compromiso de ruta' (valores de x).
x_range_for_plot = np.linspace(problem.xl[0], problem.xu[0], 500) # 500 puntos para una curva suave
# Necesitamos que cada punto 'x' sea una fila para el método _evaluate.
X_search_space_eval = x_range_for_plot.reshape(-1, 1)

# Crear una instancia temporal del problema para evaluar el espacio completo
# (No usamos el objeto 'problem' original para evitar modificarlo o interferir con los `out` de `minimize`)
dummy_problem = RutaCamionProblem()
out_dummy = {} # Diccionario auxiliar para el método _evaluate
dummy_problem._evaluate(X_search_space_eval, out_dummy)
F_search_space = out_dummy["F"] # Los valores de los objetivos para todo el espacio de búsqueda

# Graficar usando Matplotlib para tener control preciso sobre la superposición de datos
plt.figure(figsize=(10, 7))

# Plotear todo el espacio de búsqueda (todas las posibles soluciones)
plt.scatter(F_search_space[:, 0], F_search_space[:, 1],
            s=12, facecolors='none', edgecolors='gray', alpha=0.5,
            label='Espacio de Búsqueda Completo (Todas las Rutas Posibles)')

# Plotear el frente de Pareto encontrado por NSGA-II
plt.scatter(F_pareto[:, 0], F_pareto[:, 1],
            s=80, facecolors='red', edgecolors='red', marker='o', zorder=2,
            label='Frente de Pareto (Soluciones Óptimas No Dominadas)')

# Opcional: Unir los puntos del frente de Pareto para mostrar la "curva"
# Primero, ordenar los puntos del frente de Pareto por el primer objetivo para que la línea se dibuje correctamente
sorted_F_pareto = F_pareto[F_pareto[:, 0].argsort()]
plt.plot(sorted_F_pareto[:, 0], sorted_F_pareto[:, 1],
         color='red', linestyle='-', linewidth=2, alpha=0.8, zorder=1)


plt.title('Frente de Pareto para el Problema de Ruta de Camión')
plt.xlabel('Tiempo de Viaje (Objetivo a Minimizar - $f_1$)')
plt.ylabel('Costo por Carga Insuficiente (Objetivo a Minimizar - $f_2$)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Análisis de Resultados e Interpretación
print("\n--- Análisis de las Soluciones en el Frente de Pareto ---")
print("El Frente de Pareto representa las mejores soluciones de compromiso:")
print("No se puede mejorar un objetivo sin empeorar el otro.")

# Mostrar algunas de las soluciones del frente de Pareto
print("\nPrimeras 10 soluciones en el Frente de Pareto:")
print(f"{'X':<5} | {'Tiempo':<8} | {'Costo Carga Insuficiente':<25}")
print("-" * 50)
for i in range(min(10, len(F_pareto))):
    x_val = X_pareto[i][0]
    tiempo_viaje = F_pareto[i][0]
    costo_carga_insuficiente = F_pareto[i][1]
    print(f"{x_val:.2f} | {tiempo_viaje:.2f}     | {costo_carga_insuficiente:.2f}")

print("\n--- Interpretación del Gráfico ---")
print("1.  **Espacio de Búsqueda Completo (Puntos Grises):** Representa todas las combinaciones posibles de 'Tiempo de Viaje' y 'Costo por Carga Insuficiente' que la empresa de transporte podría obtener al elegir diferentes 'factores de compromiso de ruta' (valores de X entre 0 y 4).")
print("2.  **Frente de Pareto (Puntos Rojos y Línea Roja):** Es el conjunto de soluciones óptimas no dominadas. Estas son las mejores rutas posibles, donde:")
print("    *   Cualquier punto fuera del frente (por ejemplo, arriba y a la derecha) es 'dominado', lo que significa que hay una solución en el frente que es mejor o igual en ambos objetivos.")
print("    *   Para moverte a lo largo del frente de Pareto, siempre tendrás que compensar: Si disminuyes el 'Tiempo de Viaje' (te muevas a la izquierda en el Eje X), el 'Costo por Carga Insuficiente' aumentará (te moverás hacia arriba en el Eje Y).")
print("    *   Esto significa que la empresa debe decidir qué equilibrio entre tiempo y carga es más importante para cada envío en particular.")