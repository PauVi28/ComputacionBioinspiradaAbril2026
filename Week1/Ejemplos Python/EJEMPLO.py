# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:59:55 2025

@author: MARCELOFGB
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Definir Variables del Universo (Rangos)
# ------------------------------------------
universo_antiguedad = np.arange(0, 25, 1)  # 0 a 24 meses
universo_adopcion = np.arange(0, 101, 1) # 0 a 100 %
universo_novedad = np.arange(0, 11, 1)   # 0 a 10 puntos

# Crear las variables difusas (antecedentes y consecuente)
antiguedad = ctrl.Antecedent(universo_antiguedad, 'antiguedad')
adopcion = ctrl.Antecedent(universo_adopcion, 'adopcion')
novedad = ctrl.Consequent(universo_novedad, 'novedad')


# 2. Definir Funciones de Pertenencia
# -----------------------------------

# Antigüedad:
# 'nuevo': Usaremos una sigmoidal en forma de Z (valor negativo para el parámetro de ancho/pendiente)
# skfuzzy.membership.sigmf(x, center, width_param)
# center: punto de inflexión de la sigmoide.
# width_param: controla la pendiente. Positivo para forma S, negativo para forma Z.
antiguedad['nuevo'] = fuzz.sigmf(antiguedad.universe, 6, -1)  # Inflexión en 6 meses, pendiente descendente
antiguedad['establecido'] = fuzz.trimf(antiguedad.universe, [4, 12, 20]) # Triangular para variedad
antiguedad['antiguo'] = fuzz.sigmf(antiguedad.universe, 18, 1)   # Inflexión en 18 meses, pendiente ascendente (S-shape)

# Adopción:
adopcion['baja'] = fuzz.sigmf(adopcion.universe, 20, -0.2)  # Inflexión en 20%, pendiente descendente más suave
adopcion['media'] = fuzz.gaussmf(adopcion.universe, 50, 15) # Gaussiana para variedad
adopcion['alta'] = fuzz.sigmf(adopcion.universe, 70, 0.2)   # Inflexión en 70%, pendiente ascendente más suave

# Novedad (usaremos triangulares para la salida, común para defuzzificación como centroide)
novedad['poco_novedoso'] = fuzz.trimf(novedad.universe, [0, 0, 4])
novedad['algo_novedoso'] = fuzz.trimf(novedad.universe, [2, 5, 8])
novedad['muy_novedoso'] = fuzz.trimf(novedad.universe, [6, 10, 10])


# Visualizar funciones de pertenencia (opcional pero recomendado)
# antiguedad.view()
# adopcion.view()
# novedad.view()
# plt.show() # Necesario si no se ejecuta en un entorno interactivo como Jupyter

# Visualizar las funciones de pertenencia de una variable específica, por ejemplo 'antiguedad'
antiguedad.view()
plt.title("Funciones de Pertenencia para Antigüedad")
plt.show()

adopcion.view()
plt.title("Funciones de Pertenencia para Adopcion")
plt.show()

novedad.view()
plt.title("Funciones de Pertenencia para Adopcion")
plt.show()




# 3. Definir Reglas Difusas
# -------------------------
regla1 = ctrl.Rule(antiguedad['nuevo'] & adopcion['baja'], novedad['muy_novedoso'])
regla2 = ctrl.Rule(antiguedad['establecido'] & (adopcion['baja'] | adopcion['media']), novedad['algo_novedoso'])
regla3 = ctrl.Rule(antiguedad['antiguo'] | adopcion['alta'], novedad['poco_novedoso'])
# regla4 = ctrl.Rule(antiguedad['nuevo'] & adopcion['media'], novedad['algo_novedoso']) # Podríamos añadir más reglas

# 4. Crear el Sistema de Control y Simulación
# -------------------------------------------
control_novedad = ctrl.ControlSystem([regla1, regla2, regla3]) # Añadir regla4 aquí si se define
simulacion_novedad = ctrl.ControlSystemSimulation(control_novedad)

# Ingresar algunos valores de entrada
simulacion_novedad.input['antiguedad'] = 3    # 3 meses de antigüedad
simulacion_novedad.input['adopcion'] = 10   # 10% de adopción

# Computar el resultado
simulacion_novedad.compute()

# Obtener y mostrar el resultado
resultado_novedad = simulacion_novedad.output['novedad']
print(f"Puntaje de Novedad: {resultado_novedad:.2f}")

# Visualizar el resultado de la inferencia en la función de pertenencia de salida
novedad.view(sim=simulacion_novedad)
plt.title(f"Novedad con Antigüedad=3 meses, Adopción=10% (Resultado: {resultado_novedad:.2f})")
plt.show()


# Probar con otros valores
simulacion_novedad.input['antiguedad'] = 20  # 20 meses
simulacion_novedad.input['adopcion'] = 80  # 80% de adopción
simulacion_novedad.compute()
resultado_novedad_2 = simulacion_novedad.output['novedad']
print(f"Puntaje de Novedad (2): {resultado_novedad_2:.2f}")
novedad.view(sim=simulacion_novedad)
plt.title(f"Novedad con Antigüedad=20 meses, Adopción=80% (Resultado: {resultado_novedad_2:.2f})")
plt.show()

