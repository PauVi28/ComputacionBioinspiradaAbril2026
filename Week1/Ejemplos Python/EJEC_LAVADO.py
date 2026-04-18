# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:36:48 2025

@author: MARCELOFGB
"""

import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# --- 1. Definición de los Universos de Discurso ---
# x_peso: Rango de 0 a 10 kg con pasos de 0.1
x_peso = np.arange(0, 10.1, 0.1) 
# x_suciedad: Rango de 0 a 10 (escala subjetiva) con pasos de 0.1
x_suciedad = np.arange(0, 10.1, 0.1)
# x_duracion: Rango de 0 a 120 minutos con pasos de 1
x_duracion = np.arange(0, 121, 1)

# --- 2. Creación de las Funciones de Pertenencia ---

# A. Peso de la Ropa (Input Antecedent)
peso = ctrl.Antecedent(x_peso, 'peso')
peso['ligero'] = fuzz.trapmf(x_peso, [0, 0, 2, 4])
peso['medio'] = fuzz.trimf(x_peso, [3, 6, 9])
peso['pesado'] = fuzz.trapmf(x_peso, [7, 10, 10, 10])

# B. Nivel de Suciedad (Input Antecedent)
suciedad = ctrl.Antecedent(x_suciedad, 'suciedad')
suciedad['poca'] = fuzz.trapmf(x_suciedad, [0, 0, 2, 5])
suciedad['normal'] = fuzz.trimf(x_suciedad, [3, 6, 9])
suciedad['mucha'] = fuzz.trapmf(x_suciedad, [7, 10, 10, 10])

# C. Duración del Ciclo de Lavado (Output Consequent)
duracion = ctrl.Consequent(x_duracion, 'duracion')
duracion['corta'] = fuzz.trapmf(x_duracion, [0, 0, 15, 30])
duracion['media'] = fuzz.trimf(x_duracion, [25, 45, 65])
duracion['larga'] = fuzz.trimf(x_duracion, [55, 75, 95])
duracion['muy_larga'] = fuzz.trapmf(x_duracion, [85, 105, 120, 120])


# --- 3. Graficar las Funciones de Pertenencia ---
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(10, 12))

# Gráfico para Peso de la Ropa
ax0.plot(x_peso, peso['ligero'].mf, 'b', linewidth=1.5, label='Ligero') # .mf accede a la función de pertenencia
ax0.plot(x_peso, peso['medio'].mf, 'g', linewidth=1.5, label='Medio')
ax0.plot(x_peso, peso['pesado'].mf, 'r', linewidth=1.5, label='Pesado')
ax0.set_title('Funciones de Pertenencia - Peso de la Ropa')
ax0.set_xlabel('Peso (kg)')
ax0.set_ylabel('Grado de Pertenencia')
ax0.legend()
ax0.grid(True, linestyle=':', alpha=0.7)

# Gráfico para Nivel de Suciedad
ax1.plot(x_suciedad, suciedad['poca'].mf, 'b', linewidth=1.5, label='Poca Suciedad')
ax1.plot(x_suciedad, suciedad['normal'].mf, 'g', linewidth=1.5, label='Normalmente Sucio')
ax1.plot(x_suciedad, suciedad['mucha'].mf, 'r', linewidth=1.5, label='Muy Sucio')
ax1.set_title('Funciones de Pertenencia - Nivel de Suciedad')
ax1.set_xlabel('Nivel de Suciedad (escala 0-10)')
ax1.set_ylabel('Grado de Pertenencia')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# Gráfico para Duración del Ciclo de Lavado
ax2.plot(x_duracion, duracion['corta'].mf, 'b', linewidth=1.5, label='Corta')
ax2.plot(x_duracion, duracion['media'].mf, 'g', linewidth=1.5, label='Media')
ax2.plot(x_duracion, duracion['larga'].mf, 'r', linewidth=1.5, label='Larga')
ax2.plot(x_duracion, duracion['muy_larga'].mf, 'purple', linewidth=1.5, label='Muy Larga')
ax2.set_title('Funciones de Pertenencia - Duración del Ciclo de Lavado')
ax2.set_xlabel('Duración (minutos)')
ax2.set_ylabel('Grado de Pertenencia')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()


# --- 4. Definición de las Reglas Difusas ---
rule1 = ctrl.Rule(peso['ligero'] & suciedad['poca'], duracion['corta'])
rule2 = ctrl.Rule(peso['ligero'] & suciedad['normal'], duracion['media'])
rule3 = ctrl.Rule(peso['ligero'] & suciedad['mucha'], duracion['larga']) 

rule4 = ctrl.Rule(peso['medio'] & suciedad['poca'], duracion['media'])
rule5 = ctrl.Rule(peso['medio'] & suciedad['normal'], duracion['larga'])
rule6 = ctrl.Rule(peso['medio'] & suciedad['mucha'], duracion['muy_larga'])

rule7 = ctrl.Rule(peso['pesado'] & suciedad['poca'], duracion['media']) 
rule8 = ctrl.Rule(peso['pesado'] & suciedad['normal'], duracion['larga'])
rule9 = ctrl.Rule(peso['pesado'] & suciedad['mucha'], duracion['muy_larga']) 

# --- 5. Creación del Sistema de Control y la Simulación ---
lavadora_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
lavadora_sim = ctrl.ControlSystemSimulation(lavadora_ctrl)

# --- 6. Ejemplos de Ejecución ---
print("\n--- Ejecución de la Lógica Difusa para Casos de Ejemplo ---")

# Caso 1: Ropa Ligera, Poca Suciedad
peso_val1 = 1.9
suciedad_val1 = 2.0
lavadora_sim.input['peso'] = peso_val1
lavadora_sim.input['suciedad'] = suciedad_val1
lavadora_sim.compute() 
print(f"\nCaso 1: Peso = {peso_val1}kg, Suciedad = {suciedad_val1}")
print(f"Duración de lavado sugerida: {lavadora_sim.output['duracion']:.2f} minutos")
# Muestra el proceso de defuzzificación para este caso
duracion.view(sim=lavadora_sim)
plt.title(f"Defuzzificación para Caso 1 (Peso={peso_val1}kg, Suciedad={suciedad_val1})")
plt.show()


# Caso 2: Ropa Moderada, Suciedad Normal
peso_val2 = 5.0
suciedad_val2 = 6.0
lavadora_sim.input['peso'] = peso_val2
lavadora_sim.input['suciedad'] = suciedad_val2
lavadora_sim.compute()
print(f"\nCaso 2: Peso = {peso_val2}kg, Suciedad = {suciedad_val2}")
print(f"Duración de lavado sugerida: {lavadora_sim.output['duracion']:.2f} minutos")
duracion.view(sim=lavadora_sim)
plt.title(f"Defuzzificación para Caso 2 (Peso={peso_val2}kg, Suciedad={suciedad_val2})")
plt.show()

# Caso 3: Ropa Pesada, Muy Sucia
peso_val3 = 9.5
suciedad_val3 = 8.5
lavadora_sim.input['peso'] = peso_val3
lavadora_sim.input['suciedad'] = suciedad_val3
lavadora_sim.compute()
print(f"\nCaso 3: Peso = {peso_val3}kg, Suciedad = {suciedad_val3}")
print(f"Duración de lavado sugerida: {lavadora_sim.output['duracion']:.2f} minutos")
duracion.view(sim=lavadora_sim)
plt.title(f"Defuzzificación para Caso 3 (Peso={peso_val3}kg, Suciedad={suciedad_val3})")
plt.show()

# Caso 4: Ropa Pesada, Poca Suciedad (un caso interesante)
peso_val4 = 8.0
suciedad_val4 = 1.0
lavadora_sim.input['peso'] = peso_val4
lavadora_sim.input['suciedad'] = suciedad_val4
lavadora_sim.compute()
print(f"\nCaso 4: Peso = {peso_val4}kg, Suciedad = {suciedad_val4}")
print(f"Duración de lavado sugerida: {lavadora_sim.output['duracion']:.2f} minutos")
duracion.view(sim=lavadora_sim)
plt.title(f"Defuzzificación para Caso 4 (Peso={peso_val4}kg, Suciedad={suciedad_val4})")
plt.show()

# Caso 5: Ropa Ligera, Muy Sucia (otro caso interesante)
peso_val5 = 0.5
suciedad_val5 = 9.0
lavadora_sim.input['peso'] = peso_val5
lavadora_sim.input['suciedad'] = suciedad_val5
lavadora_sim.compute()
print(f"\nCaso 5: Peso = {peso_val5}kg, Suciedad = {suciedad_val5}")
print(f"Duración de lavado sugerida: {lavadora_sim.output['duracion']:.2f} minutos")
duracion.view(sim=lavadora_sim)
plt.title(f"Defuzzificación para Caso 5 (Peso={peso_val5}kg, Suciedad={suciedad_val5})")
plt.show()