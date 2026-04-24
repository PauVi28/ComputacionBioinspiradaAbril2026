# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:30:22 2025

@author: MARCELOFGB
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. Definición de las Variables de Entrada (Antecedentes) y Salida (Consecuentes) ---

# Altitud_Error: Qué tan lejos estamos de la altitud objetivo (por ejemplo, 0m significa en objetivo)
# Rango: de -50m (demasiado bajo) a 50m (demasiado alto)
altitude_error = ctrl.Antecedent(np.arange(-50, 51, 1), 'altitude_error')

# Viento: La velocidad del viento percibida
# Rango: de 0 m/s (calma) a 30 m/s (fuerte)
wind_speed = ctrl.Antecedent(np.arange(0, 31, 1), 'wind_speed')

# Batería: Nivel de carga de la batería
# Rango: de 0% (vacía) a 100% (llena)
battery_level = ctrl.Antecedent(np.arange(0, 101, 1), 'battery_level')

# Velocidad_Vertical: Dónde queremos que se mueva el dron verticalmente
# Rango: de -10 m/s (descenso rápido) a 10 m/s (ascenso rápido)
vertical_speed = ctrl.Consequent(np.arange(-10, 11, 1), 'vertical_speed')

# Velocidad_Horizontal_Correccion: Cuánto hay que corregir horizontalmente (por ejemplo, contra el viento o para regresar)
# Rango: de 0 m/s (sin corrección) a 15 m/s (gran corrección)
horizontal_speed_correction = ctrl.Consequent(np.arange(0, 16, 1), 'horizontal_speed_correction')


#horizontal_speed_correction.defuzzify_method ='mom'
#horizontal_speed_correction.defuzzify_method ='centroid'
#horizontal_speed_correction.defuzzify_method ='bisector'
#horizontal_speed_correction.defuzzify_method ='som'
#horizontal_speed_correction.defuzzify_method ='lom'

# --- 2. Definición de las Funciones de Pertenencia ---

# Altitud_Error
altitude_error['demasiado_bajo'] = fuzz.trimf(altitude_error.universe, [-50, -50, -20])
altitude_error['bajo'] = fuzz.trimf(altitude_error.universe, [-25, -10, 0])
altitude_error['en_objetivo'] = fuzz.trimf(altitude_error.universe, [-5, 0, 5])
altitude_error['alto'] = fuzz.trimf(altitude_error.universe, [0, 10, 25])
altitude_error['demasiado_alto'] = fuzz.trimf(altitude_error.universe, [20, 50, 50])

# Viento
wind_speed['calma'] = fuzz.trimf(wind_speed.universe, [0, 0, 10])
wind_speed['moderado'] = fuzz.trimf(wind_speed.universe, [5, 15, 25])
wind_speed['fuerte'] = fuzz.trimf(wind_speed.universe, [20, 30, 30])

# Batería
battery_level['critica'] = fuzz.trimf(battery_level.universe, [0, 0, 15])
battery_level['baja'] = fuzz.trimf(battery_level.universe, [10, 25, 40])
battery_level['media'] = fuzz.trimf(battery_level.universe, [30, 55, 80])
battery_level['llena'] = fuzz.trimf(battery_level.universe, [70, 100, 100])

# Velocidad_Vertical
vertical_speed['descenso_rapido'] = fuzz.trimf(vertical_speed.universe, [-10, -10, -5])
vertical_speed['descenso_lento'] = fuzz.trimf(vertical_speed.universe, [-6, -3, 0])
vertical_speed['flotar'] = fuzz.trimf(vertical_speed.universe, [-1, 0, 1])
vertical_speed['ascenso_lento'] = fuzz.trimf(vertical_speed.universe, [0, 3, 6])
vertical_speed['ascenso_rapido'] = fuzz.trimf(vertical_speed.universe, [5, 10, 10])

# Velocidad_Horizontal_Correccion
horizontal_speed_correction['ninguna'] = fuzz.trimf(horizontal_speed_correction.universe, [0, 0, 5])
horizontal_speed_correction['pequeña'] = fuzz.trimf(horizontal_speed_correction.universe, [3, 6, 9])
horizontal_speed_correction['media'] = fuzz.trimf(horizontal_speed_correction.universe, [7, 10, 13])
horizontal_speed_correction['grande'] = fuzz.trimf(horizontal_speed_correction.universe, [11, 15, 15])

# --- 3. Visualización de las Funciones de Pertenencia ---
print("Visualizando Funciones de Pertenencia...")
altitude_error.view()
plt.show()
wind_speed.view()
plt.show()
battery_level.view()
plt.show()
vertical_speed.view()
plt.show()
horizontal_speed_correction.view()
plt.show()

# --- 4. Definición de las Reglas Difusas ---

# Reglas de control de altitud
rule1 = ctrl.Rule(altitude_error['demasiado_bajo'], vertical_speed['ascenso_rapido'])
rule2 = ctrl.Rule(altitude_error['bajo'], vertical_speed['ascenso_lento'])
rule3 = ctrl.Rule(altitude_error['en_objetivo'], vertical_speed['flotar'])
rule4 = ctrl.Rule(altitude_error['alto'], vertical_speed['descenso_lento'])
rule5 = ctrl.Rule(altitude_error['demasiado_alto'],vertical_speed['descenso_rapido'])

# Reglas de compensación de viento
rule6 = ctrl.Rule(wind_speed['calma'],horizontal_speed_correction['ninguna'])
rule7 = ctrl.Rule(wind_speed['moderado'],horizontal_speed_correction['media'])
rule8 = ctrl.Rule(wind_speed['fuerte'],horizontal_speed_correction['grande'])

# Reglas relacionadas con la bateria
rule9 = ctrl.Rule(battery_level['critica'], vertical_speed['descenso_rapido'])
rule10 = ctrl.Rule(battery_level['critica'], horizontal_speed_correction['grande'])
rule11 = ctrl.Rule(battery_level['baja'] & altitude_error['en_objetivo'],
                   vertical_speed['descenso_lento'])

# Reglas combinadas
rule12 = ctrl.Rule(altitude_error['en_objetivo'] & wind_speed['moderado'] & battery_level['media'],
                   (vertical_speed['flotar'],horizontal_speed_correction['media']))

rule13 = ctrl.Rule(altitude_error['demasiado_bajo'] & battery_level['critica'],
                   (vertical_speed['descenso_rapido'],horizontal_speed_correction['grande']))

                                 
#-- 5 ---- Creación del sistema de control difuso
# agrupación de reglas en un sistema de control
drone_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, 
                                 rule7, rule8, rule9, rule10, rule11, rule12, rule13])
#Crear la simulación del sistema
drone_sim = ctrl.ControlSystemSimulation(drone_ctrl)

# -- 6 Simulación de situaciones en el vuelo y visualización de la defuzzificación ---

def run_escenario (nombre_escenario, alt_err, wind_spd, bat_lvl ):
    print(f"\n ****** Escenario:{nombre_escenario} ******")
    drone_sim.input['altitude_error']=alt_err
    drone_sim.input['wind_speed']=wind_spd
    drone_sim.input['battery_level']=bat_lvl
    
    # Computar las salidas
    drone_sim.compute()
    
    # Mostrar los resultados
    print(f"\n Entradas: Altitud error ={alt_err} metros, Viento ={wind_spd} m/s, Batería ={bat_lvl}%")
    print(f"\n Velocidad vertical sugerida:{drone_sim.output['vertical_speed']:.2f} m/s")
    print(f"\n Corrección de velocidad horizontal sugerida:{drone_sim.output['horizontal_speed_correction']:.2f} m/s")
    
    # Decisión de retorno a la base 
    bat_critica_membership = fuzz.interp_membership(
        battery_level.universe, battery_level['critica'].mf,bat_lvl)
    
    bat_baja_membership = fuzz.interp_membership(
        battery_level.universe, battery_level['baja'].mf,bat_lvl)
    
    if bat_critica_membership >= 0.5:
        print("¡ATENCION! la carga de la batería es crítica. Iniciando retorno a la base")
    if bat_baja_membership >= 0.5:
        print("Bateria baja, considere regresar a la base")
    
    #Visualización de defuzzificación para cada salida
    print("Visualización de defuzzificación")
    
    vertical_speed.view (sim=drone_sim)
    plt.title(f"Defuzzificación velocidad vertical \n ({nombre_escenario})  ")
    
    horizontal_speed_correction.view (sim=drone_sim)
    plt.title(f"Defuzzificación V. Horizotal \n ({nombre_escenario})  ")
    
    plt.tight_layout()
    plt.show()

#Escenario 1
run_escenario("Vuelo normal con viento", alt_err=-5, wind_spd=12, bat_lvl=6 )


#Escenario 2
run_escenario("Batería Crítica (EMERGENCIA)", alt_err=30, wind_spd=5, bat_lvl=10 )




















