# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:55:18 2025

@author: MARCELOFGB
"""

import matplotlib.pyplot as plt
import numpy as np
#import time

# --- Parámetros de la simulación ---
AREA_SIZE = 10
NUM_SENTINELS = 35
NUM_AERODESI = 10
MAX_STEPS = 200
VISUAL_RANGE = 2.5
SPEED = 0.6
NEUTRALIZATION_THRESHOLD = 10
### MODIFICACIÓN ###
# Se añade un parámetro para controlar la frecuencia de las instantáneas gráficas.
# Se generará un gráfico cada 15 pasos.
SNAPSHOT_INTERVAL = 5

SEARCH = 0
SWARM = 1

ACTIVE = 0
NEUTRALIZED = 1

class Sentinel:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.behavior = SEARCH
        self.target_x = None
        self.target_y = None

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.x = max(0, min(AREA_SIZE - 1, self.x))
        self.y = max(0, min(AREA_SIZE - 1, self.y))

    def get_position(self):
        return (self.x, self.y)

class Aerodeslizador:
    def __init__(self, id, x, y, is_decoy=False):
        self.id = id
        self.x = x
        self.y = y
        self.status = ACTIVE
        self.is_decoy = is_decoy
        self.nearby_sentinels = 0

    def get_position(self):
        return (self.x, self.y)

    def set_status(self, status):
        self.status = status

    def update_nearby_sentinels(self, sentinels):
        self.nearby_sentinels = 0
        for sentinel in sentinels:
            dist = np.linalg.norm(np.array(sentinel.get_position()) - np.array([self.x, self.y]))
            if dist < VISUAL_RANGE:
                self.nearby_sentinels += 1

def calculate_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def calculate_centroid(sentinels):
    if not sentinels:
        return (0, 0)
    sum_x = sum(s.x for s in sentinels)
    sum_y = sum(s.y for s in sentinels)
    return (sum_x / len(sentinels), sum_y / len(sentinels))

def update_sentinel_behavior(sentinel, active_aerodeslizadores, all_sentinels):
    closest_aerodeslizador = None
    min_dist_to_aerodeslizador = float('inf')

    for aerodeslizador in active_aerodeslizadores:
        dist = calculate_distance(sentinel.get_position(), aerodeslizador.get_position())
        if dist < min_dist_to_aerodeslizador:
            min_dist_to_aerodeslizador = dist
            closest_aerodeslizador = aerodeslizador

    DETECTION_THRESHOLD = VISUAL_RANGE * 1.5
    if closest_aerodeslizador and min_dist_to_aerodeslizador < DETECTION_THRESHOLD:
        sentinel.behavior = SWARM
        sentinel.target_x, sentinel.target_y = closest_aerodeslizador.get_position()
    elif sentinel.behavior == SWARM and sentinel.target_x is not None:
        pass 
    else:
        sentinel.behavior = SEARCH

    if sentinel.behavior == SWARM and sentinel.target_x is not None:
        target_pos = np.array([sentinel.target_x, sentinel.target_y])
        current_pos = np.array([sentinel.x, sentinel.y])

        sentinels_for_target = [s for s in all_sentinels if s.behavior == SWARM and s.target_x == sentinel.target_x and s.target_y == sentinel.target_y]
        
        centroid_pos = np.array([0.0,0.0])
        if sentinels_for_target:
            centroid_x, centroid_y = calculate_centroid(sentinels_for_target)
            centroid_pos = np.array([centroid_x, centroid_y])
        else:
            centroid_pos = target_pos 

        move_vector = np.array([0.0, 0.0])
        
        direction_to_centroid = centroid_pos - current_pos
        if np.linalg.norm(direction_to_centroid) > 0.1: 
            direction_to_centroid /= np.linalg.norm(direction_to_centroid)
            move_vector += direction_to_centroid * SPEED * 0.7

        direction_to_target = target_pos - current_pos
        if np.linalg.norm(direction_to_target) > 0.1: 
            direction_to_target /= np.linalg.norm(direction_to_target)
            move_vector += direction_to_target * SPEED * 0.3

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector) * SPEED 

        sentinel.move(move_vector[0], move_vector[1])

    elif sentinel.behavior == SEARCH:
        angle = np.random.uniform(0, 2 * np.pi)
        move_dist = SPEED * np.random.uniform(0.5, 1.2)
        dx = move_dist * np.cos(angle)
        dy = move_dist * np.sin(angle)
        sentinel.move(dx, dy)

# --- Inicialización ---
sentinels = []
for i in range(NUM_SENTINELS):
    sentinels.append(Sentinel(i, np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE)))

aerodeslizadores = []
decoy_index = np.random.randint(0, NUM_AERODESI)
for i in range(NUM_AERODESI):
    aerodeslizadores.append(Aerodeslizador(i, np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE), is_decoy=(i == decoy_index)))

print("Simulación iniciada. Centinelas: {}, Aerodeslizadores: {}".format(NUM_SENTINELS, NUM_AERODESI))

### MODIFICACIÓN ###
# La función `update_graphics` ha sido reescrita para generar una instantánea autónoma.
# Ya no actualiza un gráfico, sino que crea uno nuevo desde cero en cada llamada.
def show_snapshot(step, sentinels, aerodeslizadores):
    """
    Crea y muestra una instantánea del estado actual de la simulación.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, AREA_SIZE)
    ax.set_ylim(0, AREA_SIZE)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Instantánea de la Simulación: Paso {step}")
    ax.grid(True)

    # Dibujar Centinelas
    ax.plot([s.x for s in sentinels], [s.y for s in sentinels], 'bo', markersize=3, label='Centinelas')

    # Dibujar Aerodeslizadores activos
    legend_handles = [plt.Line2D([], [], marker='o', color='blue', linestyle='None', markersize=3, label='Centinelas')]
    added_labels = {'Centinelas'}

    for aero in aerodeslizadores:
        if aero.status == ACTIVE:
            marker = 'D' if aero.is_decoy else '^'
            color = 'gray' if aero.is_decoy else 'red'
            label = 'Aerodeslizador Señuelo' if aero.is_decoy else 'Aerodeslizador Real'
            
            ax.plot(aero.x, aero.y, marker=marker, color=color, markersize=10, label=label)
            ax.text(aero.x, aero.y + 0.2, str(aero.nearby_sentinels), color='black', fontsize=8, ha='center')
            
            # Añadir a la leyenda solo si la etiqueta es nueva
            if label not in added_labels:
                legend_handles.append(plt.Line2D([], [], marker=marker, color=color, linestyle='None', markersize=10, label=label))
                added_labels.add(label)

    ax.legend(handles=legend_handles, loc='upper right')
    plt.show()

# --- Bucle de Simulación ---
# Mostrar el estado inicial
show_snapshot(0, sentinels, aerodeslizadores)

for step in range(1, MAX_STEPS + 1):
    active_aerodeslizadores = [aero for aero in aerodeslizadores if aero.status == ACTIVE]

    if not active_aerodeslizadores:
        print(f"\nTodos los aerodeslizadores han sido neutralizados en el paso {step}.")
        break

    for aero in active_aerodeslizadores:
        aero.update_nearby_sentinels(sentinels)
        if aero.nearby_sentinels >= NEUTRALIZATION_THRESHOLD:
            aero.set_status(NEUTRALIZED)
            if not aero.is_decoy:
                print(f"Paso {step}: Aerodeslizador {aero.id} (Real) neutralizado por {aero.nearby_sentinels} centinelas.")
            else:
                print(f"Paso {step}: Aerodeslizador Señuelo {aero.id} neutralizado por {aero.nearby_sentinels} centinelas.")
                
    active_aerodeslizadores = [aero for aero in aerodeslizadores if aero.status == ACTIVE]
    
    # Si después de la comprobación ya no quedan activos, salimos del bucle
    if not active_aerodeslizadores:
        # Se mostrará la instantánea final fuera del bucle
        continue

    for sentinel in sentinels:
        update_sentinel_behavior(sentinel, active_aerodeslizadores, sentinels)

    ### MODIFICACIÓN ###
    # Se llama a la función de visualización solo en los pasos que son múltiplos
    # del intervalo definido, generando una instantánea.
    if step % SNAPSHOT_INTERVAL == 0:
        print(f"--- Generando instantánea en el paso {step} ---")
        show_snapshot(step, sentinels, aerodeslizadores)

else: 
    print(f"\nSe alcanzó el número máximo de pasos ({MAX_STEPS}). Simulación finalizada.")

# Mostrar siempre la instantánea del estado final
print("\n--- Generando instantánea final ---")
show_snapshot(step, sentinels, aerodeslizadores)

print("Simulación finalizada.")