# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:06:37 2026

@author: MARCELOFGB
"""

import matplotlib.pyplot as plt
import numpy as np
#import time

# --- Parámetros de la simulación ---
# [TA] Estos valores configuran el "entorno de Matrix". 
# Nota: El PDF pedía 4 aerodeslizadores y 30 centinelas, aquí se usan 10 y 40 
# para hacer la simulación visualmente más rica, pero la lógica es la misma.
AREA_SIZE = 10
NUM_SENTINELS = 40
NUM_AERODESI = 10
MAX_STEPS = 200
VISUAL_RANGE = 2.1 # [TA] Representa el 'Visual' mencionado en el modelo matemático del FSA.
SPEED = 0.6        # [TA] Representa el tamaño de paso máximo ('Step') que un centinela puede dar.
NEUTRALIZATION_THRESHOLD = 10 # Condición del reto: 10 centinelas para neutralizar.

SNAPSHOT_INTERVAL = 5 #CADA CUANTO SE IMPRIME UNA INSTANTANEA

# [TA] Estados posibles de los centinelas (Peces Artificiales)
SEARCH = 0  # Comportamiento Aleatorio / Búsqueda ciega
SWARM = 1   # Comportamiento de Enjambre / Presa

# [TA] Estados de los Aerodeslizadores (Alimento)
ACTIVE = 0
NEUTRALIZED = 1

class Sentinel:
    # [TA] Representa el "Pez Artificial" (AF) del algoritmo FSA.
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.behavior = SEARCH # Inician explorando aleatoriamente.
        self.target_x = None
        self.target_y = None

    def move(self, dx, dy):
        # [TA] Actualiza la posición X_i (Estado del AF). 
        # Los max y min actúan como paredes invisibles para que no salgan de la zona 10x10.
        self.x += dx
        self.y += dy
        self.x = max(0, min(AREA_SIZE - 1, self.x))
        self.y = max(0, min(AREA_SIZE - 1, self.y))

    def get_position(self):
        return (self.x, self.y)

class Aerodeslizador:
    # [TA] Representa el "Alimento" o la "Función Objetivo" en el problema de optimización.
    def __init__(self, id, x, y, is_decoy=False):
        self.id = id
        self.x = x
        self.y = y
        self.status = ACTIVE
        self.is_decoy = is_decoy # [TA] Señuelo (Alimento de baja calidad) o Real (Alta calidad)
        self.nearby_sentinels = 0

    def get_position(self):
        return (self.x, self.y)

    def set_status(self, status):
        self.status = status

    def update_nearby_sentinels(self, sentinels):
        # [TA] Evalúa la "calidad de la solución". Cuantos más centinelas estén dentro
        # del VISUAL_RANGE, más cerca está de cumplirse la meta.
        self.nearby_sentinels = 0
        for sentinel in sentinels:
            dist = np.linalg.norm(np.array(sentinel.get_position()) - np.array([self.x, self.y]))
            if dist < VISUAL_RANGE: # Distancia Euclidiana (Ecuación 2 del PDF)
                self.nearby_sentinels += 1

def calculate_distance(pos1, pos2):
    # [TA] Implementación de la Ecuación 2 del PDF: dist_ij = ||X_i - X_j||
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def calculate_centroid(sentinels):
    # [TA] Implementación exacta de la Fórmula 5 del PDF.
    # X_centro = (1/n) * sumatoria(X_k). Promedio de las coordenadas del grupo.
    if not sentinels:
        return (0, 0)
    sum_x = sum(s.x for s in sentinels)
    sum_y = sum(s.y for s in sentinels)
    return (sum_x / len(sentinels), sum_y / len(sentinels))

def update_sentinel_behavior(sentinel, active_aerodeslizadores, all_sentinels):
    # [TA] ESTE ES EL CORAZÓN DEL ALGORITMO BIOINSPIRADO
    closest_aerodeslizador = None
    min_dist_to_aerodeslizador = float('inf')

    # 1. Búsqueda de alimento (Evaluar el entorno)--FITNESS min_dist_to_aerodeslizador 
    for aerodeslizador in active_aerodeslizadores:
        dist = calculate_distance(sentinel.get_position(), aerodeslizador.get_position())
        if dist < min_dist_to_aerodeslizador:
            min_dist_to_aerodeslizador = dist
            closest_aerodeslizador = aerodeslizador #calidad del alimento

    # Ampliamos un poco el rango de detección visual inicial para motivar el enjambre
    DETECTION_THRESHOLD = VISUAL_RANGE * 1.5 
    
    # [TA] Cambio de estado basado en el entorno
    if closest_aerodeslizador and min_dist_to_aerodeslizador < DETECTION_THRESHOLD:
        # Si encuentra comida, cambia a modo ENJAMBRE/PRESA
        sentinel.behavior = SWARM
        sentinel.target_x, sentinel.target_y = closest_aerodeslizador.get_position()
    elif sentinel.behavior == SWARM and sentinel.target_x is not None:
        pass 
    else:
        # Si no hay nada cerca, sigue buscando aleatoriamente
        sentinel.behavior = SEARCH

    # 2. Ejecución de los Comportamientos FSA
    if sentinel.behavior == SWARM and sentinel.target_x is not None:
        target_pos = np.array([sentinel.target_x, sentinel.target_y])
        current_pos = np.array([sentinel.x, sentinel.y])

        # [TA] COMPORTAMIENTO DE ENJAMBRE (Fórmula 5 aproximada)
        # Filtra solo los centinelas que van al mismo objetivo
        sentinels_for_target = [s for s in all_sentinels if s.behavior == SWARM and s.target_x == sentinel.target_x and s.target_y == sentinel.target_y]
         
        centroid_pos = np.array([0.0,0.0])
        if sentinels_for_target:
            centroid_x, centroid_y = calculate_centroid(sentinels_for_target)
            centroid_pos = np.array([centroid_x, centroid_y])
        else:
            centroid_pos = target_pos 

        move_vector = np.array([0.0, 0.0])
         
        # [TA] Combinación de vectores (Sustituye matemáticamente la lógica rígida del PDF)
        # Fuerza 1: Atracción al Centroide (70% de peso). Mantiene al enjambre unido.
        direction_to_centroid = centroid_pos - current_pos
        if np.linalg.norm(direction_to_centroid) > 0.1: 
            direction_to_centroid /= np.linalg.norm(direction_to_centroid) # Normaliza el vector
            move_vector += direction_to_centroid * SPEED * 0.7

        # Fuerza 2: Atracción a la Presa (30% de peso). Emula "Comportamiento de presa" (Fórmula 3).
        direction_to_target = target_pos - current_pos
        if np.linalg.norm(direction_to_target) > 0.1: 
            direction_to_target /= np.linalg.norm(direction_to_target)
            move_vector += direction_to_target * SPEED * 0.3

        # Normalizar el vector resultante para no exceder la velocidad permitida ('Step')
        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector) * SPEED 

        sentinel.move(move_vector[0], move_vector[1])

    elif sentinel.behavior == SEARCH:
        # [TA] COMPORTAMIENTO ALEATORIO (Fórmula 6 del PDF)
        # X_siguiente = X_i + Visual * R(0,1).  (Aquí R es un ángulo aleatorio)
        angle = np.random.uniform(0, 2 * np.pi)
        move_dist = SPEED * np.random.uniform(0.5, 1.2)
        dx = move_dist * np.cos(angle)
        dy = move_dist * np.sin(angle)
        sentinel.move(dx, dy)

# --- Inicialización ---
sentinels = []
for i in range(NUM_SENTINELS):
    # Distribución inicial aleatoria (Requisito del PDF)
    sentinels.append(Sentinel(i, np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE)))

aerodeslizadores = []
decoy_index = np.random.randint(0, NUM_AERODESI) # Asigna aleatoriamente cuál será el señuelo
for i in range(NUM_AERODESI):
    aerodeslizadores.append(Aerodeslizador(i, np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE), is_decoy=(i == decoy_index)))

print("Simulación iniciada. Centinelas: {}, Aerodeslizadores: {}".format(NUM_SENTINELS, NUM_AERODESI))

# [TA] Función exclusiva para renderizar la gráfica, no afecta la lógica bioinspirada.
def show_snapshot(step, sentinels, aerodeslizadores):
    """
    Crea y muestra una instantánea del estado actual de la simulación.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, AREA_SIZE)
    ax.set_ylim(0, AREA_SIZE)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Instantánea de la Simulación: Paso {}")
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
            # [TA] Muestra el número de centinelas cercanos para validar la condición de neutralización.
            ax.text(aero.x, aero.y + 0.2, str(aero.nearby_sentinels), color='black', fontsize=8, ha='center')
             
            if label not in added_labels:
                legend_handles.append(plt.Line2D([], [], marker=marker, color=color, linestyle='None', markersize=10, label=label))
                added_labels.add(label)

    ax.legend(handles=legend_handles, loc='upper right')
    plt.show()

# --- Bucle de Simulación ---
show_snapshot(0, sentinels, aerodeslizadores)

for step in range(1, MAX_STEPS + 1):
    active_aerodeslizadores = [aero for aero in aerodeslizadores if aero.status == ACTIVE]

    if not active_aerodeslizadores:
        print("\nTodos los aerodeslizadores han sido neutralizados en el paso {}.")
        break

    # [TA] Fase de evaluación de la Función Objetivo (Aptitud de la solución)
    for aero in active_aerodeslizadores:
        aero.update_nearby_sentinels(sentinels)
        # [TA] Condición de éxito: 10 centinelas logran acorralar la presa.
        if aero.nearby_sentinels >= NEUTRALIZATION_THRESHOLD:
            aero.set_status(NEUTRALIZED)
            if not aero.is_decoy:
                print("Paso {}: Aerodeslizador {aero.id} (Real) neutralizado por {aero.nearby_sentinels} centinelas.")
            else:
                # [TA] Si caen en el señuelo, la Matrix perdió tiempo, pero cumple el reto de detectarlo.
                print("Paso {}: Aerodeslizador Señuelo {aero.id} neutralizado por {aero.nearby_sentinels} centinelas.")
                 
    active_aerodeslizadores = [aero for aero in aerodeslizadores if aero.status == ACTIVE]
     
    if not active_aerodeslizadores:
        continue

    # [TA] Fase de actualización de posición (Movimiento del enjambre)
    for sentinel in sentinels:
        update_sentinel_behavior(sentinel, active_aerodeslizadores, sentinels)

    # Instantáneas visuales
    if step % SNAPSHOT_INTERVAL == 0:
        print("--- Generando instantánea en el paso {} ---")
        show_snapshot(step, sentinels, aerodeslizadores)

else: 
    print("\nSe alcanzó el número máximo de pasos ({}). Simulación finalizada.")

print("\n--- Generando instantánea final ---")
show_snapshot(step, sentinels, aerodeslizadores)
print("Simulación finalizada.")