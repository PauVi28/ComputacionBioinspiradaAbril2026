# -*- coding: utf-8 -*-
"""


@author: MARCELOFGB
"""

import random
import matplotlib.pyplot as plt
import pandas as pd # Para una visualización tabular del horario final

# --- 1. DEFINICIÓN DEL PROBLEMA ---

# Días de la semana y horas disponibles
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
HORAS = [9, 10, 11, 12, 14, 15, 16] # Horas de inicio de las clases

# Aulas disponibles: ID y Capacidad
AULAS = {
    "A101": {"capacidad": 30},
    "B201": {"capacidad": 50},
    "C301": {"capacidad": 20},
    "D401": {"capacidad": 40},
    "E101": {"capacidad": 60}
}
ID_AULAS = list(AULAS.keys())

# Clases a programar: ID, Nombre, Cantidad de Estudiantes, Horario Preferido (opcional)
# La posición en esta lista será la "ID" de la clase para el cromosoma.
CLASES = [
    {"nombre": "Introducción a la Programación", "estudiantes": 35, "preferencia_dia": "Lunes", "preferencia_hora": 10},
    {"nombre": "Cálculo I", "estudiantes": 45, "preferencia_dia": "Martes", "preferencia_hora": 9},
    {"nombre": "Algoritmos y Estructuras de Datos", "estudiantes": 25, "preferencia_dia": "Lunes", "preferencia_hora": 14},
    {"nombre": "Bases de Datos", "estudiantes": 30, "preferencia_dia": "Miércoles", "preferencia_hora": 11},
    {"nombre": "Inteligencia Artificial", "estudiantes": 25, "preferencia_dia": "Jueves", "preferencia_hora": 15},
    {"nombre": "Redes de Computadoras", "estudiantes": 50, "preferencia_dia": "Viernes", "preferencia_hora": 9},
    {"nombre": "Sistemas Operativos", "estudiantes": 40, "preferencia_dia": "Martes", "preferencia_hora": 14},
    {"nombre": "Programación Orientada a Objetos", "estudiantes": 30, "preferencia_dia": "Jueves", "preferencia_hora": 10}
]
NUM_CLASES = len(CLASES)

# --- 2. PARÁMETROS DEL ALGORITMO GENÉTICO ---
TAMANO_POBLACION = 700
NUM_GENERACIONES = 50
TASA_MUTACION = 0.5
TASA_CRUCE = 0.5
ELITISMO_PORCENTAJE = 0.05 # Top 5% de la población se transfiere directamente

# Penalizaciones para la función de aptitud (menor aptitud = peor)
PENALIDAD_CONFLICTO_HORA_AULA = 1000 # Muy alta! Esto es un HARD CONSTRAINT
PENALIDAD_CAPACIDAD_INSUFICIENTE = 500 # Alta
PENALIDAD_PREFERENCIA_NO_CUMPLIDA = 50 # Baja (SOFT CONSTRAINT)

# --- 3. FUNCIONES DEL ALGORITMO GENÉTICO ---

def generar_gen_aleatorio():
    """Genera una asignación aleatoria (día, hora, aula_id) para una clase."""
    dia = random.choice(DIAS)
    hora = random.choice(HORAS)
    aula_id = random.choice(ID_AULAS)
    return (dia, hora, aula_id)

def inicializar_poblacion():
    """Crea una población inicial de cromosomas aleatorios."""
    poblacion = []
    for _ in range(TAMANO_POBLACION):
        cromosoma = [generar_gen_aleatorio() for _ in range(NUM_CLASES)]
        poblacion.append(cromosoma)
    return poblacion

def calcular_aptitud(cromosoma):
    """
    Calcula la aptitud de un cromosoma (horario).
    Mayor aptitud es mejor. Penaliza los conflictos.
    """
    aptitud = 0 # Empezamos con 0, sumaremos puntos por un buen horario, restaremos por conflictos

    # Usaremos un mapa para detectar conflictos de tiempo/aula
    # key: (dia, hora, aula_id) -> value: [lista de IDs de clases programadas]
    horario_ocupacion = {}

    # Por cada clase en el cromosoma (horario propuesto)
    for i, (dia, hora, aula_id) in enumerate(cromosoma):
        clase_info = CLASES[i]
        clase_nombre = clase_info["nombre"]
        num_estudiantes = clase_info["estudiantes"]
        aula_capacidad = AULAS[aula_id]["capacidad"]

        # --- Penalización por conflicto de Capacidad ---
        if num_estudiantes > aula_capacidad:
            # Penaliza por cada estudiante extra que excede la capacidad
            aptitud -= PENALIDAD_CAPACIDAD_INSUFICIENTE * (num_estudiantes - aula_capacidad) / 10 # Escalado de penalización

        # --- Penalización por Preferencia de Horario (Soft Constraint) ---
        if "preferencia_dia" in clase_info and "preferencia_hora" in clase_info:
            if dia != clase_info["preferencia_dia"] or hora != clase_info["preferencia_hora"]:
                aptitud -= PENALIDAD_PREFERENCIA_NO_CUMPLIDA

        # --- Registrar ocupación para conflictos de tiempo/aula ---
        identificador_slot = (dia, hora, aula_id)
        if identificador_slot not in horario_ocupacion:
            horario_ocupacion[identificador_slot] = []
        horario_ocupacion[identificador_slot].append(i) # Añadir ID de la clase

    # --- Penalización por conflicto de Aulas/Tiempo ---
    for slot, clases_en_slot in horario_ocupacion.items():
        if len(clases_en_slot) > 1:
            # Si hay más de una clase en el mismo slot de tiempo/aula, hay un conflicto
            # Penaliza por cada clase extra en el mismo slot
            aptitud -= PENALIDAD_CONFLICTO_HORA_AULA * (len(clases_en_slot) - 1)

    return aptitud

def seleccionar_padres(poblacion, aptitudes):
    """
    Selección por Torneo:
    1. Se seleccionan aleatoriamente `k` individuos de la población.
    2. El individuo con la mejor aptitud de ese grupo es el ganador del torneo.
    3. Se repite para seleccionar otro padre.
    """
    K = 5 # Tamaño del torneo
    
    def _seleccionar_un_padre():
        competidores_indices = random.sample(range(TAMANO_POBLACION), K)
        mejor_competidor_idx = -1
        mejor_aptitud_en_torneo = -float('inf')

        for idx in competidores_indices:
            if aptitudes[idx] > mejor_aptitud_en_torneo:
                mejor_aptitud_en_torneo = aptitudes[idx]
                mejor_competidor_idx = idx
        return poblacion[mejor_competidor_idx]

    padre1 = _seleccionar_un_padre()
    padre2 = _seleccionar_un_padre() # Asegurar que sean diferentes? No estrictamente necesario para el GA básico.

    return padre1, padre2

def cruzar(padre1, padre2):
    """
    Cruce en un punto:
    Combina dos cromosomas padres para crear dos hijos.
    Se elige un punto de corte aleatorio y se intercambian los "genes"
    después de ese punto.
    """
    if random.random() < TASA_CRUCE:
        punto_cruce = random.randint(1, NUM_CLASES - 1) # Asegurarse de que el punto no sea al principio o al final
        
        hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
        hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
        return hijo1, hijo2
    else:
        return padre1, padre2 # No hay cruce, los hijos son copias de los padres

def mutar(cromosoma):
    """
    Mutación:
    Cambia aleatoriamente el día, hora o aula de un gen (clase) en el cromosoma.
    """
    for i in range(NUM_CLASES):
        if random.random() < TASA_MUTACION:
            # Reemplazar el gen con uno completamente nuevo y aleatorio
            cromosoma[i] = generar_gen_aleatorio()
    return cromosoma

# --- 4. BUCLE PRINCIPAL DEL ALGORITMO GENÉTICO ---

def ejecutar_algoritmo_genetico():
    poblacion = inicializar_poblacion()
    mejores_aptitudes_historial = []
    mejor_cromosoma_global = None
    mejor_aptitud_global = -float('inf') # Inicializar con valor muy bajo

    for generacion in range(NUM_GENERACIONES):
        aptitudes = [calcular_aptitud(cromosoma) for cromosoma in poblacion]

        # Actualizar el mejor de todos los tiempos
        idx_mejor_actual = aptitudes.index(max(aptitudes))
        aptitud_actual = aptitudes[idx_mejor_actual]

        if aptitud_actual > mejor_aptitud_global:
            mejor_aptitud_global = aptitud_actual
            mejor_cromosoma_global = poblacion[idx_mejor_actual]

        mejores_aptitudes_historial.append(max(aptitudes))

        nueva_poblacion = []

        # Elitismo: Mantener los mejores individuos
        # Ordenar la población por aptitud de mayor a menor y tomar el porcentaje top
        elite_count = int(TAMANO_POBLACION * ELITISMO_PORCENTAJE)
        poblacion_ordenada = sorted(zip(poblacion, aptitudes), key=lambda x: x[1], reverse=True)
        for i in range(elite_count):
            nueva_poblacion.append(poblacion_ordenada[i][0])
        
        # Llenar el resto de la nueva población
        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre1, padre2 = seleccionar_padres(poblacion, aptitudes)
            hijo1, hijo2 = cruzar(padre1, padre2)
            
            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2) # Mutar a ambos hijos

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < TAMANO_POBLACION: # Asegurar que no se exceda el tamaño
                nueva_poblacion.append(hijo2)
        
        poblacion = nueva_poblacion
        
        if generacion % 50 == 0:
            print(f"Generación {generacion}: Mejor Aptitud = {mejor_aptitud_global}")

    print(f"\nAlgoritmo Genético Finalizado después de {NUM_GENERACIONES} generaciones.")
    print(f"Mejor Aptitud Final: {mejor_aptitud_global}")
    
    return mejor_cromosoma_global, mejores_aptitudes_historial, mejor_aptitud_global

# --- 5. VISUALIZACIÓN DE RESULTADOS ---

def mostrar_horario(cromosoma):
    """Muestra el horario de forma tabular y fácil de entender."""
    print("\n--- MEJOR HORARIO ENCONTRADO ---")
    
    # Crear un DataFrame para una mejor visualización tipo tabla
    horario_df = pd.DataFrame(columns=["Clase", "Día", "Hora", "Aula", "Estudiantes", "Capacidad Aula", "¿Conflicto Capacidad?"])

    horario_ocupacion_check = {} # para verificar conflictos en la salida final
    conflictos_aula_tiempo = 0
    conflictos_capacidad_total = 0

    for i, (dia, hora, aula_id) in enumerate(cromosoma):
        clase_info = CLASES[i]
        clase_nombre = clase_info["nombre"]
        num_estudiantes = clase_info["estudiantes"]
        aula_capacidad = AULAS[aula_id]["capacidad"]

        conflicto_capacidad = "❌" if num_estudiantes > aula_capacidad else "✅"
        if num_estudiantes > aula_capacidad:
            conflictos_capacidad_total += 1

        identificador_slot = (dia, hora, aula_id)
        if identificador_slot not in horario_ocupacion_check:
            horario_ocupacion_check[identificador_slot] = []
        horario_ocupacion_check[identificador_slot].append(clase_nombre)

        horario_df.loc[i] = [clase_nombre, dia, hora, aula_id, num_estudiantes, aula_capacidad, conflicto_capacidad]
    
    print(horario_df.to_string(index=False))

    print("\n--- Resumen de Conflictos en el Mejor Horario ---")
    
    for slot, clases_en_slot in horario_ocupacion_check.items():
        if len(clases_en_slot) > 1:
            conflictos_aula_tiempo += 1
            print(f"⚠️ CONFLICTO EN {slot[0]} {slot[1]}h en {slot[2]}: Clases {', '.join(clases_en_slot)}")
    
    if conflictos_aula_tiempo == 0:
        print("✅ No hay conflictos de aula/tiempo (mismo aula a la misma hora).")
    else:
        print(f"❌ Total conflictos de aula/tiempo: {conflictos_aula_tiempo}")

    if conflictos_capacidad_total == 0:
        print("✅ No hay conflictos de capacidad (todas las clases en aulas apropiadas).")
    else:
        print(f"❌ Total conflictos de capacidad: {conflictos_capacidad_total}")
    
    aptitud_final_recalculada = calcular_aptitud(cromosoma)
    print(f"\nAptitud Recalculada del Mejor Horario: {aptitud_final_recalculada}")
    print("---------------------------------------------")


def graficar_rendimiento(mejores_aptitudes_historial):
    """Genera un gráfico de la mejor aptitud a lo largo de las generaciones."""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(mejores_aptitudes_historial)), mejores_aptitudes_historial, marker='o', markersize=3, linestyle='-', color='skyblue')
    plt.title('Evolución de la Mejor Aptitud por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.grid(True)
    plt.show()

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("Iniciando Algoritmo Genético para Optimización de Horarios...\n")
    
    mejor_cromosoma_final, aptitudes_historial, mejor_aptitud_final = ejecutar_algoritmo_genetico()
    
    graficar_rendimiento(aptitudes_historial)
    mostrar_horario(mejor_cromosoma_final)

    # Nota: Si la aptitud es un número grande negativo, significa que hay muchos conflictos iniciales.
    # El GA intentará empujar este número hacia 0 o positivo si logra resolver todos los conflictos críticos.
    # Un horario con CERO conflictos de aula/tiempo y capacidad debería tener una aptitud relativamente alta
