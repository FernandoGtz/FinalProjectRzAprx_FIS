from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =========================================================
# CONTRATOS DE DATOS
# =========================================================
class EjercicioHistorico(BaseModel):
    id_ejercicio: str
    peso: float
    reps: int


class EjercicioActual(BaseModel):
    id_ejercicio: str
    peso: float
    reps: int
    rpe: float
    ct: float
    fa: float


class EjercicioRutina(BaseModel):
    id_ejercicio: str
    nombre: str
    musculo: str
    tier: int


class PayloadEvaluacion(BaseModel):
    perfil: str
    actual: List[EjercicioActual]
    historico: List[EjercicioHistorico]
    rutina: List[EjercicioRutina]


# =========================================================
# ESTADO GLOBAL
# =========================================================
configuraciones = {}
motores_fis_estaticos = {}

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def calcular_1rm(peso, reps):
    if reps <= 1:
        return peso
    return peso * (1 + 0.033 * reps)


def procesar_datos(registros, config, rutina):
    datos_procesados = []

    historico_dict = {
        reg['id_ejercicio']: reg
        for reg in registros['historico']
    }

    rutina_dict = {
        ej['id_ejercicio']: ej
        for ej in rutina
    }

    for reg_actual in registros['actual']:

        id_ej = reg_actual['id_ejercicio']

        if id_ej not in historico_dict:
            raise HTTPException(
                status_code=422,
                detail=f"No existe histórico para el ejercicio {id_ej}"
            )

        if id_ej not in rutina_dict:
            raise HTTPException(
                status_code=422,
                detail=f"No existe configuración de rutina para el ejercicio {id_ej}"
            )

        reg_historico = historico_dict[id_ej]
        rutina_info = rutina_dict[id_ej]

        rm_actual = calcular_1rm(reg_actual['peso'], reg_actual['reps'])

        rm_historico = calcular_1rm(reg_historico['peso'], reg_historico['reps'])

        if rm_historico == 0:
            raise HTTPException(status_code=422, detail=f"1RM histórico inválido para {id_ej}")

        delta_1rm = ((rm_actual - rm_historico) / rm_historico) * 100

        limite_min = config.get('limite_min', -15)
        limite_max = config.get('limite_max', 15)

        delta_1rm = float(np.clip(delta_1rm, limite_min, limite_max))

        datos_procesados.append({
            'id_ejercicio': id_ej,
            'nombre': rutina_info['nombre'],
            'musculo': rutina_info['musculo'],
            'tier': rutina_info['tier'],
            'delta_1rm': delta_1rm,
            'rpe': reg_actual['rpe'],
            'ct': reg_actual['ct'],
            'fa': reg_actual['fa'],
            'rm_actual': rm_actual,
            'rm_historico': rm_historico
        })

    return datos_procesados


# =========================================================
# CONSTRUCCION FIS
# =========================================================
def construir_fis(config):
    # Definición de Rangos de las Variables (Universo Discurso)
    # -- FIS 1 --
    # Antecedentes
    delta_1rm = ctrl.Antecedent(np.arange(-15, 16, 1), 'delta_1rm')
    rpe = ctrl.Antecedent(np.arange(0, 11, 1), 'rpe')
    # Consecuente
    rm = ctrl.Consequent(np.arange(0, 101, 1), 'rm')

    # -- FIS 2 --
    # Antecedentes
    ct = ctrl.Antecedent(np.arange(0.0, 5.5, 0.5), 'ct')
    fa = ctrl.Antecedent(np.arange(0, 11, 1), 'fa')
    # Consecuente
    ce = ctrl.Consequent(np.arange(0, 101, 1), 'ce')

    # -- FIS 3 --
    # Antecedentes
    rm_fis3 = ctrl.Antecedent(np.arange(0, 101, 1), 'rm_fis3')
    ce_fis3 = ctrl.Antecedent(np.arange(0, 101, 1), 'ce_fis3')
    # Consecuente
    ics = ctrl.Consequent(np.arange(0, 101, 1), 'ics')

    # Asignación de Funciones de Pertenencia
    # FIS 1 - Delta 1RM
    delta_1rm['Retroceso'] = fuzz.trapmf(delta_1rm.universe, [-15, -15, -3, 0])
    delta_1rm['Mantenimiento'] = fuzz.trimf(delta_1rm.universe, [-3, 0, 3])
    delta_1rm['Progreso'] = fuzz.trapmf(delta_1rm.universe, [0, 3, 15, 15])

    # FIS 1 - RPE
    rpe['Suboptimo'] = fuzz.gaussmf(rpe.universe, 3, 1.5)
    rpe['Optimo'] = fuzz.gaussmf(rpe.universe, 8, 1.5)
    rpe['Limite'] = fuzz.gaussmf(rpe.universe, 10, 1.0)

    # FIS 1 - RM
    rm['Deficiente'] = fuzz.trapmf(rm.universe, [0, 0, 20, 40])
    rm['Aceptable'] = fuzz.trimf(rm.universe, [30, 50, 70])
    rm['Sobresaliente'] = fuzz.trapmf(rm.universe, [60, 80, 100, 100])

    # FIS 2 - CT (Capacidad Tecnica)
    ct['Comprometida'] = fuzz.trapmf(ct.universe, [0.5, 0.5, 1.5, 2.5])
    ct['Aceptable'] = fuzz.trimf(ct.universe, [2, 3.5, 4.5])
    ct['Impecable'] = fuzz.trapmf(ct.universe, [4, 4.75, 5.5, 5.5])

    # FIS 2 - FA (Fatiga)
    fa['Baja'] = fuzz.trapmf(fa.universe, [0, 0, 2, 5])
    fa['Manejable'] = fuzz.trimf(fa.universe, [3, 6, 9])
    fa['Critica'] = fuzz.trapmf(fa.universe, [7, 9, 11, 11])

    # FIS 2 - CE (Calidad de Ejecucion)
    ce['Deficiente'] = fuzz.trapmf(ce.universe, [0, 0, 20, 40])
    ce['Estandar'] = fuzz.trimf(ce.universe, [30, 50, 70])
    ce['Optima'] = fuzz.trapmf(ce.universe, [60, 80, 100, 100])

    # FIS 3 - RM (Rendimiento Muscular)
    rm_fis3['Deficiente'] = fuzz.trapmf(rm_fis3.universe, [0, 0, 20, 40])
    rm_fis3['Aceptable'] = fuzz.trimf(rm_fis3.universe, [30, 50, 70])
    rm_fis3['Sobresaliente'] = fuzz.trapmf(rm_fis3.universe, [60, 80, 100, 100])

    # FIS 3 - CE (Calidad de Ejecucion)
    ce_fis3['Deficiente'] = fuzz.trapmf(ce_fis3.universe, [0, 0, 20, 40])
    ce_fis3['Estandar'] = fuzz.trimf(ce_fis3.universe, [30, 50, 70])
    ce_fis3['Optima'] = fuzz.trapmf(ce_fis3.universe, [60, 80, 100, 100])

    # FIS 3 - ICS (Indice de Calidad de Sesion)
    ics['Pobre'] = fuzz.trapmf(ics.universe, [0, 0, 20, 40])
    ics['Productivo'] = fuzz.trimf(ics.universe, [30, 50, 70])
    ics['AltoRendimiento'] = fuzz.trapmf(ics.universe, [60, 80, 100, 100])

    # =====================================================
    # REGLAS
    # =====================================================
    # Generacion de Reglas
    reglas_rm_obj = []
    for regla in config['matriz_reglas_rm']:
        delta_term = delta_1rm[regla['delta_1rm']]
        rpe_term = rpe[regla['rpe']]
        rm_term = rm[regla['rm']]
        regla_obj = ctrl.Rule(delta_term & rpe_term, rm_term)
        reglas_rm_obj.append(regla_obj)

    reglas_ce_obj = []
    for regla in config['matriz_reglas_ce']:
        ct_term = ct[regla['ct']]
        fa_term = fa[regla['fa']]
        ce_term = ce[regla['ce']]
        regla_obj = ctrl.Rule(ct_term & fa_term, ce_term)
        reglas_ce_obj.append(regla_obj)

    reglas_final_obj = []
    for regla in config['matriz_reglas_final']:
        rm_term = rm_fis3[regla['rm']]
        ce_term = ce_fis3[regla['ce']]
        ics_term = ics[regla['ics']]
        regla_obj = ctrl.Rule(rm_term & ce_term, ics_term)
        reglas_final_obj.append(regla_obj)

    # =====================================================
    # CONTROLADORES
    # =====================================================
    rm_ctrl = ctrl.ControlSystem(reglas_rm_obj)
    ce_ctrl = ctrl.ControlSystem(reglas_ce_obj)
    ics_ctrl = ctrl.ControlSystem(reglas_final_obj)

    return {
        'rm_ctrl': rm_ctrl,
        'ce_ctrl': ce_ctrl,
        'ics_ctrl': ics_ctrl
    }


# =========================================================
# CICLO DE VIDA
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):

    global configuraciones
    global motores_fis_estaticos

    # ============================================
    # CARGAR CONFIGURACIONES
    # ============================================
    configuraciones['hipertrofia'] = json.load(
        open('config_perfil_hipertrofia.json')
    )

    configuraciones['recuperacion'] = json.load(
        open('config_perfil_recuperacion.json')
    )

    # ============================================
    # PRECOMPILAR CONTROLADORES FIS
    # ============================================
    motores_fis_estaticos['hipertrofia'] = construir_fis(
        configuraciones['hipertrofia']
    )

    motores_fis_estaticos['recuperacion'] = construir_fis(
        configuraciones['recuperacion']
    )

    print("FIS precompilados correctamente")

    yield

    print("Apagando servidor...")


app = FastAPI(
    title="FIS BMG API",
    lifespan=lifespan
)


# =========================================================
# ENDPOINT
# =========================================================
@app.post("/api/evaluar")
def evaluar_sesion(payload: PayloadEvaluacion):

    perfil_solicitado = payload.perfil.lower()

    if perfil_solicitado not in configuraciones:
        raise HTTPException(
            status_code=400,
            detail=f"Perfil '{perfil_solicitado}' no configurado"
        )

    config_activa = configuraciones[perfil_solicitado]

    registros = {
        'historico': [
            item.model_dump()
            for item in payload.historico
        ],
        'actual': [
            item.model_dump()
            for item in payload.actual
        ]
    }

    rutina = [
        item.model_dump()
        for item in payload.rutina
    ]

    datos_procesados = procesar_datos(
        registros,
        config_activa,
        rutina
    )

    rm_sim = ctrl.ControlSystemSimulation(
        motores_fis_estaticos[perfil_solicitado]['rm_ctrl']
    )

    ce_sim = ctrl.ControlSystemSimulation(
        motores_fis_estaticos[perfil_solicitado]['ce_ctrl']
    )

    ics_sim = ctrl.ControlSystemSimulation(
        motores_fis_estaticos[perfil_solicitado]['ics_ctrl']
    )

    resultados = []
    resultados_por_musculo = {}

    for dato in datos_procesados:
        try:
            # =============================================
            # FIS 1
            # =============================================
            rm_sim.input['delta_1rm'] = np.clip(dato['delta_1rm'],-15,15)
            rm_sim.input['rpe'] = np.clip(dato['rpe'],0,10)
            rm_sim.compute()
            rm_output = float(rm_sim.output['rm'])

            # =============================================
            # FIS 2
            # =============================================
            ce_sim.input['ct'] = np.clip(dato['ct'],0.5,5.5)
            ce_sim.input['fa'] = np.clip(dato['fa'],0,10)
            ce_sim.compute()
            ce_output = float(ce_sim.output['ce'])

            # =============================================
            # FIS 3
            # =============================================
            ics_sim.input['rm_fis3'] = np.clip(rm_output,0,100)
            ics_sim.input['ce_fis3'] = np.clip(ce_output,0,100)
            ics_sim.compute()
            ics_output = float(ics_sim.output['ics'])

        except Exception as e:

            raise HTTPException(
                status_code=422,
                detail=f"Error en inferencia difusa para {dato['id_ejercicio']}: {str(e)}"
            )

        tier = dato['tier']

        ponderacion = float(config_activa['ponderaciones_tier'][str(tier)])

        resultado_ejercicio = {
            'id_ejercicio': dato['id_ejercicio'],
            'nombre': dato['nombre'],
            'musculo': dato['musculo'],
            'tier': tier,
            'delta_1rm': round(dato['delta_1rm'], 2),
            'rpe': dato['rpe'],
            'ct': dato['ct'],
            'fa': dato['fa'],
            'rm_score': round(rm_output, 2),
            'ce_score': round(ce_output, 2),
            'ics_score': round(ics_output, 2),
            'ponderacion': ponderacion,
            'ics_ponderado': round(ics_output * ponderacion,2)
        }

        resultados.append(resultado_ejercicio)

        musculo = dato['musculo']

        if musculo not in resultados_por_musculo:
            resultados_por_musculo[musculo] = {
                'ejercicios': [],
                'ics_suma_ponderada': 0,
                'ponderacion_total': 0
            }

        resultados_por_musculo[musculo]['ejercicios'].append(resultado_ejercicio)
        resultados_por_musculo[musculo]['ics_suma_ponderada'] += ics_output * ponderacion
        resultados_por_musculo[musculo]['ponderacion_total'] += ponderacion

    # =====================================================
    # CALIFICACION POR MUSCULO
    # =====================================================
    ics_por_musculo = {}

    for musculo, datos in resultados_por_musculo.items():
        if datos['ponderacion_total'] > 0:
            ics_por_musculo[musculo] = round(datos['ics_suma_ponderada'] / datos['ponderacion_total'],2)

    # =====================================================
    # CALIFICACION FINAL
    # =====================================================
    ics_suma_total = sum(
        r['ics_ponderado']
        for r in resultados
    )

    ponderacion_suma_total = sum(
        r['ponderacion']
        for r in resultados
    )

    calificacion_final = round(
        (
            ics_suma_total /
            ponderacion_suma_total
        ),
        2
    ) if ponderacion_suma_total > 0 else 0

    # =====================================================
    # RESPONSE
    # =====================================================
    return {
        "perfil": perfil_solicitado,
        "calificacion_final": calificacion_final,
        "calificaciones_por_musculo": ics_por_musculo,
        "detalle_ejercicios": resultados
    }