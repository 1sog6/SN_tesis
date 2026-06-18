import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snewpy.models.ccsn as models
import re

from snewpy.neutrino import Flavor
from snewpy.rate_calculator import RateCalculator

from scipy.integrate import quad

from src import ibd
from src import es

# Lista modelos de interés
modelos_obj = {
    'Nakazato_2013': models.Nakazato_2013,
    'Sukhbold_2015': models.Sukhbold_2015,
    'Tamborra_2014': models.Tamborra_2014, 
    'Bollig_2016' : models.Bollig_2016,
    # 'Fischer_2020' : models.Fischer_2020,
    'OConnor_2015' : models.OConnor_2015,
}


"""
CONSTANTES
"""

# Variables globales
estado_actual = {'model_family': None, 'model_params': None}
modelo_actual = None
t_actual = None
espectro_actual = None
figura_actual = None

# energiatest = np.arange(0.1, 60.5, 0.5) * u.MeV 
energiatest = np.linspace(0.1, 60, 200) * u.MeV

factor_conversion = 3.894e-22 # MeV^-2 to cm^-2, util para DUNE

# Numero de blancos en cada detector
targets_hyperk_ibd = 1.2567e34 # Protones
targets_hyperk_nes = 6.28e+34 # Electrones

targets_juno_ibd = 1.432e+33 # Protones
targets_juno_nes = 6.76e+33 # Electrones

targets_dune_argon = 6.02e+32 # Argon 
targets_dune_nes = 1.08e+34 # Electrones

rc = RateCalculator()


"""
SECCIONES TRANSVERSALES
"""
# NEUTRINO-ELECTRON SCATTERING
xsec_es = {flavor: None for flavor in Flavor}

for flavor in Flavor:
    canal_es = es.Channel(flavor)
    aux_es = []
    for E_nu in energiatest.value:
        eE_min, eE_max = canal_es.bounds_eE(E_nu)
        sigma = quad(lambda eE: canal_es.dSigma_dE(E_nu, eE) * factor_conversion, eE_min, eE_max)[0]
        aux_es.append(sigma)

    xsec_es[flavor] = np.array(aux_es) * u.cm**2


# INVERSE BETA DECAY
canal_ibd = ibd.Channel(Flavor.NU_E_BAR)

aux_ibd = []

for E_nu in energiatest.value:
    # Límites 
    eE_min, eE_max = canal_ibd.bounds_eE(E_nu)
    
    # Se integra la sección (dSigma/dE) 
    # quad devuelve una tupla (data, error)
    sigma_total = quad(lambda eE: canal_ibd.dSigma_dE(E_nu, eE) * factor_conversion, 
                       eE_min, eE_max)[0]
    
    # Se extraen los valores de quad[0] y se almacenan en una lista
    aux_ibd.append(sigma_total) # 

xsec_ibd = np.array(aux_ibd) * u.cm**2

# ARGÓN
xsec_arnue = rc.load_xsec('nue_Ar40', Flavor.NU_E)(energiatest)
xsec_arnuebar = rc.load_xsec('nuebar_Ar40', Flavor.NU_E_BAR)(energiatest)


"""
DICCIONARIOS
"""
catalogo_modelos = {
    nombre: clase.get_param_combinations()
    for nombre, clase in modelos_obj.items()
}

opciones_widget = {}
for nombre_modelo, lista_combos in catalogo_modelos.items():
    opciones_widget[nombre_modelo] = {
        ", ".join([f"{k}={v}" for k, v in combo.items()]): combo 
        for combo in lista_combos
    }

canales_deteccion = {
    # CANALES IBD
    'Hyper-K | Canal IBD ': {'targets': targets_hyperk_ibd, 'xsec': xsec_ibd, 'flavor': Flavor.NU_E_BAR, 'tipo': 'cc'},
    'JUNO | Canal IBD': {'targets': targets_juno_ibd, 'xsec': xsec_ibd, 'flavor': Flavor.NU_E_BAR, 'tipo': 'cc'},
    
    # CANALES EN ARGÓN (CC)
    'DUNE | Canal \u03BD\u2091 CC': {'targets': targets_dune_argon, 'xsec': xsec_arnue, 'flavor': Flavor.NU_E, 'tipo': 'cc'},
    'DUNE | Canal \u03BD\u0304\u2091 CC': {'targets': targets_dune_argon, 'xsec': xsec_arnuebar, 'flavor': Flavor.NU_E_BAR, 'tipo': 'cc'},
    
    # ELECTRON SCATTERING
    'Hyper-K | Canal ES': {'targets': targets_hyperk_nes, 'xsec': xsec_es, 'tipo': 'nc' },
    'JUNO | Canal ES': {'targets': targets_juno_nes, 'xsec': xsec_es, 'tipo': 'nc'},
    'DUNE | Canal ES': {'targets': targets_dune_nes, 'xsec': xsec_es, 'tipo': 'nc'}
}


diccionario_PH = {
    'Adiabática (PH = 0)':     0.0,
    'Mixta (PH = 0.5)':        0.5,
    'No Adiabática (PH = 1)':  1.0,
}
   

"""
ALGUNAS FUNCIONES
"""

def mostrar_opciones():
    """
    Imprime el catálogo de modelos y la combinación de parámetros disponibles.
    """
    print(f"{'MODELO':<20} | {'COMBINACIONES DISPONIBLES'}")
    print("-" * 60)
    for nombre, combos in catalogo_modelos.items():
        print(f"{nombre:<20} | {len(combos)} combinaciones encontradas.")

def listar_combinaciones(nombre_modelo):
    """
    Muestra todas las opciones para un modelo específico.
    """
    combos = catalogo_modelos.get(nombre_modelo)
    if combos:
        df = pd.DataFrame(combos)
        print(f"Opciones disponibles para {nombre_modelo}:")
        display(df)
    else:
        print("Modelo no encontrado.")

def cargar_modelo(nombre_modelo, indice_comb):
    """
    Instancia un modelo de snewpy usando el diccionario de combinaciones.
    """
    clase_modelo = modelos_obj[nombre_modelo]
    parametros = catalogo_modelos[nombre_modelo][indice_comb]

    return clase_modelo(**parametros)


def calcular_probabilidades(P_H, jerarquia):
    """
    Calcula p y p_bar usando las fórmulas de Dighe-Smirnov con P_L = 0.

    NMO:
        p     = |Ue2|² * P_H + |Ue3|² * (1 - P_H)
        p_bar = |Ue1|²
    IMO:
        p     = |Ue2|²
        p_bar = |Ue1|² * P_H + |Ue3|² * (1 - P_H)
    """
    angulos = {
        'Normal (NMO)':    (33.76, 8.62, 43.29),
        'Invertida (IMO)': (33.76, 8.65, 47.90),
    }
    
    t12, t13, t23 = [np.deg2rad(a) for a in angulos[jerarquia]]

    Ue1_sq = np.cos(t12)**2 * np.cos(t13)**2
    Ue2_sq = np.sin(t12)**2 * np.cos(t13)**2
    Ue3_sq = np.sin(t13)**2

    if jerarquia == 'Normal (NMO)':
        p     = Ue2_sq * P_H + Ue3_sq * (1.0 - P_H)
        p_bar = Ue1_sq

    elif jerarquia == 'Invertida (IMO)':
        p     = Ue2_sq
        p_bar = Ue1_sq * P_H + Ue3_sq * (1.0 - P_H)

    return p, p_bar


def calcular_flujo_tierra(P_H, espectro_aux, distancia_kpc): # Se podría añadir un slider que controle la distancia a la supernova
    """
    Calcula los flujos en la Tierra (flujo_nmo, flujo_imo) a partir de una probabilidad ph
    """
    distancia = (distancia_kpc*u.kpc).to(u.cm)
    factor_T = 1/(4*np.pi*distancia**2)


    psurvival_nmo, psurvival_bar_nmo = calcular_probabilidades(P_H, 'Normal (NMO)')
    psurvival_imo, psurvival_bar_imo = calcular_probabilidades(P_H, 'Invertida (IMO)')

    # NMO
    flujo_tierra_nmo = {sabor: None for sabor in Flavor}
    
    # Flujo en Tierra (tiempo x energía)
    flujo_tierra_nmo[Flavor.NU_E] = (psurvival_nmo*espectro_aux[Flavor.NU_E] + (1-psurvival_nmo)*espectro_aux[Flavor.NU_X]) * factor_T
    flujo_tierra_nmo[Flavor.NU_X] = (1/2)*((1-psurvival_nmo)*espectro_aux[Flavor.NU_E] + (1+psurvival_nmo)*espectro_aux[Flavor.NU_X]) * factor_T
    flujo_tierra_nmo[Flavor.NU_E_BAR] = (psurvival_bar_nmo*espectro_aux[Flavor.NU_E_BAR] + (1-psurvival_bar_nmo)*espectro_aux[Flavor.NU_X_BAR]) * factor_T
    flujo_tierra_nmo[Flavor.NU_X_BAR] = (1/2)*((1-psurvival_bar_nmo)*espectro_aux[Flavor.NU_E_BAR] + (1+psurvival_bar_nmo)*espectro_aux[Flavor.NU_X_BAR]) * factor_T

    # IMO
    flujo_tierra_imo = {sabor: None for sabor in Flavor}
    
    # Flujo en Tierra (tiempo x energía)
    flujo_tierra_imo[Flavor.NU_E] = (psurvival_imo*espectro_aux[Flavor.NU_E] + (1-psurvival_imo)*espectro_aux[Flavor.NU_X]) * factor_T
    flujo_tierra_imo[Flavor.NU_X] = (1/2)*((1-psurvival_imo)*espectro_aux[Flavor.NU_E] + (1+psurvival_imo)*espectro_aux[Flavor.NU_X]) * factor_T
    flujo_tierra_imo[Flavor.NU_E_BAR] = (psurvival_bar_imo*espectro_aux[Flavor.NU_E_BAR] + (1-psurvival_bar_imo)*espectro_aux[Flavor.NU_X_BAR]) * factor_T
    flujo_tierra_imo[Flavor.NU_X_BAR] = (1/2)*((1-psurvival_bar_imo)*espectro_aux[Flavor.NU_E_BAR] + (1+psurvival_bar_imo)*espectro_aux[Flavor.NU_X_BAR]) * factor_T


    return flujo_tierra_nmo, flujo_tierra_imo


def eventos_panel_interactivo(model_family, model_params, nombre_transformacion, nombre_canal, distancia_kpc):
    global modelo_actual, t_actual, espectro_actual, estado_actual
    
    if estado_actual['model_family'] != model_family or estado_actual['model_params'] != model_params:
        clase_modelo = modelos_obj[model_family]
        parametros = opciones_widget[model_family][model_params]
        modelo_actual = clase_modelo(**parametros)
        
        t_actual = modelo_actual.time
        espectro_aux = modelo_actual.get_initial_spectra(t_actual, energiatest)
        espectro_actual = {flavor: data.to(1/(u.MeV*u.s)) for flavor, data in espectro_aux.items()}
        
        estado_actual['model_family'] = model_family
        estado_actual['model_params'] = model_params
        print(f"Modelo Cargado: {model_family} | {model_params}")
        
    P_H = diccionario_PH[nombre_transformacion]
    flujo_nmo, flujo_imo = calcular_flujo_tierra(P_H, espectro_actual, distancia_kpc)

    canal = canales_deteccion[nombre_canal]
    targets = canal['targets']

    if canal['tipo'] == 'cc':
        flavor = canal['flavor']
        d2NdEdT_nmo = targets * flujo_nmo[flavor] * canal['xsec']
        d2NdEdT_imo = targets * flujo_imo[flavor] * canal['xsec']
        
    elif canal['tipo'] == 'nc':
        d2NdEdT_nmo = sum(targets * flujo_nmo[flavor] * canal['xsec'][flavor] for flavor in Flavor)
        d2NdEdT_imo = sum(targets * flujo_imo[flavor] * canal['xsec'][flavor] for flavor in Flavor)

    # Integrales
    dNdE_nmo = np.trapezoid(d2NdEdT_nmo, x=t_actual, axis=0)
    dNdE_imo = np.trapezoid(d2NdEdT_imo, x=t_actual, axis=0)
    
    dNdT_nmo = np.trapezoid(d2NdEdT_nmo, x=energiatest, axis=1)
    dNdT_imo = np.trapezoid(d2NdEdT_imo, x=energiatest, axis=1)

    total_nmo = np.trapezoid(dNdE_nmo, x=energiatest)
    total_imo = np.trapezoid(dNdE_imo, x=energiatest)

    # Gráficas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=120)
    
    ax1.plot(t_actual, dNdT_nmo, lw=2.5, label=f'NMO (Total: {total_nmo:.0f})')
    ax1.plot(t_actual, dNdT_imo, lw=2.5, linestyle='--', label=f'IMO (Total: {total_imo:.0f})')
    ax1.set_xlabel(r'$t_{pb}$ [s]', fontsize=12)
    ax1.set_ylabel(r'$\frac{dN}{dt}$ [' + str(dNdT_imo.unit) + ']', fontsize=12)
    ax1.set_xlim(-0.05, 0.56) 
    # ax1.set_xscale('log')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11)

    ax2.plot(energiatest, dNdE_nmo, lw=2.5, label=f'NMO (Total: {total_nmo:.0f})')
    ax2.plot(energiatest, dNdE_imo, lw=2.5, linestyle='--', label=f'IMO (Total: {total_imo:.0f})')
    ax2.set_xlabel(r'$E_\nu$ [' + str(energiatest.unit) + ']', fontsize=12)
    ax2.set_ylabel(r'$\frac{dN}{dE_\nu}$ [' + str(dNdE_imo.unit) + ']', fontsize=12)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=11)

    fig.suptitle(f'{nombre_canal} | Transición: {nombre_transformacion} | d = {distancia_kpc:.2f} kpc \n {model_family}', fontsize=14, weight='bold')
    plt.tight_layout()

    global figura_actual
    figura_actual = fig
    
    plt.show()

def guardar_figura_actual(fig_familia, fig_params, fig_canal, fig_transicion): # hace falta añadir la distancia en el nombre
    """
    Toma la figura actual y la guarda segun los parametros 
    """
    global figura_actual

    # Se da forma al nombre del archivo
    if figura_actual is not None:
        canal = canal_crudo.split('|')[0].strip()
        transicion = transicion_cruda.split('(')[0].strip()
        
        nombre_crudo = f"{familia}_{params}_{canal}_{transicion}"
        nombre_limpio = re.sub(r'[^a-zA-Z0-9]', '_', nombre_crudo)
        nombre_final = re.sub(r'_+', '_', nombre_limpio) + '.png'
        
        figura_actual.savefig('eventos/' + nombre_final, bbox_inches='tight', dpi=300)
        
        return f"Imagen guardada -> {nombre_final}"
    else:
        return "Error: Aún no hay gráfica en memoria para guardar."

print('Listo')