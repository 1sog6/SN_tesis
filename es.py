"""
from SNEWS2/sntools

nu + e- -> nu + e-

Appendices of Bahcall et al. 1995 (https://doi.org/10.1103/PhysRevD.51.6146).
Includes radiative corrections from QCD & QED effects.
"""

from math import pi, sqrt, log
from scipy import integrate
from snewpy.neutrino import Flavor

sin2theta_w = 0.2317  # weak mixing angle
alpha = 1 / 137.036  # fine structure constant
mE = 0.5109989  # electron mass (MeV)
gF = 1.16637e-11  # Fermi coupling constant
rho_NC = 1.0126  # numerical factor from Bahcall et al.

cherenkov_threshold = 0.77  # Cherenkov threshold of electron/positron in water

_cache = {}  # save time by avoiding repeat calculations
def spence(n):
    n = round(n, 6)  # negligible decrease in accuracy to save time & memory
    if n not in _cache:
        _cache[n] = integrate.quad(lambda t: log(abs(1 - t)) / t, 0, n)[0]
    return _cache[n]

class Channel:
    def __init__(self, flavor):
        if flavor not in list(Flavor):
            raise ValueError(f"Flavor {flavor} cannot interact in ES channel.")
        self.flavor = flavor
        self.eE_min = cherenkov_threshold

    # Se omiten funciones innecesarias

    def dSigma_dE(self, eNu, eE):
        """Return differential cross section in MeV^-2.

        Inputs:
            eNu: neutrino energy
            eE:  energy of outgoing (detected) particle
        """
        if eE < self.bounds_eE(eNu)[0] or eE > self.bounds_eE(eNu)[1]:
            return 0

        # Appendix A: Radiative Corrections
        L = sqrt(eE**2 - mE**2)           
        beta = L / eE
        T = eE - mE  # kinetic energy of recoil electron          
        z = T / eNu
        x = sqrt(1 + 2 * mE / T)
        i = 1 / 6 * (1 / 3 + (3 - x**2) * (x / 2 * log((x + 1) / (x - 1)) - 1))

        if self.flavor in (Flavor.NU_E, Flavor.NU_E_BAR):
            k = 0.9791 + 0.0097 * i
        elif self.flavor in (Flavor.NU_X, Flavor.NU_X_BAR):
            k = 0.9970 - 0.00037 * i

        g1 = rho_NC * (0.5 - k * sin2theta_w)
        g2 = -rho_NC * k * sin2theta_w

        if self.flavor == Flavor.NU_E:
            gL = g1 - 1
            gR = g2
        elif self.flavor == Flavor.NU_E_BAR:
            gL = g2
            gR = g1 - 1
        elif self.flavor == Flavor.NU_X:
            gL = g1
            gR = g2
        elif self.flavor == Flavor.NU_X_BAR:
            gL = g2
            gR = g1

        # Appendix B: QED Effects
        f0 = eE / L * log((eE + L) / mE) - 1  # common factor of all three f_*
        log_zmE = log(1 - z - mE / (eE + L))  # Warning: imprecise at low E, throws ValueError in extreme cases
        log1z = log(1 - z)

        # fMinus(z)
        f1 = f0 * (2 * log_zmE - log1z - log(z) / 2 - 5 / 12) \
            + 0.5 * (spence(z) - spence(beta)) \
            - 0.5 * log1z**2 - (11 / 12 + z / 2) * log1z \
            + z * (log(z) + 0.5 * log(2 * eNu / mE)) \
            - (31 / 18 + 1 / 12 * log(z)) * beta \
            - 11 / 12 * z + z**2 / 24

        # (1 - z)**2 * fPlus(z)
        f2 = f0 * ((1 - z)**2 * (2 * log_zmE - log1z - log(z) / 2 - 2 / 3) - (z**2 * log(z) + 1 - z) / 2) \
            - (1 - z)**2 / 2 * (log1z**2 + beta * (spence(1 - z) - log(z) * log1z)) \
            + log1z * (z**2 / 2 * log(z) + (1 - z) / 3 * (2 * z - 0.5)) \
            - z**2 / 2 * spence(1 - z) - z * (1 - 2 * z) / 3 * log(z) - z * (1 - z) / 6 \
            - beta / 12 * (log(z) + (1 - z) * (115 - 109 * z) / 6)

        # fPlusMinus(z)
        f3 = f0 * 2 * log_zmE

        result = 2 * mE * gF**2 / pi * (gL**2 * (1 + alpha / pi * f1)
                                        + gR**2 * ((1 - z)**2 + alpha / pi * f2)
                                        - gR * gL * mE / eNu * z * (1 + alpha / pi * f3))
        if result < 0:
            if eNu < cherenkov_threshold:
                # Approximations in f_* may be imprecise at very low energies.
                # This is below threshold in HK anyway, so we suppress it.
                result = 0
            else:
                raise ValueError(f"Calculated negative cross section for E_nu={eNu}, E_e={eE}. Aborting...")

        return result
        
    def bounds_eE(self, eNu, *args): # ignore additional arguments handed over by integrate.nquad()
        """Return kinematic bounds for integration over eE.

        Input:
            eNu:  neutrino energy (in MeV)
            args: [ignore this]
        Output:
            list with minimum & maximum allowed energy of outgoing (detected) particle
        """
        eE_max = mE + 2 * eNu**2 / (2 * eNu + mE)  
        return [self.eE_min, eE_max]

