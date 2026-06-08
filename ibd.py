"""
anti-nu_e + p -> n + e+

Strumia/Vissani (2003), arXiv:astro-ph/0302055.
"""

from math import pi, sqrt, log

mN = 939.56563  # neutron mass (MeV)
mP = 938.27231  # proton mass (MeV)
mE = 0.5109907  # electron mass (MeV)
mPi = 139.56995  # pion mass (MeV)
delta = mN - mP
mAvg = (mP + mN) / 2
alpha = 1 / 137.035989  # fine structure constant
gF = 1.16639e-11  # Fermi coupling constant
sigma0 = 2 * mP * gF**2 * 0.9746**2 / (8 * pi * mP**2)  # from eqs. (3), (11)
delta_cm = (mN**2 - mP**2 - mE**2) / (2 * mP)
eThr = ((mN + mE)**2 - mP**2) / (2 * mP)  # threshold energy for IBD: ca. 1.8 MeV

# List of neutrino flavors ("e", "eb", "x", "xb") that interact in this channel.
possible_flavors = ["eb"]

class Channel:
    def __init__(self, flavor):
        if flavor not in possible_flavors:
            raise ValueError(f"Flavor {flavor} cannot interact in IBD channel.")
        self.flavor = flavor

    # Se omite generate_event y funciones no relevantes 
    
    def dSigma_dE(self, eNu, eE):  # eqs. (11), (3)
        """Return differential cross section in MeV^-2.

        Inputs:
            eNu: neutrino energy
            eE:  energy of outgoing (detected) particle
        """
        if eNu < eThr or eE < self.bounds_eE(eNu)[0] or eE > self.bounds_eE(eNu)[1]:
            return 0
        # above eq. (11)
        s_minus_u = 2 * mP * (eNu + eE) - mE**2
        t = mN**2 - mP**2 - 2 * mP * (eNu - eE)

        # eq. (7)
        x = 0 + t / (4 * mAvg**2)
        y = 1 - t / 710**2
        z = 1 - t / 1030**2
        f1 = (1 - 4.706 * x) / ((1 - x) * y**2)
        f2 = 3.706 / ((1 - x) * y**2)
        g1 = -1.27 / z**2
        g2 = 2 * g1 * mAvg**2 / (mPi**2 - t)

        A = (t - mE**2) * (
                4 * f1**2 * (4 * mAvg**2 + t + mE**2)
                + 4 * g1**2 * (-4 * mAvg**2 + t + mE**2)
                + f2**2 * (t**2 / mAvg**2 + 4 * t + 4 * mE**2)
                + 4 * mE**2 * t * g2**2 / mAvg**2
                + 8 * f1 * f2 * (2 * t + mE**2)
                + 16 * mE**2 * g1 * g2) \
            - delta**2 * (
                (4 * f1**2 + t * f2**2 / mAvg**2) * (4 * mAvg**2 + t - mE**2)
                + 4 * g1**2 * (4 * mAvg**2 - t + mE**2)
                + 4 * mE**2 * g2**2 * (t - mE**2) / mAvg**2
                + 8 * f1 * f2 * (2 * t - mE**2)
                + 16 * mE**2 * g1 * g2) \
            - 32 * mE**2 * mAvg * delta * g1 * (f1 + f2)
        A /= 16

        B = 16 * t * g1 * (f1 + f2) + 4 * mE**2 * delta * (f2**2 + f1 * f2 + 2 * g1 * g2) / mAvg
        B /= 16

        C = 4 * (f1**2 + g1**2) - t * f2**2 / mAvg**2
        C /= 16

        abs_M_squared = A - B * s_minus_u + C * s_minus_u**2  # eq. (5)
        rad_correction = alpha / pi * (6.00352 + 3 / 2 * log(mP / (2 * eE)) + 1.2 * (mE / eE)**1.5)  # eq. (14)

        result = sigma0 / eNu**2 * abs_M_squared * (1 + rad_correction)

        if result < 0:
            raise ValueError(f"Calculated negative cross section for E_nu={eNu}, E_e={eE}. Aborting...")

        return result

    def bounds_eE(self, eNu, *args):
        """Return kinematic bounds for integration over eE.

        Input:
            eNu:  neutrino energy (in MeV)
            args: [ignore this]
        Output:
            list with minimum & maximum allowed energy of outgoing (detected) particle
        """
        if eNu <= eThr:
            return [0.0, 0.0]
            
        s = 2 * mP * eNu + mP**2
        pE_cm = sqrt((s - (mN - mE)**2) * (s - (mN + mE)**2)) / (2 * sqrt(s))
        eE_cm = (s - mN**2 + mE**2) / (2 * sqrt(s))

        eE_min = eNu - delta_cm - eNu / sqrt(s) * (eE_cm + pE_cm)
        eE_max = eNu - delta_cm - eNu / sqrt(s) * (eE_cm - pE_cm)
        return [eE_min, eE_max]

