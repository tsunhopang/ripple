"""Pins the dark-field effective-magnetic-field normalization to the reference note.

Every check here is against ``Note_pulsar_search_update.tex``, the version that
fixed a factor of 2 in the Xe and Rb benchmarks. Earlier drafts of the note quoted
0.4 fT and 2.1 fT for the scalar Xe case where the boxed equation gives 0.8 fT and
6.4 fT, so these tests exist to keep the implementation on the boxed-equation side.
"""

import jax.numpy as jnp
import pytest

from ripplegw import IMRPhenomD, ScalarWaveform
from ripplegw.constants import (
    E_CHARGE_HL,
    FT_GEV2,
    G_XE129,
    GAMMA_XE,
    HZ_GEV,
    M_PL_GEV,
    M_PROTON_GEV,
    MTSUN,
    MPC,
    C,
    PI,
    TWO_PI,
    V_H_GEV,
)

#: Benchmark point of Note_pulsar_search_update.tex eq. (28): (k, Lambda in GeV,
#: B_eff in fT) at phi = 1e-5 GeV and omega = 2 pi * 100 Hz.
SCALAR_BENCHMARKS = [(2, 2.0, 0.8), (3, 0.02, 6.4)]


def test_gamma_xe_matches_note():
    """GAMMA_XE reproduces the 0.249 GeV^-1 quoted below eq. (17)."""
    assert GAMMA_XE == pytest.approx(0.249, rel=2e-3)


@pytest.mark.parametrize("k, Lambda, expected_fT", SCALAR_BENCHMARKS)
def test_scalar_benchmark_values(k, Lambda, expected_fT):
    """Eq. (28): B_eff = 2 k omega phi^k / (Lambda^k gamma_Xe).

    The note quotes 0.8 fT to one significant figure against an exact 0.849, so
    the tolerance has to absorb its rounding. It is still far tighter than the
    factor of 2 that an earlier draft of the note carried here.
    """
    phi = 1e-5
    omega = TWO_PI * 100.0 * HZ_GEV
    b_eff = 2.0 * k * omega * phi**k / (Lambda**k * GAMMA_XE) / FT_GEV2
    assert b_eff == pytest.approx(expected_fT, rel=0.08)


def test_scalar_benchmark_matches_g_factor_form():
    """The 4 k m_N / (g_N e Lambda^k) grouping the waveform uses is the same thing."""
    phi, k, Lambda = 1e-5, 2, 2.0
    omega = TWO_PI * 100.0 * HZ_GEV
    via_gamma = 2.0 * k * omega * phi**k / (Lambda**k * GAMMA_XE)
    via_g_factor = (
        4.0
        * k
        * M_PROTON_GEV
        * omega
        * phi**k
        / (abs(G_XE129) * E_CHARGE_HL * Lambda**k)
    )
    assert via_g_factor == pytest.approx(via_gamma, rel=1e-12)


def test_darkphoton_benchmark():
    """Eq. (17): B_eff = 4.4e-6 (30 TeV / Lambda_N)^2 B' for 129Xe."""
    Lambda_N = 3.0e4
    b_eff = 8.0 * M_PROTON_GEV * V_H_GEV / (abs(G_XE129) * E_CHARGE_HL * Lambda_N**2)
    assert b_eff == pytest.approx(4.4e-6, rel=2e-2)


@pytest.mark.parametrize("k", [2, 3])
def test_scalar_waveform_normalization(k):
    """ScalarWaveform's harmonics carry exactly the eq. (22) prefactor.

    Rebuilds each harmonic's amplitude from the note rather than from the
    implementation: it is the base quadrupole amplitude rescaled by
    ``B_eff / gw_amp``, with ``B_eff = 2 grad(phi**k) / (Lambda**k gamma_Xe)``.
    Harmonics are checked one at a time so the comparison does not depend on the
    phasing that combines them.
    """
    base = IMRPhenomD(f_ref=20.0)
    model = ScalarWaveform(base, k=k)
    params = {
        "M_c": 30.0,
        "eta": 0.2,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "d_L": 400.0,
        "phase_c": 0.0,
        "iota": PI / 2.0,
        "sigma_1": 0.3,
        "sigma_2": 0.0,
        "Lambda": 1.0,
    }
    frequency = jnp.linspace(25.0, 60.0, 16)

    total_mass_sec = params["M_c"] * params["eta"] ** (-3.0 / 5.0) * MTSUN
    reduced_mass_sec = params["eta"] * total_mass_sec
    dist_sec = params["d_L"] * MPC / C
    delta = abs(params["sigma_1"] - params["sigma_2"])

    base_params = {
        key: value
        for key, value in params.items()
        if key not in ("sigma_1", "sigma_2", "Lambda")
    }
    base_params["iota"] = 0.0
    base_params["phase_c"] = 0.0

    for m, coeff, offset in model._HARMONICS[k]:
        amp = jnp.abs(base(2.0 * frequency / m, base_params)["p"])
        orb_freq = frequency / m
        vel = jnp.power(TWO_PI * total_mass_sec * orb_freq, 1.0 / 3.0)
        phi_0 = (
            M_PL_GEV
            * reduced_mass_sec
            * delta
            * jnp.sin(params["iota"])
            * vel
            / (4.0 * PI * dist_sec)
        )
        omega = TWO_PI * orb_freq * HZ_GEV
        b_eff = (
            2.0 * k * omega * jnp.power(phi_0, k) / (params["Lambda"] ** k * GAMMA_XE)
        ) / FT_GEV2
        gw_amp = 4.0 * reduced_mass_sec * vel**2.0 / dist_sec
        expected = amp * coeff * jnp.sqrt(2.0 / m) * b_eff / gw_amp

        actual = jnp.abs(model._harmonic(frequency, params, m, coeff, offset))
        assert jnp.allclose(actual, expected, rtol=1e-10)
