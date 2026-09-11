from abc import ABC, abstractmethod

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc
from ripplegw.waveforms.IMRPhenomHM import gen_IMRPhenomHM
from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from ripplegw.waveforms.IMRPhenomXHM import gen_IMRPhenomXHM_hphc
from ripplegw.waveforms.IMRPhenomXP import gen_IMRPhenomXP_hphc
from ripplegw.waveforms.IMRPhenomXPHM import generate_xphm
from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.constants import MTSUN, G, C, MPC, EPSILON0, M_PL_GEV, PI, TWO_PI


class Waveform(ABC):
    """Abstract base class for gravitational waveform models.

    Subclasses implement the frequency- (or time-) domain waveform and expose it
    via ``__call__``, returning a dictionary with polarization keys ``"p"`` (plus)
    and ``"c"`` (cross).

    Attributes:
        f_ref (float): Reference frequency in Hz, at which the phase is aligned.
            Set by every frequency-domain model; time-domain models do not have
            one.
    """

    f_ref: float

    def __init__(self):
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Ordered tuple of parameter names required by this waveform model.

        Returns:
            tuple[str, ...]: Parameter names in the order they are consumed,
                matching the keys expected in the ``params`` dict passed to
                ``__call__``.
        """
        raise NotImplementedError(
            "Waveform.parameter_names must be implemented by subclasses"
        )

    @abstractmethod
    def __call__(
        self, axis: Float[Array, " n"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n"] | Complex[Array, " n"]]:
        """Evaluate the waveform.

        Args:
            axis (Float[Array, " n"]): Frequency or time grid.
            params (dict[str, Float]): Source parameter dictionary.

        Returns:
            dict[str, Float[Array, " n"] | Complex[Array, " n"]]: Dictionary
                with keys ``"p"`` (plus polarization) and ``"c"`` (cross
                polarization). Frequency-domain waveforms return complex arrays;
                time-domain waveforms return real arrays.
        """
        raise NotImplementedError("Waveform.__call__ must be implemented by subclasses")


class TaylorF2(Waveform):
    """TaylorF2 post-Newtonian frequency-domain waveform including tidal effects.

    Attributes:
        f_ref (float): Reference frequency in Hz.
        use_lambda_tildes (bool): If True, expects ``lambda_tilde`` and
            ``delta_lambda_tilde``; otherwise expects ``lambda_1`` and ``lambda_2``.
    """

    f_ref: float
    use_lambda_tildes: bool

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool): Whether to parameterise tidal deformability
                via ``lambda_tilde`` / ``delta_lambda_tilde`` (as in Eq. 5-6 of
                arXiv:1402.5156) instead of ``lambda_1`` / ``lambda_2``.
                Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            *(
                ("lambda_tilde", "delta_lambda_tilde")
                if self.use_lambda_tildes
                else ("lambda_1", "lambda_2")
            ),
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the TaylorF2 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys ``M_c``,
                ``eta``, ``s1_z``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``,
                plus tidal keys depending on ``use_lambda_tildes``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_TaylorF2_hphc(
            frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"TaylorF2(f_ref={self.f_ref})"


class IMRPhenomD(Waveform):
    """IMRPhenomD frequency-domain waveform (non-precessing, aligned spins).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomD waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_z``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomD(f_ref={self.f_ref})"


class IMRPhenomD_NRTidalv2(Waveform):
    """IMRPhenomD_NRTidalv2 frequency-domain waveform (non-precessing, NRTidalv2 tides).

    Attributes:
        f_ref (float): Reference frequency in Hz.
        use_lambda_tildes (bool): If True, expects ``lambda_tilde`` /
            ``delta_lambda_tilde``; otherwise ``lambda_1`` / ``lambda_2``.
        no_taper (bool): If True, the Planck taper in the amplitude is disabled.
    """

    f_ref: float
    use_lambda_tildes: bool
    no_taper: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool): Whether to parameterise tidal deformability
                via ``lambda_tilde`` / ``delta_lambda_tilde`` (Eq. 5-6 of
                arXiv:1402.5156) instead of ``lambda_1`` / ``lambda_2``.
                Defaults to False.
            no_taper (bool): Whether to remove the Planck taper in the amplitude
                (useful for relative binning runs). Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            *(
                ("lambda_tilde", "delta_lambda_tilde")
                if self.use_lambda_tildes
                else ("lambda_1", "lambda_2")
            ),
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomD_NRTidalv2 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys ``M_c``,
                ``eta``, ``s1_z``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``,
                plus tidal keys depending on ``use_lambda_tildes``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomD_NRTidalv2(f_ref={self.f_ref})"


class IMRPhenomHM(Waveform):
    """IMRPhenomHM frequency-domain waveform (aligned spins, higher-order modes).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        output = {}
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        hp, hc = gen_IMRPhenomHM(
            frequency,
            m1,
            m2,
            params["s1_z"],
            params["s2_z"],
            params["d_L"],
            params["iota"],
            params["phase_c"],
            self.f_ref,
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"IMRPhenomHM(f_ref={self.f_ref})"


class IMRPhenomPv2(Waveform):
    """IMRPhenomPv2 frequency-domain waveform (precessing spins).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_x",
            "s1_y",
            "s1_z",
            "s2_x",
            "s2_y",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomPv2 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomPv2(f_ref={self.f_ref})"


class IMRPhenomXAS(Waveform):
    """IMRPhenomXAS frequency-domain waveform (non-precessing, aligned spins, X family).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXAS waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_z``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXAS_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXAS(f_ref={self.f_ref})"


class IMRPhenomXAS_NRTidalv3(Waveform):
    """IMRPhenomXAS_NRTidalv3 frequency-domain waveform (non-precessing, NRTidalv3 tides).

    Attributes:
        f_ref (float): Reference frequency in Hz.
        use_lambda_tildes (bool): If True, expects ``lambda_tilde`` /
            ``delta_lambda_tilde``; otherwise ``lambda_1`` / ``lambda_2``.
        no_taper (bool): If True, the Planck taper in the amplitude is disabled.
    """

    f_ref: float
    use_lambda_tildes: bool
    no_taper: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool): Whether to parameterise tidal deformability
                via ``lambda_tilde`` / ``delta_lambda_tilde`` rather than
                ``lambda_1`` / ``lambda_2``. Defaults to False.
            no_taper (bool): Whether to disable tapering (useful for relative
                binning runs). Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            *(
                ("lambda_tilde", "delta_lambda_tilde")
                if self.use_lambda_tildes
                else ("lambda_1", "lambda_2")
            ),
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXAS_NRTidalv3 waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys ``M_c``,
                ``eta``, ``s1_z``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``,
                plus tidal keys depending on ``use_lambda_tildes``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXAS_NRTidalv3_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXAS_NRTidalv3(f_ref={self.f_ref})"


class IMRPhenomXHM(Waveform):
    """IMRPhenomXHM frequency-domain waveform (aligned spins, higher-order modes).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXHM waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_z``, ``s2_z``, ``d_L``,
                ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        theta = jnp.array(
            [
                m1,
                m2,
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXHM_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXHM(f_ref={self.f_ref})"


class IMRPhenomXP(Waveform):
    """IMRPhenomXP frequency-domain waveform (precessing spins, 22-mode only).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_x",
            "s1_y",
            "s1_z",
            "s2_x",
            "s2_y",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXP waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                params["d_L"],
                0.0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomXP_hphc(frequency, theta, self.f_ref)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXP(f_ref={self.f_ref})"


class IMRPhenomXPHM(Waveform):
    """IMRPhenomXPHM frequency-domain waveform (precessing spins, higher-order modes).

    Attributes:
        f_ref (float): Reference frequency in Hz.
    """

    f_ref: float

    def __init__(self, f_ref: float = 20.0) -> None:
        """
        Args:
            f_ref (float): Reference frequency in Hz. Defaults to 20.0.
        """
        self.f_ref = f_ref

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "M_c",
            "eta",
            "s1_x",
            "s1_y",
            "s1_z",
            "s2_x",
            "s2_y",
            "s2_z",
            "d_L",
            "phase_c",
            "iota",
        )

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the IMRPhenomXPHM waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters with keys
                ``M_c``, ``eta``, ``s1_x``, ``s1_y``, ``s1_z``,
                ``s2_x``, ``s2_y``, ``s2_z``, ``d_L``, ``phase_c``, ``iota``.

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        m1, m2 = Mc_eta_to_ms(jnp.array([params["M_c"], params["eta"]]))
        hp, hc = generate_xphm(
            m1,
            m2,
            params["s1_x"],
            params["s1_y"],
            params["s1_z"],
            params["s2_x"],
            params["s2_y"],
            params["s2_z"],
            params["d_L"],
            params["iota"],
            params["phase_c"],
            frequency,
            self.f_ref,
        )
        return {"p": hp, "c": hc}

    def __repr__(self):
        return f"IMRPhenomXPHM(f_ref={self.f_ref})"


class SineGaussian(Waveform):
    """Sine-Gaussian time-domain burst waveform."""

    def __init__(self) -> None:
        pass

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("Q", "f_0", "hrss", "phase", "e")

    def __call__(
        self, t: Float[Array, " n_time"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_time"]]:
        """Evaluate the SineGaussian waveform.

        Args:
            t (Float[Array, " n_time"]): Time grid centered at t=0. Create using
                ``jnp.arange(-duration/2, duration/2, 1/fs)``.
            params (dict[str, Float]): Source parameters with keys ``Q``
                (quality factor), ``f_0`` (central frequency in Hz), ``hrss``,
                ``phase``, ``e`` (eccentricity).

        Returns:
            dict[str, Float[Array, " n_time"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        theta = jnp.array(
            [
                params["Q"],
                params["f_0"],
                params["hrss"],
                params["phase"],
                params["e"],
            ]
        )
        hp, hc = gen_SineGaussian_hphc(t, theta)
        return {"p": hp, "c": hc}

    def __repr__(self):
        return "SineGaussian()"


def _continuous_phase(
    strain: Complex[Array, " n_freq"],
    frequency: Float[Array, " n_freq"],
    f_ref: float,
    chirp_mass_sec: Float,
) -> Float[Array, " n_freq"]:
    """Unwrapped strain phase, anchored so it is the true continuous phase.

    Two things can go wrong with a bare ``jnp.unwrap(jnp.angle(strain))``.

    It only tracks the right branch while the true phase advance between
    adjacent bins stays below ``PI``. That advance is ``TWO_PI * df * t(f)``
    with ``t(f)`` the time from ``f`` to merger, and ``t`` grows like
    ``f**(-8/3)``, so a grid reaching low frequency on a short segment aliases:
    measured 26.6 rad per bin for the ``m=3`` scalar harmonic of a 9.3 solar
    mass chirp on a 16 s segment. The leading-order SPA phase is subtracted
    before unwrapping and added back after, which is exact and costs no extra
    waveform evaluation. It leaves 0.54 rad per bin in that same worst case.
    The residual still exceeds ``PI`` below roughly 2.5 solar masses of chirp
    mass on a 4 s segment, which is a signal longer than its own segment.

    Separately, ``jnp.unwrap`` pins its first element to ``(-PI, PI]``, so its
    output can sit a whole multiple of ``2 PI`` away from the continuous phase.
    Rescaling that phase to another harmonic turns the offset into a spurious
    constant, and for odd harmonics into a sign flip that switches on and off
    as the source parameters vary. Waveforms aligned at ``f_ref`` have exactly
    zero phase there once ``t_c`` and ``phase_c`` are zero, which pins it.

    Args:
        strain (Complex[Array, " n_freq"]): Complex strain, generated with
            ``phase_c = 0``.
        frequency (Float[Array, " n_freq"]): Frequency array in Hz. Must span
            ``f_ref``, else the anchor is taken at the nearest grid edge and the
            branch is again undetermined.
        f_ref (float): Reference frequency of the waveform that produced
            ``strain``, in Hz.
        chirp_mass_sec (Float): Detector-frame chirp mass in seconds, used to
            build the de-chirping reference.

    Returns:
        Float[Array, " n_freq"]: Continuous phase in radians.
    """
    # ripple's angle(h) is -Psi_standard, hence the leading sign
    reference = -(3.0 / 128.0) * jnp.power(PI * chirp_mass_sec * frequency, -5.0 / 3.0)
    phase = jnp.unwrap(jnp.angle(strain) - reference) + reference
    anchor = phase[jnp.argmin(jnp.abs(frequency - f_ref))]
    return phase - TWO_PI * jnp.round(anchor / TWO_PI)


class DarkPhotonWaveform(Waveform):
    """Wraps a base frequency-domain waveform to model dark-photon dipole radiation.

    Generates the base waveform face-on (``iota=0``), decomposes its plus
    polarization into amplitude and phase, and reconstructs the dark-photon
    waveform with half the base GW phase. The charge-dependent amplitude
    scaling is reintroduced in ``_amplitude_scale``; the inclination
    dependence is reapplied in ``__call__``.

    Attributes:
        base_waveform (Waveform): The underlying waveform model to wrap.
    """

    base_waveform: Waveform

    def __init__(self, base_waveform: Waveform) -> None:
        """
        Args:
            base_waveform (Waveform): Waveform instance to wrap, e.g. ``IMRPhenomD()``.
        """
        self.base_waveform = base_waveform

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (*self.base_waveform.parameter_names, "sigma_1", "sigma_2")

    def _calc_fE(self, sigma_1, sigma_2, eta):
        q = (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) / (1.0 + jnp.sqrt(1.0 - 4.0 * eta))
        G12 = 1.0 - sigma_1 * sigma_2
        X1 = 1.0 / (1.0 + q)
        X2 = q / (1.0 + q)
        term = (
            -1.0
            / 3.0
            / G12**2
            * (
                G12**2 * ((1.0 + eta) / 4.0 + 3.0 / G12)
                - 1.0
                - X1 * sigma_2**2 * q
                - X2 * sigma_1**2 / q
                + 2.0 * sigma_1 * sigma_2
            )
        )
        return term

    def _calc_fgamma(self, sigma_1, sigma_2, eta):
        q = (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) / (1.0 + jnp.sqrt(1.0 - 4.0 * eta))
        G12 = 1.0 - sigma_1 * sigma_2
        X1 = 1.0 / (1.0 + q)
        X2 = q / (1.0 + q)
        term = (
            1.0
            / 6.0
            / G12**2
            * (
                G12**2 * (1.0 - 2.0 * eta)
                + 3.0 * G12
                + 2.0
                + 2.0 * X1 * sigma_2**2 * q
                + 2.0 * X2 * sigma_1**2 / q
                - 4.0 * sigma_1 * sigma_2
            )
        )
        return term

    def _calc_fT1byr2(self, sigma_1, sigma_2, eta):
        return 16.0 * (1.0 - 4.0 * eta)

    def _calc_fTv2byr(self, sigma_1, sigma_2, eta):
        q = (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) / (1.0 + jnp.sqrt(1.0 - 4.0 * eta))
        G12 = 1.0 - sigma_1 * sigma_2
        X1 = 1.0 / (1.0 + q)
        X2 = q / (1.0 + q)
        term = (
            -8.0
            / G12**2
            * (
                20.0 * (17.0 - eta)
                + 84.0 * (sigma_1**2 * X2 / q + X1 * sigma_2**2 * q)
                + (sigma_1 * sigma_2) ** 2 * (67.0 - 20.0 * eta)
                - sigma_1 * sigma_2 * (491.0 - 40.0 * eta)
            )
        )
        return term

    def _calc_fTv4(self, sigma_1, sigma_2, eta):
        G12 = 1.0 - sigma_1 * sigma_2
        return (785.0 - 281.0 * sigma_1 * sigma_2) / G12 - 852.0 * eta

    def _calc_fV1byr(self, sigma_1, sigma_2, eta):
        q = (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) / (1.0 + jnp.sqrt(1.0 - 4.0 * eta))
        G12 = 1.0 - sigma_1 * sigma_2
        X1 = 1.0 / (1.0 + q)
        X2 = q / (1.0 + q)
        diff = sigma_1 - sigma_2
        term = -2.0 * (
            2.0 * eta * diff**2
            + 2.0 / 5.0 * (X2**2 * sigma_1 - X1**2 * sigma_2) * diff
            - 5.0 * (sigma_1 * sigma_2) / G12**2 * diff**2
            + (X1 - X2) * (sigma_1 * X1 + sigma_2 * X2) * diff
            + (4.0 + X2 * sigma_1**2 / q + X1 * sigma_2**2 * q) * diff**2 / G12**2
        )
        return term

    def _calc_fVv2(self, sigma_1, sigma_2, eta):
        q = (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) / (1.0 + jnp.sqrt(1.0 - 4.0 * eta))
        G12 = 1.0 - sigma_1 * sigma_2
        X1 = 1.0 / (1.0 + q)
        X2 = q / (1.0 + q)
        diff = sigma_1 - sigma_2
        term = (
            2.0 / 5.0 * (X2**2 * sigma_1 - X1**2 * sigma_2) * diff
            + 2.0 * (X1 - X2) * (sigma_1 * X1 + sigma_2 * X2) * diff
            + diff**2 / G12 * (2.0 + 6.0 * eta + sigma_1 * sigma_2 * (1.0 - 6.0 * eta))
        )
        return term

    def _minus_one_pn_correction(self, sigma_1, sigma_2, eta):
        G12 = 1.0 - sigma_1 * sigma_2
        return -5.0 * G12 * jnp.power(sigma_1 - sigma_2, 2.0) / 3584.0 / eta

    def _zero_pn_correction(self, sigma_1, sigma_2, eta):
        q = (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) / (1.0 + jnp.sqrt(1.0 - 4.0 * eta))
        G12 = 1.0 - sigma_1 * sigma_2
        X1 = 1.0 / (1.0 + q)
        X2 = q / (1.0 + q)
        diff = sigma_1 - sigma_2
        fE = self._calc_fE(sigma_1, sigma_2, eta)
        fgamma = self._calc_fgamma(sigma_1, sigma_2, eta)
        fT1byr2 = self._calc_fT1byr2(sigma_1, sigma_2, eta)
        fTv2byr = self._calc_fTv2byr(sigma_1, sigma_2, eta)
        fTv4 = self._calc_fTv4(sigma_1, sigma_2, eta)
        fV1byr = self._calc_fV1byr(sigma_1, sigma_2, eta)
        fVv2 = self._calc_fVv2(sigma_1, sigma_2, eta)
        GRterm = 3.0 / 128.0 / eta
        term = (
            -G12
            / 4096.0
            / eta
            * (
                5.0
                / 168.0
                * 2.0
                * diff**2
                * (336.0 * fE - 672.0 * fgamma - fT1byr2 - fTv2byr - fTv4)
                - 96.0
                + 10.0 * (fV1byr + fVv2)
                + 40.0 * fgamma * diff**2
                + 16.0 * (X2 * sigma_1 + X1 * sigma_2) ** 2
            )
        )
        return term - GRterm

    def _one_pn_correction(self, sigma_1, sigma_2, eta):
        q = (1.0 - jnp.sqrt(1.0 - 4.0 * eta)) / (1.0 + jnp.sqrt(1.0 - 4.0 * eta))
        G12 = 1.0 - sigma_1 * sigma_2
        X1 = 1.0 / (1.0 + q)
        X2 = q / (1.0 + q)
        diff = sigma_1 - sigma_2
        fE = self._calc_fE(sigma_1, sigma_2, eta)
        fgamma = self._calc_fgamma(sigma_1, sigma_2, eta)
        fT1byr2 = self._calc_fT1byr2(sigma_1, sigma_2, eta)
        fTv2byr = self._calc_fTv2byr(sigma_1, sigma_2, eta)
        fTv4 = self._calc_fTv4(sigma_1, sigma_2, eta)
        fV1byr = self._calc_fV1byr(sigma_1, sigma_2, eta)
        fVv2 = self._calc_fVv2(sigma_1, sigma_2, eta)
        GRterm = 5.0 * (743.0 + 924.0 * eta) / (32256.0 * eta)
        term = (
            -5.0
            * G12
            / 1548288.0
            / eta
            * (
                -32256.0 * fE
                + (48.0 - 20.0 * fE * diff**2)
                * (672.0 * fgamma + fT1byr2 + fTv2byr + fTv4)
                + 5.0
                / 224.0
                * 2.0
                * diff**2
                * (672.0 * fgamma + fT1byr2 + fTv2byr + fTv4) ** 2
                - (672.0 * fgamma + fT1byr2 + fTv2byr + fTv4 - 336.0 * fE)
                * (
                    10.0 * (fV1byr + fVv2)
                    + 40.0 * fgamma * diff**2
                    + 16.0 * (X2 * sigma_1 + X1 * sigma_2) ** 2
                )
            )
        )
        return term - GRterm

    def _amplitude_scale(
        self,
        freq: Float[Array, " n_freq"],
        params: dict[str, Float],
    ) -> Float[Array, " n_freq"]:
        """Ratio of the dark-photon dipole to the GR quadrupole amplitude.

        Equals ``sqrt(2) * delta * v**2 / (4 * M_s)`` with
        ``v = (TWO_PI * M_s * freq)**(1/3)``, ``delta = sigma_1 - sigma_2`` and
        ``freq`` the orbital frequency. The ``sqrt(2)`` is the stationary-phase
        Jacobian ratio between the ``m=1`` and ``m=2`` harmonics.

        Args:
            freq (Float[Array, " n_freq"]): Orbital frequency array in Hz.
            params (dict[str, Float]): Full source parameter dictionary passed
                to ``__call__`` (includes ``sigma_1``, ``sigma_2``, ``iota``,
                and all of ``base_waveform.parameter_names``).

        Returns:
            Float[Array, " n_freq"]: Amplitude scaling factor.
        """
        delta = params["sigma_1"] - params["sigma_2"]
        total_mass = params["M_c"] * jnp.power(params["eta"], -3.0 / 5.0)
        total_mass *= MTSUN
        conv = (
            jnp.sqrt(2.0)
            * TWO_PI ** (2.0 / 3.0)
            * delta
            / (4.0 * total_mass ** (1.0 / 3.0))
            * freq ** (2.0 / 3.0)
        )
        return conv

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the dark-photon-modulated waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Frequency array in Hz.
            params (dict[str, Float]): Source parameters for ``base_waveform``,
                plus ``sigma_1`` and ``sigma_2`` (dark-photon charge-to-mass
                ratios of bodies 1 and 2).

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Plus (``"p"``) and cross (``"c"``)
                polarizations.
        """
        base_params = {
            k: v for k, v in params.items() if k not in ("sigma_1", "sigma_2")
        }
        # extract the iota and the phase_c
        iota = base_params["iota"]
        phase_c = base_params["phase_c"]
        # set iota to zero for easier amplitude and phase extraction
        base_params["iota"] = 0.0
        base_params["phase_c"] = 0.0
        base_hphc = self.base_waveform(frequency * 2.0, base_params)

        amp = jnp.abs(base_hphc["p"])
        phase = _continuous_phase(
            base_hphc["p"],
            frequency * 2.0,
            self.base_waveform.f_ref,
            params["M_c"] * MTSUN,
        )

        # Maxwell (-1, 0, +1 PN) dephasing from the dark-charge-dependent
        # orbital-phase evolution, on top of the pure-GR quadrupole phasing
        eta = params["eta"]
        sigma_1 = params["sigma_1"]
        sigma_2 = params["sigma_2"]
        total_mass_sec = params["M_c"] * eta ** (-3.0 / 5.0) * MTSUN
        vel = jnp.power(TWO_PI * total_mass_sec * frequency, 1.0 / 3.0)
        dephasing = (
            self._minus_one_pn_correction(sigma_1, sigma_2, eta) * vel ** (-7.0)
            + self._zero_pn_correction(sigma_1, sigma_2, eta) * vel ** (-5.0)
            + self._one_pn_correction(sigma_1, sigma_2, eta) * vel ** (-3.0)
        )
        phase = phase - dephasing

        scale = self._amplitude_scale(frequency, params)
        # ripple's angle(h) is -Psi_standard, so it carries a +PI/4 SPA term;
        # strip it before halving (else it becomes +PI/8) and reapply it at
        # full weight after
        phase_EM = (phase - PI / 4.0) / 2.0 + PI / 4.0 + phase_c
        amp_EM = amp * scale

        # unit conversion, convert waveform back to Telsa-second
        ke = 1.0 / (4.0 * PI * EPSILON0)
        amp_EM *= jnp.sqrt(ke / G)

        # convert to fT-second
        amp_EM *= 1e15

        return {
            "p": amp_EM * jnp.exp(1j * phase_EM),
            "c": amp_EM * jnp.cos(iota) * (-1j) * jnp.exp(1j * phase_EM),
        }

    def __repr__(self):
        return f"DarkPhotonWaveform(base_waveform={self.base_waveform!r})"


class ScalarWaveform(Waveform):
    """Wraps a base frequency-domain waveform to model scalar dipole radiation.

    Models the massless scalar dipole field radiated by a compact binary in
    shift-symmetric ESGB gravity, coupled to a nucleon spin through the
    derivative operator ``d_mu(phi**k)``, which produces an effective magnetic
    field ``B_eff = eps_BD * grad(phi**k)``. Only the radiation-zone gradient is
    kept, ``grad(phi**k) = -n_hat * d/dt(phi**k)``; the ``grad(1/R**k)`` piece is
    suppressed by ``1 / (Omega * R)`` and dropped.

    The dipole field is ``phi = phi_0(t) * sin(Psi(t))`` with ``Psi`` the orbital
    phase, so ``d/dt(phi**k)`` carries the harmonics of
    ``sin(Psi)**(k-1) * cos(Psi)``: a single ``m=2`` harmonic for ``k=2``, and
    ``m=1`` plus ``m=3`` with opposite signs for ``k=3``. Each harmonic is built
    from the base waveform evaluated at ``2 * frequency / m``, generated face-on
    (``iota=0``) so that its plus polarization gives the bare quadrupole
    amplitude and phase.

    The field is longitudinal, pointing along the line of sight, so the output
    has a single component keyed ``"s"`` rather than the usual plus/cross pair,
    and carries no polarization-angle dependence.

    Output is in fT-second, normalised to the benchmark couplings of
    ``Note_pulsar_search.md`` eq. (22) and (23), so ``eps_BD`` on the detector
    side is the dimensionless ratio ``(Lambda_ref / Lambda_k)**k`` with
    ``Lambda_ref = 0.5 GeV`` for ``k=2`` and ``0.01 GeV`` for ``k=3``. The
    benchmarks are read as the amplitude prefactor ``B_0``; any O(1) ambiguity
    there is absorbed by ``eps_BD``.

    Attributes:
        base_waveform (Waveform): The underlying waveform model to wrap.
        k (int): Power of the scalar field in the derivative coupling, 2 or 3.
    """

    base_waveform: Waveform
    k: int

    #: Harmonics of ``sin(Psi)**(k-1) * cos(Psi)`` as
    #: ``(m, coefficient, constant phase offset)``.
    _HARMONICS: dict[int, tuple[tuple[int, float, float], ...]] = {
        2: ((2, 0.5, -PI / 2.0),),
        3: ((1, 0.25, 0.0), (3, 0.25, PI)),
    }
    #: Effective-field benchmarks in fT, ``Note_pulsar_search.md`` eq. (22), (23).
    _B_REF: dict[int, float] = {2: 0.019, 3: 7.0e-3}
    #: Reference field amplitude of the benchmarks, in GeV.
    _PHI_REF: float = 1.0e-6
    #: Reference frequency of the benchmarks, in Hz.
    _F_REF: float = 50.0

    def __init__(self, base_waveform: Waveform, k: int = 2) -> None:
        """
        Args:
            base_waveform (Waveform): Waveform instance to wrap, e.g. ``IMRPhenomD()``.
            k (int): Power of the scalar field in the derivative coupling,
                either 2 or 3.
        """
        if k not in self._HARMONICS:
            raise ValueError(f"k must be one of {tuple(self._HARMONICS)}, got {k}")
        self.base_waveform = base_waveform
        self.k = k

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (*self.base_waveform.parameter_names, "sigma_1", "sigma_2")

    def _minus_one_pn_correction(
        self, sigma_1: Float, sigma_2: Float, eta: Float
    ) -> Float:
        """Scalar dipole dephasing at -1PN.

        Scalar dipole radiation drains orbital energy at
        ``P_dip = eta**2 * (sigma_1 - sigma_2)**2 * v**8 / (12 * pi * G)``, one
        PN order below the GR quadrupole, so it enters the phasing at ``v**-7``.
        Not yet derived in ripple's phasing convention; returns zero.

        Args:
            sigma_1 (Float): Scalar charge-to-mass ratio of body 1.
            sigma_2 (Float): Scalar charge-to-mass ratio of body 2.
            eta (Float): Symmetric mass ratio.

        Returns:
            Float: Dephasing coefficient, currently zero.
        """
        return 0.0

    def _harmonic(
        self,
        frequency: Float[Array, " n_freq"],
        params: dict[str, Float],
        m: int,
        coeff: float,
        offset: float,
    ) -> Complex[Array, " n_freq"]:
        """Evaluate a single harmonic of ``grad(phi**k)``.

        Args:
            frequency (Float[Array, " n_freq"]): Field frequency array in Hz.
            params (dict[str, Float]): Source parameters, as passed to ``__call__``.
            m (int): Harmonic number, so the orbital frequency is ``frequency / m``.
            coeff (float): Coefficient of this harmonic in
                ``sin(Psi)**(k-1) * cos(Psi)``.
            offset (float): Constant phase offset of this harmonic, in radians.

        Returns:
            Complex[Array, " n_freq"]: Harmonic contribution in fT-second.
        """
        base_params = {
            key: value
            for key, value in params.items()
            if key not in ("sigma_1", "sigma_2")
        }
        # extract the iota and the phase_c
        iota = base_params["iota"]
        phase_c = base_params["phase_c"]
        # set iota to zero for easier amplitude and phase extraction
        base_params["iota"] = 0.0
        base_params["phase_c"] = 0.0
        base_hphc = self.base_waveform(2.0 * frequency / m, base_params)

        amp = jnp.abs(base_hphc["p"])
        phase = _continuous_phase(
            base_hphc["p"],
            2.0 * frequency / m,
            self.base_waveform.f_ref,
            params["M_c"] * MTSUN,
        )

        eta = params["eta"]
        total_mass_sec = params["M_c"] * eta ** (-3.0 / 5.0) * MTSUN
        reduced_mass_sec = eta * total_mass_sec
        dist_sec = params["d_L"] * MPC / C
        orb_freq = frequency / m
        vel = jnp.power(TWO_PI * total_mass_sec * orb_freq, 1.0 / 3.0)

        # scalar dipole dephasing, on top of the pure-GR quadrupole phasing;
        # applied before the harmonic rescaling below
        phase = phase - self._minus_one_pn_correction(
            params["sigma_1"], params["sigma_2"], eta
        ) * vel ** (-7.0)

        # the sign of the charge difference is degenerate with a PI shift of
        # phase_c, so only its magnitude is used
        delta = jnp.abs(params["sigma_1"] - params["sigma_2"])
        field = (
            M_PL_GEV
            * reduced_mass_sec
            * delta
            * jnp.sin(iota)
            * vel
            / (4.0 * PI * dist_sec)
        )
        field_amp = (
            self._B_REF[self.k]
            * (orb_freq / self._F_REF)
            * jnp.power(field / self._PHI_REF, self.k)
        )
        # leading-order GR quadrupole amplitude, used to strip the one power of
        # the base amplitude that the scalar harmonic inherits from it
        gw_amp = 4.0 * reduced_mass_sec * vel**2.0 / dist_sec
        # sqrt(2 / m) is the stationary-phase Jacobian ratio, since the base
        # waveform's m=2 harmonic sweeps twice the orbital frequency
        scale = coeff * jnp.sqrt(2.0 / m) * field_amp / gw_amp

        # ripple's angle(h) is -Psi_standard, so it carries a +PI/4 SPA term;
        # strip it before rescaling to the m-th harmonic and reapply it at full
        # weight after
        phase_s = 0.5 * m * (phase - PI / 4.0) + PI / 4.0 + m * phase_c + offset
        return amp * scale * jnp.exp(1j * phase_s)

    def __call__(
        self, frequency: Float[Array, " n_freq"], params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate the scalar effective-field waveform.

        Args:
            frequency (Float[Array, " n_freq"]): Field frequency array in Hz.
                This is a harmonic of the orbital frequency, not the GW
                frequency. The array must be restricted to the analysis band,
                since the phase unwrapping couples all of its entries.
            params (dict[str, Float]): Source parameters for ``base_waveform``,
                plus ``sigma_1`` and ``sigma_2`` (scalar charge-to-mass ratios
                of bodies 1 and 2).

        Returns:
            dict[str, Complex[Array, " n_freq"]]: Longitudinal component
                (``"s"``) of ``grad(phi**k) / eps_BD`` in fT-second.
        """
        harmonics = [
            self._harmonic(frequency, params, m, coeff, offset)
            for m, coeff, offset in self._HARMONICS[self.k]
        ]
        return {"s": sum(harmonics[1:], start=harmonics[0])}

    def __repr__(self):
        return f"ScalarWaveform(base_waveform={self.base_waveform!r}, k={self.k})"


#: Mapping from model name strings to ``Waveform`` subclasses.
#: Useful for selecting waveform models by name at runtime, e.g. from a
#: configuration file.
waveform_preset: dict[str, type[Waveform]] = {
    "TaylorF2": TaylorF2,
    "IMRPhenomD": IMRPhenomD,
    "IMRPhenomD_NRTidalv2": IMRPhenomD_NRTidalv2,
    "IMRPhenomHM": IMRPhenomHM,
    "IMRPhenomPv2": IMRPhenomPv2,
    "IMRPhenomXAS": IMRPhenomXAS,
    "IMRPhenomXAS_NRTidalv3": IMRPhenomXAS_NRTidalv3,
    "IMRPhenomXHM": IMRPhenomXHM,
    "IMRPhenomXP": IMRPhenomXP,
    "IMRPhenomXPHM": IMRPhenomXPHM,
    "SineGaussian": SineGaussian,
}
