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
from ripplegw.constants import MTSUN, PI, TWO_PI


class Waveform(ABC):
    """Abstract base class for gravitational waveform models.

    Subclasses implement the frequency- (or time-) domain waveform and expose it
    via ``__call__``, returning a dictionary with polarization keys ``"p"`` (plus)
    and ``"c"`` (cross).
    """

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


class DarkPhotonWaveform(Waveform):
    """Wraps a base frequency-domain waveform to model dark-photon dipole radiation.

    Generates the base waveform face-on (``iota=0``), decomposes its plus and
    cross polarizations into amplitude and phase, and reconstructs the
    dark-photon waveform with half the base GW phase. The inclination
    dependence (removed by generating at ``iota=0``) and the charge-dependent
    amplitude scaling are both reintroduced in ``_amplitude_scale``.

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
        """Charge- and inclination-dependent amplitude scaling.

        Args:
            amp (Float[Array, " n_freq"]): Plus-polarization amplitude of the
                base waveform, generated at ``iota=0``.
            params (dict[str, Float]): Full source parameter dictionary passed
                to ``__call__`` (includes ``sigma_1``, ``sigma_2``, ``iota``,
                and all of ``base_waveform.parameter_names``).

        Returns:
            Float[Array, " n_freq"]: Scaled amplitude.
        """
        delta = params["sigma_1"] - params["sigma_2"]
        total_mass = params["M_c"] * jnp.power(params["eta"], -3.0 / 5.0)
        total_mass *= MTSUN
        conv = (
            (PI ** (2.0 / 3.0) * delta)
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
        phase = jnp.unwrap(jnp.angle(base_hphc["p"]))

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
        # base_waveform's SPA phase embeds a -PI/4 correction; strip it before
        # halving (else it becomes -PI/8) and reapply it at full weight after
        phase_EM = (phase + PI / 4.0) / 2.0 - PI / 4.0 + phase_c
        amp_EM = amp * scale

        return {
            "p": amp_EM * jnp.exp(1j * phase_EM),
            "c": amp_EM * jnp.cos(iota) * (-1j) * jnp.exp(1j * phase_EM),
        }

    def __repr__(self):
        return f"DarkPhotonWaveform(base_waveform={self.base_waveform!r})"


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
