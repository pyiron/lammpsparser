from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class CalcMDInput:
    """
    Input parameters for a LAMMPS molecular-dynamics run via :func:`~lammpsparser.compatibility.calculate.calc_md`.

    All time-like values (``time_step``, ``temperature_damping_timescale``,
    ``pressure_damping_timescale``) are given in femtoseconds and are
    internally rescaled to the target LAMMPS unit system before being written
    to the input script.

    Attributes:
        temperature: Target temperature in K.  Scalar for a fixed target, or
            a two-element list ``[T_start, T_end]`` for a linear ramp.  If
            ``None``, an NVE run is performed.
        pressure: Target pressure in GPa.  Scalar for isotropic, list of up
            to 6 values for ``[xx, yy, zz, xy, xz, yz]`` components.  ``None``
            values leave the corresponding cell degree of freedom unconstrained.
            If ``None``, no barostat is applied.
        n_ionic_steps: Number of MD timesteps to run.
        time_step: MD timestep in fs (default: ``1.0``).
        n_print: Write thermo and dump output every this many steps.
        temperature_damping_timescale: Nosé–Hoover thermostat coupling time
            in fs (LAMMPS ``Tdamp``).
        pressure_damping_timescale: Nosé–Hoover barostat coupling time in fs
            (LAMMPS ``Pdamp``).
        seed: Random seed for velocity initialisation and Langevin dynamics.
        tloop: Optional ``tloop`` argument for the ``fix nvt/npt`` command.
        initial_temperature: Temperature used to initialise velocities.
            ``None`` defaults to twice the target temperature.  ``0`` skips
            velocity initialisation.
        langevin: Use Langevin dynamics instead of Nosé–Hoover.
        delta_temp: Deprecated alias for ``temperature_damping_timescale``.
        delta_press: Deprecated alias for ``pressure_damping_timescale``.
        rotation_matrix: 3×3 rotation matrix from the ASE to LAMMPS coordinate
            frame.  Derived automatically when ``None``.
    """

    temperature: Optional[Union[float, list]] = None
    pressure: Optional[Union[float, list, np.ndarray]] = None
    n_ionic_steps: int = 1
    time_step: float = 1.0
    n_print: int = 100
    temperature_damping_timescale: float = 100.0
    pressure_damping_timescale: float = 1000.0
    seed: int = 80996
    tloop: Optional[int] = None
    initial_temperature: Optional[float] = None
    langevin: bool = False
    delta_temp: Optional[float] = None
    delta_press: Optional[float] = None
    rotation_matrix: Optional[Union[list, np.ndarray]] = None


@dataclass
class CalcMinimizeInput:
    """
    Input parameters for a LAMMPS geometry optimisation via :func:`~lammpsparser.compatibility.calculate.calc_minimize`.

    Attributes:
        ionic_energy_tolerance: Convergence criterion on the energy change
            between consecutive steps in eV (LAMMPS ``etol``).  The
            minimisation stops when ``|ΔE| ≤ ionic_energy_tolerance``.
        ionic_force_tolerance: Convergence criterion on the global force
            vector magnitude in eV/Å (LAMMPS ``ftol``).
        max_iter: Maximum number of minimisation iterations.
        pressure: Target pressure in GPa for cell relaxation.  Scalar for
            isotropic, list of up to 6 Voigt components.  ``None`` means
            fixed-volume relaxation.
        n_print: Write thermo and dump output every this many steps.
        style: LAMMPS ``min_style`` keyword (e.g. ``"cg"`` for conjugate
            gradient, ``"sd"`` for steepest descent).
        rotation_matrix: 3×3 rotation matrix from the ASE to LAMMPS frame.
    """

    ionic_energy_tolerance: float = 0.0
    ionic_force_tolerance: float = 1e-4
    max_iter: int = 100000
    pressure: Optional[Union[float, list, np.ndarray]] = None
    n_print: int = 1
    style: str = "cg"
    rotation_matrix: Optional[Union[list, np.ndarray]] = None
