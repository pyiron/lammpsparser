from dataclasses import dataclass
from typing import Optional, Union

from ase.atoms import Atoms
import numpy as np


@dataclass
class CalcMDInput:
    temperature: Optional[Union[float, list]] = None
    pressure: Optional[Union[float, list, np.ndarray]] = None
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
    units: str = "metal"


@dataclass
class CalcMinimizeInput:
    structure: Atoms
    ionic_energy_tolerance: float = 0.0
    ionic_force_tolerance: float = 1e-4
    max_iter: int = 100000
    pressure: Optional[Union[float, list, np.ndarray]] = None
    n_print: int = 100
    style: str = "cg"
    rotation_matrix: Optional[Union[list, np.ndarray]] = None
    units: str = "metal"
