import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ase.atoms import Atoms


def write_mlip_input_file(
    working_directory: str,
    mtp_filename: str,
    active_learning: bool = False,
    threshold: float = 2.0,
    threshold_break: float = 5.0,
    save_selected: str = "selected.cfg",
    load_state: str = "state.mvs",
    log: str = "selection.log",
    file_name: str = "mlip.ini",
) -> str:
    """
    Write the ``mlip.ini`` control file consumed by LAMMPS' ``pair_style mlip``.

    Args:
        working_directory (str): Directory the file is written into. Created
            if it does not already exist.
        mtp_filename (str): Absolute path to the underlying ``.mtp`` potential
            file. Resolve this the same way other potential file paths are
            resolved elsewhere in lammpsparser (e.g. via
            ``potential["Filename"][0]`` after
            :func:`~lammpsparser.potential.update_potential_paths`).
        active_learning (bool): If ``True``, enable MLIP's active-learning
            selection mode: LAMMPS flags (and optionally stops on) structures
            where the potential is extrapolating.
        threshold (float): Extrapolation grade above which a structure is
            selected for the training set. Only used when
            ``active_learning=True``.
        threshold_break (float): Extrapolation grade above which the LAMMPS
            run is stopped. Only used when ``active_learning=True``.
        save_selected (str): Filename (relative to ``working_directory``)
            that MLIP writes selected configurations to. Only used when
            ``active_learning=True``.
        load_state (str): Filename (relative to ``working_directory``) for
            MLIP's selection state file. Only used when
            ``active_learning=True``.
        log (str): Filename (relative to ``working_directory``) for MLIP's
            selection log. Only used when ``active_learning=True``.
        file_name (str): Name of the file to write (default: ``"mlip.ini"``).

    Returns:
        str: Absolute path to the written file.
    """
    lines = ["mtp-filename " + mtp_filename]
    if active_learning:
        lines += [
            "calculate-efs TRUE",
            "select TRUE",
            "select:threshold " + str(threshold),
            "select:threshold-break " + str(threshold_break),
            "select:save-selected " + save_selected,
            "select:load-state " + load_state,
            "select:log " + log,
            "write-cfgs:skip 0",
        ]
    else:
        lines.append("select FALSE")

    os.makedirs(working_directory, exist_ok=True)
    file_path = os.path.abspath(os.path.join(working_directory, file_name))
    with open(file_path, "w") as f:
        f.writelines([line + "\n" for line in lines])
    return file_path


def check_mlip_convergence(
    working_directory: str, error_file_name: str = "error.out"
) -> bool:
    """
    Check whether an MLIP active-learning run converged (did not break early).

    Args:
        working_directory (str): Directory containing the LAMMPS run's
            ``error.out`` file, if any.
        error_file_name (str): Filename (relative to ``working_directory``)
            to check (default: ``"error.out"``).

    Returns:
        bool: ``True`` if ``error_file_name`` does not exist, or exists but
        contains no line starting with ``"MLIP: Breaking threshold
        exceeded"``. ``False`` otherwise.
    """
    error_file_path = os.path.join(working_directory, error_file_name)
    if not os.path.exists(error_file_path):
        return True
    with open(error_file_path) as f:
        for line in f:
            if line.startswith("MLIP: Breaking threshold exceeded"):
                return False
    return True


MLIP_STRESS_COMPONENTS: Tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass
class MlipConfiguration:
    """
    One parsed configuration from an MLIP ``.cfg`` file.

    Attributes:
        cell (numpy.ndarray): Shape ``(3, 3)`` simulation cell, from the
            ``Supercell`` section.
        positions (numpy.ndarray): Shape ``(N, 3)`` Cartesian atomic
            positions.
        types (numpy.ndarray): Shape ``(N,)`` integer species-type indices,
            0-based, as written by MLIP.
        forces (numpy.ndarray, optional): Shape ``(N, 3)`` if the
            ``AtomData:`` header includes ``fx fy fz``, else ``None``.
        energy (float, optional): From the ``Energy`` section, if present.
        stress (numpy.ndarray, optional): Shape ``(6,)`` in ASE Voigt order
            ``(xx, yy, zz, yz, xz, xy)``, from ``PlusStress:``, if present.
        grade (float, optional): Extrapolation grade from a
            ``Feature   MV_grade   <value>`` line, if present.
    """

    cell: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    forces: Optional[np.ndarray] = None
    energy: Optional[float] = None
    stress: Optional[np.ndarray] = None
    grade: Optional[float] = None


def load_mlip_cfgs(file_name: str) -> List[MlipConfiguration]:
    """
    Parse an MLIP ``.cfg`` file into a list of configurations.

    Args:
        file_name (str): Path to a ``.cfg`` file containing one or more
            ``BEGIN_CFG``/``END_CFG`` blocks (e.g. MLIP's
            ``select:save-selected`` output).

    Returns:
        list[MlipConfiguration]: One entry per ``BEGIN_CFG``/``END_CFG``
        block, in file order.

    Raises:
        ValueError: If an ``END_CFG`` is found without a matching
            ``BEGIN_CFG``, or a block is missing its required ``Supercell``
            or ``AtomData:`` section.
    """
    with open(file_name) as f:
        raw_lines = f.readlines()

    configurations: List[MlipConfiguration] = []
    block: Optional[List[str]] = None
    for line in raw_lines:
        stripped = line.strip()
        if stripped == "BEGIN_CFG":
            block = []
        elif stripped == "END_CFG":
            if block is None:
                raise ValueError("Found 'END_CFG' without a matching 'BEGIN_CFG'.")
            configurations.append(_parse_cfg_block(block))
            block = None
        elif block is not None:
            block.append(line)
    return configurations


def _parse_cfg_block(lines: List[str]) -> MlipConfiguration:
    """
    Parse the lines between one ``BEGIN_CFG``/``END_CFG`` pair.

    Args:
        lines (list[str]): Raw lines strictly between ``BEGIN_CFG`` and
            ``END_CFG`` (exclusive of both markers).

    Returns:
        MlipConfiguration: Parsed configuration.

    Raises:
        ValueError: If the block is missing its required ``Supercell`` or
            ``AtomData:`` section, or ``AtomData:`` appears before ``Size``.
    """
    cell: Optional[np.ndarray] = None
    size: Optional[int] = None
    atom_columns: Optional[List[str]] = None
    atom_rows: List[List[float]] = []
    energy: Optional[float] = None
    stress: Optional[np.ndarray] = None
    grade: Optional[float] = None

    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("Size"):
            i += 1
            size = int(lines[i].split()[0])
        elif stripped.startswith("Supercell"):
            cell = np.array(
                [[float(v) for v in lines[i + 1 + row].split()] for row in range(3)]
            )
            i += 3
        elif stripped.startswith("AtomData"):
            if size is None:
                raise ValueError("MLIP cfg block has 'AtomData' before 'Size'.")
            atom_columns = stripped.split(":", 1)[1].split()
            for _ in range(size):
                i += 1
                atom_rows.append([float(v) for v in lines[i].split()])
        elif stripped.startswith("Energy"):
            i += 1
            energy = float(lines[i].split()[0])
        elif stripped.startswith("PlusStress"):
            header = stripped.split(":", 1)[1].split()
            i += 1
            values = [float(v) for v in lines[i].split()]
            stress_by_component = dict(zip(header, values))
            stress = np.array(
                [stress_by_component[component] for component in MLIP_STRESS_COMPONENTS]
            )
        elif stripped.startswith("Feature") and "MV_grade" in stripped:
            grade = float(stripped.split()[-1])
        i += 1

    if cell is None or atom_columns is None:
        raise ValueError(
            "MLIP cfg block is missing a required 'Supercell' or 'AtomData' section."
        )

    atom_array = np.array(atom_rows)
    column_index = {name: idx for idx, name in enumerate(atom_columns)}
    types = atom_array[:, column_index["type"]].astype(int)
    if "cartes_x" in column_index:
        positions = atom_array[
            :,
            [
                column_index["cartes_x"],
                column_index["cartes_y"],
                column_index["cartes_z"],
            ],
        ]
    else:
        fractional = atom_array[
            :,
            [
                column_index["direct_x"],
                column_index["direct_y"],
                column_index["direct_z"],
            ],
        ]
        positions = fractional @ cell

    forces = None
    if "fx" in column_index:
        forces = atom_array[
            :, [column_index["fx"], column_index["fy"], column_index["fz"]]
        ]

    return MlipConfiguration(
        cell=cell,
        positions=positions,
        types=types,
        forces=forces,
        energy=energy,
        stress=stress,
        grade=grade,
    )


def get_mlip_selected_structures(file_name: str, species: List[str]) -> List[Atoms]:
    """
    Build structures from an MLIP ``.cfg`` file (typically ``selected.cfg``).

    Args:
        file_name (str): Path to a ``.cfg`` file, as accepted by
            :func:`load_mlip_cfgs`.
        species (list[str]): Chemical symbols in the same order as the
            interatomic potential declares them (e.g. the potential
            dataframe's ``"Species"`` column), used to map each
            configuration's integer type indices to element symbols.

    Returns:
        list[ase.atoms.Atoms]: One ``Atoms`` object per configuration in
        ``file_name``, with ``pbc=True``. When present in the source
        configuration: per-atom forces are stored in ``atoms.arrays["forces"]``,
        and the total energy, stress (ASE Voigt order), and MLIP
        extrapolation grade are stored in ``atoms.info["energy"]``,
        ``atoms.info["stress"]``, and ``atoms.info["mv_grade"]``
        respectively.
    """
    structures = []
    for cfg in load_mlip_cfgs(file_name):
        symbols = np.asarray(species)[cfg.types]
        atoms = Atoms(
            symbols=symbols,
            positions=cfg.positions,
            cell=cfg.cell,
            pbc=True,
        )
        if cfg.forces is not None:
            atoms.arrays["forces"] = cfg.forces
        if cfg.energy is not None:
            atoms.info["energy"] = cfg.energy
        if cfg.stress is not None:
            atoms.info["stress"] = cfg.stress
        if cfg.grade is not None:
            atoms.info["mv_grade"] = cfg.grade
        structures.append(atoms)
    return structures
