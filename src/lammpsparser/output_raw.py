from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class DumpData:
    steps: List = field(default_factory=lambda: [])
    natoms: List = field(default_factory=lambda: [])
    cells: List = field(default_factory=lambda: [])
    indices: List = field(default_factory=lambda: [])
    forces: List = field(default_factory=lambda: [])
    mean_forces: List = field(default_factory=lambda: [])
    velocities: List = field(default_factory=lambda: [])
    mean_velocities: List = field(default_factory=lambda: [])
    unwrapped_positions: List = field(default_factory=lambda: [])
    mean_unwrapped_positions: List = field(default_factory=lambda: [])
    positions: List = field(default_factory=lambda: [])
    computes: Dict = field(default_factory=lambda: {})


def to_amat(l_list: Union[np.ndarray, List]) -> List:
    """
    Convert LAMMPS box bounds to a cell matrix in the lower-triangular convention used by ASE.

    LAMMPS stores triclinic box boundaries as ``xlo_bound``, ``xhi_bound``, ``xy``,
    ``ylo_bound``, ``yhi_bound``, ``xz``, ``zlo_bound``, ``zhi_bound``, ``yz``
    (the 9-element form written by ``dump … BOX BOUNDS xy xz yz``), or as the
    6-element orthorhombic form without tilt factors.  This function recovers the
    conventional cell vectors ``a = (xhilo, 0, 0)``, ``b = (xy, yhilo, 0)``,
    ``c = (xz, yz, zhilo)`` following the conversion documented in the LAMMPS
    manual under *Triclinic (non-orthogonal) simulation boxes*.

    Args:
        l_list (numpy.ndarray or list): Flattened sequence of either 6 box-bound
            values ``[xlo_bound, xhi_bound, ylo_bound, yhi_bound, zlo_bound,
            zhi_bound]`` (orthogonal box) or 9 values ``[xlo_bound, xhi_bound,
            xy, ylo_bound, yhi_bound, xz, zlo_bound, zhi_bound, yz]``
            (triclinic box).

    Returns:
        list: 3×3 nested list representing the lower-triangular cell matrix
        ``[[a, 0, 0], [xy, b, 0], [xz, yz, c]]``.

    Raises:
        ValueError: If ``l_list`` does not contain exactly 6 or 9 elements.
    """
    lst = np.reshape(l_list, -1)
    if len(lst) == 9:
        (
            xlo_bound,
            xhi_bound,
            xy,
            ylo_bound,
            yhi_bound,
            xz,
            zlo_bound,
            zhi_bound,
            yz,
        ) = lst

    elif len(lst) == 6:
        xlo_bound, xhi_bound, ylo_bound, yhi_bound, zlo_bound, zhi_bound = lst
        xy, xz, yz = 0.0, 0.0, 0.0
    else:
        raise ValueError("This format for amat not yet implemented: " + str(len(lst)))

    # > xhi_bound - xlo_bound = xhi -xlo  + MAX(0.0, xy, xz, xy + xz) - MIN(0.0, xy, xz, xy + xz)
    # > xhili = xhi -xlo   = xhi_bound - xlo_bound - MAX(0.0, xy, xz, xy + xz) + MIN(0.0, xy, xz, xy + xz)
    xhilo = (
        (xhi_bound - xlo_bound)
        - max([0.0, xy, xz, xy + xz])
        + min([0.0, xy, xz, xy + xz])
    )

    # > yhilo = yhi -ylo = yhi_bound -ylo_bound - MAX(0.0, yz) + MIN(0.0, yz)
    yhilo = (yhi_bound - ylo_bound) - max([0.0, yz]) + min([0.0, yz])

    # > zhi - zlo = zhi_bound- zlo_bound
    zhilo = zhi_bound - zlo_bound

    cell = [[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]]
    return cell


def parse_raw_dump_from_h5md(file_name: str) -> Dict:
    """
    Parse a LAMMPS dump file written in H5MD format.

    H5MD (HDF5 Molecular Dynamics) output is produced when LAMMPS is compiled
    with the H5MD package and the input script uses ``dump … h5md``.  The
    function reads positions, forces, simulation steps, and cell edge lengths
    from the standard H5MD path ``/particles/all/…``.

    Args:
        file_name (str): Path to the H5MD dump file (typically ``dump.h5``).

    Returns:
        dict: Dictionary with the following keys:

        - ``"forces"`` (list of list): Per-atom forces for each snapshot.
        - ``"positions"`` (list of list): Per-atom Cartesian positions for each snapshot.
        - ``"steps"`` (list of int): Simulation timestep indices.
        - ``"cells"`` (list of numpy.ndarray): 3×3 diagonal cell matrices for each snapshot.
    """
    import h5py

    with h5py.File(file_name, mode="r", libver="latest", swmr=True) as h5md:
        positions = [pos_i.tolist() for pos_i in h5md["/particles/all/position/value"]]
        steps = [steps_i.tolist() for steps_i in h5md["/particles/all/position/step"]]
        forces = [for_i.tolist() for for_i in h5md["/particles/all/force/value"]]
        # following the explanation at: http://nongnu.org/h5md/h5md.html
        cell = [
            np.eye(3) * np.array(cell_i.tolist())
            for cell_i in h5md["/particles/all/box/edges/value"]
        ]
    return {
        "forces": forces,
        "positions": positions,
        "steps": steps,
        "cells": cell,
    }


def _iter_raw_frames(
    file_name: str,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> Iterator[Dict]:
    """
    Yield one raw frame dict at a time from a LAMMPS text dump file.

    Each yielded dict has the same keys as DumpData fields, but values are
    for a single frame (scalars or arrays, not lists).

    Args:
        file_name: Path to the LAMMPS text dump file.
        start: First frame index to yield (0-based, default 0).
        stop: Stop before this frame index. None means read to end.
        step: Yield every ``step``-th frame (default 1).

    Yields:
        dict with keys: steps, natoms, cells, indices, forces, mean_forces,
        velocities, mean_velocities, unwrapped_positions,
        mean_unwrapped_positions, positions, computes.

    Raises:
        ValueError: If a frame is malformed or the file is truncated mid-frame.
    """
    frame_index = 0
    with open(file_name, "r") as f:
        line = f.readline()
        while line:
            if "ITEM: TIMESTEP" not in line:
                line = f.readline()
                continue

            # --- read header ---
            try:
                timestep = int(f.readline())
            except ValueError as e:
                raise ValueError(f"Malformed TIMESTEP at frame {frame_index}") from e

            line = f.readline()
            if "ITEM: NUMBER OF ATOMS" not in line:
                raise ValueError(
                    f"Expected NUMBER OF ATOMS at frame {frame_index}, got: {line!r}"
                )
            try:
                n = int(f.readline())
            except ValueError as e:
                raise ValueError(
                    f"Malformed NUMBER OF ATOMS at frame {frame_index}"
                ) from e

            line = f.readline()
            if "ITEM: BOX BOUNDS" not in line:
                raise ValueError(
                    f"Expected BOX BOUNDS at frame {frame_index}, got: {line!r}"
                )
            try:
                c1 = np.fromstring(f.readline(), dtype=float, sep=" ")
                c2 = np.fromstring(f.readline(), dtype=float, sep=" ")
                c3 = np.fromstring(f.readline(), dtype=float, sep=" ")
            except Exception as e:
                raise ValueError(f"Malformed BOX BOUNDS at frame {frame_index}") from e
            cell = to_amat(np.concatenate([c1, c2, c3]))

            line = f.readline()
            if "ITEM: ATOMS" not in line:
                raise ValueError(
                    f"Expected ITEM: ATOMS at frame {frame_index}, got: {line!r}"
                )
            columns = line.lstrip("ITEM: ATOMS").split()

            # --- decide whether to process this frame ---
            in_range = (
                frame_index >= start
                and (stop is None or frame_index < stop)
                and (frame_index - start) % step == 0
            )

            # --- read atom data ---
            if in_range:
                buf = StringIO()
                for i in range(n):
                    atom_line = f.readline()
                    if not atom_line:
                        raise ValueError(
                            f"Truncated dump file: expected {n} atoms at frame {frame_index} "
                            f"(step {timestep}), got {i}"
                        )
                    buf.write(atom_line)
                buf.seek(0)

                df = pd.read_csv(
                    buf,
                    nrows=n,
                    sep="\\s+",
                    header=None,
                    names=columns,
                    engine="c",
                )
                df.sort_values(by="id", ignore_index=True, inplace=True)

                frame: Dict = {
                    "steps": timestep,
                    "natoms": n,
                    "cells": cell,
                    "indices": df["type"].array.astype(int),
                    "forces": (
                        np.stack(
                            [df["fx"].array, df["fy"].array, df["fz"].array], axis=1
                        )
                        if all(c in columns for c in ("fx", "fy", "fz"))
                        else np.array([])
                    ),
                    "mean_forces": (
                        np.stack(
                            [
                                df["f_mean_forces[1]"].array,
                                df["f_mean_forces[2]"].array,
                                df["f_mean_forces[3]"].array,
                            ],
                            axis=1,
                        )
                        if "f_mean_forces[1]" in columns
                        else np.array([])
                    ),
                    "velocities": (
                        np.stack(
                            [df["vx"].array, df["vy"].array, df["vz"].array], axis=1
                        )
                        if all(c in columns for c in ("vx", "vy", "vz"))
                        else np.array([])
                    ),
                    "mean_velocities": (
                        np.stack(
                            [
                                df["f_mean_velocities[1]"].array,
                                df["f_mean_velocities[2]"].array,
                                df["f_mean_velocities[3]"].array,
                            ],
                            axis=1,
                        )
                        if "f_mean_velocities[1]" in columns
                        else np.array([])
                    ),
                    "computes": {},
                }

                if "xsu" in columns:
                    direct = np.stack(
                        [df["xsu"].array, df["ysu"].array, df["zsu"].array], axis=1
                    )
                    frame["unwrapped_positions"] = direct
                    frame["positions"] = direct - np.floor(direct)
                else:
                    frame["unwrapped_positions"] = np.array([])
                    frame["positions"] = np.array([])

                if "f_mean_positions[1]" in columns:
                    frame["mean_unwrapped_positions"] = np.stack(
                        [
                            df["f_mean_positions[1]"].array,
                            df["f_mean_positions[2]"].array,
                            df["f_mean_positions[3]"].array,
                        ],
                        axis=1,
                    )
                else:
                    frame["mean_unwrapped_positions"] = np.array([])

                for k in columns:
                    if k.startswith("c_"):
                        frame["computes"][k.replace("c_", "")] = df[k].array

                yield frame

            else:
                # skip atom lines cheaply without building a buffer
                for _ in range(n):
                    f.readline()
                # early exit if we've passed stop
                if stop is not None and frame_index >= stop:
                    return

            frame_index += 1
            line = f.readline()


def parse_raw_dump_from_text(file_name: str) -> Dict:
    """
    Parse a LAMMPS custom text dump file into a structured dictionary.

    The file must have been created with a ``dump … custom`` command that writes
    at least the columns ``id``, ``type``, ``xsu``, ``ysu``, ``zsu``, ``fx``,
    ``fy``, ``fz``.  Optional columns ``vx``/``vy``/``vz`` (velocities),
    ``f_mean_forces[1-3]``, ``f_mean_velocities[1-3]``, ``f_mean_positions[1-3]``
    (time-averaged fixes), and any per-atom compute columns starting with ``c_``
    are handled automatically.

    Coordinates are expected as scaled (fractional) unwrapped positions
    (``xsu``/``ysu``/``zsu``).  Wrapped fractional positions are derived by
    taking ``xsu - floor(xsu)``.

    Args:
        file_name (str): Path to the LAMMPS text dump file (typically ``dump.out``).

    Returns:
        dict: Dictionary with the following keys (empty lists when the
        corresponding columns were absent from the dump):

        - ``"steps"`` (list of int): Simulation timestep indices.
        - ``"natoms"`` (list of int): Number of atoms at each snapshot.
        - ``"cells"`` (list of list): 3×3 cell matrices (one per snapshot).
        - ``"indices"`` (list of numpy.ndarray): Integer LAMMPS atom-type indices (1-based).
        - ``"forces"`` (list of numpy.ndarray): Per-atom force vectors, shape ``(N, 3)``.
        - ``"mean_forces"`` (list of numpy.ndarray): Time-averaged forces from ``fix ave/atom``.
        - ``"velocities"`` (list of numpy.ndarray): Per-atom velocity vectors, shape ``(N, 3)``.
        - ``"mean_velocities"`` (list of numpy.ndarray): Time-averaged velocities.
        - ``"unwrapped_positions"`` (list of numpy.ndarray): Unwrapped fractional coordinates, shape ``(N, 3)``.
        - ``"mean_unwrapped_positions"`` (list of numpy.ndarray): Time-averaged unwrapped positions.
        - ``"positions"`` (list of numpy.ndarray): Wrapped fractional coordinates, shape ``(N, 3)``.
        - ``"computes"`` (dict): Per-atom compute results keyed by compute ID (``c_`` prefix stripped).
    """
    dump = DumpData()
    for frame in _iter_raw_frames(file_name):
        dump.steps.append(frame["steps"])
        dump.natoms.append(frame["natoms"])
        dump.cells.append(frame["cells"])
        dump.indices.append(frame["indices"])
        if len(frame["forces"]):
            dump.forces.append(frame["forces"])
        if len(frame["mean_forces"]):
            dump.mean_forces.append(frame["mean_forces"])
        if len(frame["velocities"]):
            dump.velocities.append(frame["velocities"])
        if len(frame["mean_velocities"]):
            dump.mean_velocities.append(frame["mean_velocities"])
        if len(frame["unwrapped_positions"]):
            dump.unwrapped_positions.append(frame["unwrapped_positions"])
            dump.positions.append(frame["positions"])
        if len(frame["mean_unwrapped_positions"]):
            dump.mean_unwrapped_positions.append(frame["mean_unwrapped_positions"])
        for k, v in frame["computes"].items():
            if k not in dump.computes:
                dump.computes[k] = []
            dump.computes[k].append(v)
    return asdict(dump)


def parse_raw_lammps_log(file_name: str) -> pd.DataFrame:
    """
    Parse the thermodynamic output from a LAMMPS log file.

    LAMMPS writes thermo output to ``log.lammps`` (or a user-specified log
    file) between a header line that starts with ``Step`` and a footer line
    that starts with ``Loop``.  Multiple ``run`` commands in a single input
    script produce multiple such blocks; all blocks are concatenated and an
    extra ``LogStep`` column is added to identify which run each row belongs
    to.

    Lines beginning with ``WARNING:`` inside a thermo block are forwarded as
    Python :func:`warnings.warn` calls.  Lines beginning with ``ERROR`` cause
    the current thermo block to be closed (the error is not re-raised here,
    but the resulting DataFrame will be truncated).

    Args:
        file_name (str): Path to the LAMMPS log file (typically ``log.lammps``).

    Returns:
        pandas.DataFrame: DataFrame whose columns match the thermo keywords
        used in the LAMMPS ``thermo_style custom`` command (e.g. ``Step``,
        ``Temp``, ``PotEng``, ``TotEng``, ``Pxx``, …).  When multiple runs
        are present a ``LogStep`` column (integer run index, 0-based) is
        appended.
    """
    with open(file_name, "r") as f:
        dfs = []
        read_thermo = False
        for line in f:
            line = line.lstrip()

            if line.startswith("Step"):
                thermo_lines = ""
                read_thermo = True

            if read_thermo:
                if line.startswith("Loop") or line.startswith("ERROR"):
                    read_thermo = False
                    dfs.append(
                        pd.read_csv(StringIO(thermo_lines), sep="\\s+", engine="c")
                    )

                elif line.startswith("WARNING:"):
                    warnings.warn(f"A warning was found in the log:\n{line}")

                else:
                    thermo_lines += line

    if len(dfs) == 1:
        df = dfs[0]
    else:
        for i in range(len(dfs)):
            df = dfs[i]
            df["LogStep"] = np.ones(len(df)) * i
        df = pd.concat(dfs, ignore_index=True)
    return df
