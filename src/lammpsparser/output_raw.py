from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Dict, List, Union

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
    with open(file_name, "r") as f:
        dump = DumpData()

        for line in f:
            if "ITEM: TIMESTEP" in line:
                dump.steps.append(int(f.readline()))

            elif "ITEM: BOX BOUNDS" in line:
                c1 = np.fromstring(f.readline(), dtype=float, sep=" ")
                c2 = np.fromstring(f.readline(), dtype=float, sep=" ")
                c3 = np.fromstring(f.readline(), dtype=float, sep=" ")
                cell = np.concatenate([c1, c2, c3])
                dump.cells.append(to_amat(cell))

            elif "ITEM: NUMBER OF ATOMS" in line:
                n = int(f.readline())
                dump.natoms.append(n)

            elif "ITEM: ATOMS" in line:
                # get column names from line
                columns = line.lstrip("ITEM: ATOMS").split()

                # Read line by line of snapshot into a string buffer
                # Than parse using pandas for speed and column acces
                buf = StringIO()
                for _ in range(n):
                    buf.write(f.readline())
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
                # Coordinate transform lammps->pyiron
                dump.indices.append(df["type"].array.astype(int))

                dump.forces.append(
                    np.stack([df["fx"].array, df["fy"].array, df["fz"].array], axis=1)
                )
                if "f_mean_forces[1]" in columns:
                    dump.mean_forces.append(
                        np.stack(
                            [
                                df["f_mean_forces[1]"].array,
                                df["f_mean_forces[2]"].array,
                                df["f_mean_forces[3]"].array,
                            ],
                            axis=1,
                        )
                    )
                if "vx" in columns and "vy" in columns and "vz" in columns:
                    dump.velocities.append(
                        np.stack(
                            [
                                df["vx"].array,
                                df["vy"].array,
                                df["vz"].array,
                            ],
                            axis=1,
                        )
                    )

                if "f_mean_velocities[1]" in columns:
                    dump.mean_velocities.append(
                        np.stack(
                            [
                                df["f_mean_velocities[1]"].array,
                                df["f_mean_velocities[2]"].array,
                                df["f_mean_velocities[3]"].array,
                            ],
                            axis=1,
                        )
                    )

                if "xsu" in columns:
                    direct_unwrapped_positions = np.stack(
                        [
                            df["xsu"].array,
                            df["ysu"].array,
                            df["zsu"].array,
                        ],
                        axis=1,
                    )
                    dump.unwrapped_positions.append(direct_unwrapped_positions)
                    dump.positions.append(
                        direct_unwrapped_positions
                        - np.floor(direct_unwrapped_positions)
                    )

                if "f_mean_positions[1]" in columns:
                    dump.mean_unwrapped_positions.append(
                        np.stack(
                            [
                                df["f_mean_positions[1]"].array,
                                df["f_mean_positions[2]"].array,
                                df["f_mean_positions[3]"].array,
                            ],
                            axis=1,
                        )
                    )
                for k in columns:
                    if k.startswith("c_"):
                        kk = k.replace("c_", "")
                        if kk not in dump.computes.keys():
                            dump.computes[kk] = []
                        dump.computes[kk].append(df[k].array)

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
