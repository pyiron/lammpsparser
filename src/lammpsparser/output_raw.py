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


def parse_raw_dump_from_h5md(file_name: str, last_frame_only: bool = False) -> Dict:
    import h5py

    with h5py.File(file_name, mode="r", libver="latest", swmr=True) as h5md:
        sl = slice(-1, None) if last_frame_only else slice(None)
        positions = [pos_i.tolist() for pos_i in h5md["/particles/all/position/value"][sl]]
        steps = [steps_i.tolist() for steps_i in h5md["/particles/all/position/step"][sl]]
        forces = [for_i.tolist() for for_i in h5md["/particles/all/force/value"][sl]]
        # following the explanation at: http://nongnu.org/h5md/h5md.html
        cell = [
            np.eye(3) * np.array(cell_i.tolist())
            for cell_i in h5md["/particles/all/box/edges/value"][sl]
        ]
    return {
        "forces": forces,
        "positions": positions,
        "steps": steps,
        "cells": cell,
    }


def parse_raw_dump_from_text(file_name: str, last_frame_only: bool = False) -> Dict:
    """
    Docstring for _parse_dump_from_text

    Args:
        file_name (str): The path to the lammps dump file.
        last_frame_only (bool): If True, only the last frame is returned.

    Returns:
        Dict: Parsed dump data.
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

        if last_frame_only:
            result = asdict(dump)
            for k, v in result.items():
                if k == "computes":
                    result[k] = {ck: cv[-1:] for ck, cv in v.items()}
                elif isinstance(v, list) and len(v) > 0:
                    result[k] = v[-1:]
            return result
        return asdict(dump)


def parse_raw_lammps_log(file_name: str) -> pd.DataFrame:
    """
    Docstring for _parse_lammps_log

    Args:
        file_name (str): The path to the lammps log file.

    Returns:
        pd.DataFrame: Dataframe containing the parsed log data.
    """
    with open(file_name, "r") as f:
        dfs = []
        read_thermo = False
        for l in f:
            l = l.lstrip()

            if l.startswith("Step"):
                thermo_lines = ""
                read_thermo = True

            if read_thermo:
                if l.startswith("Loop") or l.startswith("ERROR"):
                    read_thermo = False
                    dfs.append(
                        pd.read_csv(StringIO(thermo_lines), sep="\\s+", engine="c")
                    )

                elif l.startswith("WARNING:"):
                    warnings.warn(f"A warning was found in the log:\n{l}")

                else:
                    thermo_lines += l

    if len(dfs) == 1:
        df = dfs[0]
    else:
        for i in range(len(dfs)):
            df = dfs[i]
            df["LogStep"] = np.ones(len(df)) * i
        df = pd.concat(dfs, ignore_index=True)
    return df
