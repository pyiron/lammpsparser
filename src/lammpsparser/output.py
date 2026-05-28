from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ase.atoms import Atoms

from lammpsparser.output_raw import (
    parse_raw_dump_from_h5md,
    parse_raw_dump_from_text,
    parse_raw_lammps_log,
)
from lammpsparser.structure import UnfoldingPrism
from lammpsparser.units import UnitConverter


def remap_indices_ase(
    lammps_indices: Union[np.ndarray, List],
    potential_elements: Union[np.ndarray, List],
    structure: Atoms,
) -> np.ndarray:
    """
    Give the Lammps-dumped indices, re-maps these back onto the structure's indices to preserve the species.

    The issue is that for an N-element potential, Lammps dumps the chemical index from 1 to N based on the order
    that these species are written in the Lammps input file. But the indices for a given structure are based on the
    order in which chemical species were added to that structure, and run from 0 up to the number of species
    currently in that structure. Therefore we need to be a little careful with mapping.

    Args:
        lammps_indices (numpy.ndarray/list): The Lammps-dumped integers.
        potential_elements (numpy.ndarray/list):
        structure (pyiron_atomistics.atomistics.structure.Atoms):

    Returns:
        numpy.ndarray: Those integers mapped onto the structure.
    """
    lammps_symbol_order = np.array(potential_elements)

    # Create a map between the lammps indices and structure indices to preserve species
    structure_symbol_order = np.unique(structure.get_chemical_symbols())
    map_ = np.array(
        [
            int(np.argwhere(lammps_symbol_order == symbol)[0][0]) + 1
            for symbol in structure_symbol_order
        ]
    )

    structure_indices = np.array(lammps_indices)
    for i_struct, i_lammps in enumerate(map_):
        np.place(structure_indices, lammps_indices == i_lammps, i_struct)
    # TODO: Vectorize this for-loop for computational efficiency

    return structure_indices


def parse_lammps_output(
    working_directory: str,
    structure: Atoms,
    potential_elements: Union[np.ndarray, List],
    units: str,
    prism: Optional[UnfoldingPrism] = None,
    dump_h5_file_name: str = "dump.h5",
    dump_out_file_name: str = "dump.out",
    log_lammps_file_name: str = "log.lammps",
    remap_indices_funct: Callable[..., np.ndarray] = remap_indices_ase,
    last_frame_only: bool = False,
) -> Dict[str, Dict[str, Any]]:
    if prism is None:
        prism = UnfoldingPrism(structure.cell)
    dump_dict = _parse_dump(
        dump_h5_full_file_name=os.path.join(working_directory, dump_h5_file_name),
        dump_out_full_file_name=os.path.join(working_directory, dump_out_file_name),
        prism=prism,
        structure=structure,
        potential_elements=potential_elements,
        remap_indices_funct=remap_indices_funct,
        last_frame_only=last_frame_only,
    )

    generic_keys_lst, pressure_dict, df = _parse_log(
        log_lammps_full_file_name=os.path.join(working_directory, log_lammps_file_name),
        prism=prism,
    )

    convert_units = UnitConverter(units).convert_array_to_pyiron_units

    hdf_output: Dict[str, Dict[str, Any]] = {"generic": {}, "lammps": {}}
    hdf_generic = hdf_output["generic"]
    hdf_lammps = hdf_output["lammps"]

    if "computes" in dump_dict.keys():
        for k, v in dump_dict.pop("computes").items():
            hdf_generic[k] = convert_units(np.array(v), label=k)

    hdf_generic["steps"] = convert_units(
        np.array(dump_dict.pop("steps"), dtype=int), label="steps"
    )

    for k, v in dump_dict.items():
        if len(v) > 0:
            try:
                hdf_generic[k] = convert_units(np.array(v), label=k)
            except ValueError:
                hdf_generic[k] = [convert_units(np.array(val), label=k) for val in v]

    if df is not None and pressure_dict is not None and generic_keys_lst is not None:
        for k, v in df.items():
            v = convert_units(np.array(v), label=k)
            if k in generic_keys_lst:
                hdf_generic[k] = v
            else:  # This is a hack for backward comparability
                hdf_lammps[k] = v

        # Store pressures as numpy arrays
        for key, val in pressure_dict.items():
            hdf_generic[key] = convert_units(val, label=key)
    else:
        warnings.warn("LAMMPS warning: No log.lammps output file found.")

    return hdf_output


def _parse_dump(
    dump_h5_full_file_name: str,
    dump_out_full_file_name: str,
    prism: UnfoldingPrism,
    structure: Atoms,
    potential_elements: Union[np.ndarray, List],
    remap_indices_funct: Callable[..., np.ndarray] = remap_indices_ase,
    last_frame_only: bool = False,
) -> Dict[str, Any]:
    if os.path.isfile(dump_h5_full_file_name):
        if not _check_ortho_prism(prism=prism):
            raise RuntimeError(
                "The Lammps output will not be mapped back to pyiron correctly."
            )
        return parse_raw_dump_from_h5md(
            file_name=dump_h5_full_file_name,
            last_frame_only=last_frame_only,
        )
    elif os.path.exists(dump_out_full_file_name):
        return _collect_dump_from_text(
            file_name=dump_out_full_file_name,
            prism=prism,
            structure=structure,
            potential_elements=potential_elements,
            remap_indices_funct=remap_indices_funct,
            last_frame_only=last_frame_only,
        )
    else:
        raise FileNotFoundError(
            f"Neither {dump_h5_full_file_name} nor {dump_out_full_file_name} exist."
        )


def _collect_dump_from_text(
    file_name: str,
    prism: UnfoldingPrism,
    structure: Atoms,
    potential_elements: Union[np.ndarray, List],
    remap_indices_funct: Callable[..., np.ndarray] = remap_indices_ase,
    last_frame_only: bool = False,
) -> Dict[str, Any]:
    """
    general purpose routine to extract static from a lammps dump file
    """
    rotation_lammps2orig = prism.R.T
    dump_lammps_dict = parse_raw_dump_from_text(file_name=file_name, last_frame_only=last_frame_only)
    dump_dict: Dict[str, Any] = {}
    for key, val in dump_lammps_dict.items():
        if key in ["cells"]:
            dump_dict[key] = [prism.unfold_cell(cell=cell) for cell in val]
        elif key in ["indices"]:
            dump_dict[key] = [
                remap_indices_funct(
                    lammps_indices=indices,
                    potential_elements=potential_elements,
                    structure=structure,
                )
                for indices in val
            ]
        elif key in [
            "forces",
            "mean_forces",
            "velocities",
            "mean_velocities",
            "mean_unwrapped_positions",
        ]:
            dump_dict[key] = [np.matmul(v, rotation_lammps2orig) for v in val]
        elif key in ["positions", "unwrapped_positions"]:
            dump_dict[key] = [
                np.matmul(np.matmul(v, lammps_cell), rotation_lammps2orig)
                for v, lammps_cell in zip(val, dump_lammps_dict["cells"])
            ]
        else:
            dump_dict[key] = val
    return dump_dict


def _parse_log(
    log_lammps_full_file_name: str, prism: UnfoldingPrism
) -> Union[Tuple[List[str], Dict, pd.DataFrame], Tuple[None, None, None]]:
    """
    If it exists, parses the lammps log file and either raises an exception if errors
    occurred or returns data. Just returns a tuple of Nones if there is no file at the
    given location.

    Args:
        log_lammps_full_file_name (str): The path to the lammps log file.
        prism (pyiron_atomistics.lammps.structure.UnfoldingPrism): For mapping between
            lammps and pyiron structures

    Returns:
        (list | None): Generic keys
        (dict | None): Pressures
        (pandas.DataFrame | None): A dataframe with the rest of the information

    Raises:
        (RuntimeError): If there are "ERROR" tags in the log.
    """
    if os.path.exists(log_lammps_full_file_name):
        return _collect_output_log(
            file_name=log_lammps_full_file_name,
            prism=prism,
        )
    else:
        return None, None, None


def _collect_output_log(
    file_name: str, prism: UnfoldingPrism
) -> Tuple[List[str], Dict, pd.DataFrame]:
    """
    general purpose routine to extract static from a lammps log file
    """
    df = parse_raw_lammps_log(file_name=file_name)

    h5_dict = {
        "Step": "steps",
        "Temp": "temperature",
        "PotEng": "energy_pot",
        "TotEng": "energy_tot",
        "Volume": "volume",
        "LogStep": "LogStep",
    }
    if "LogStep" not in df.columns:
        del h5_dict["LogStep"]

    for key in df.columns[df.columns.str.startswith("f_mean")]:
        h5_dict[key] = key.replace("f_", "")

    df = df.rename(index=str, columns=h5_dict)
    pressure_dict = dict()
    if all(
        [
            x in df.columns.values
            for x in [
                "Pxx",
                "Pxy",
                "Pxz",
                "Pxy",
                "Pyy",
                "Pyz",
                "Pxz",
                "Pyz",
                "Pzz",
            ]
        ]
    ):
        pressures = (
            np.stack(
                (
                    df.Pxx,
                    df.Pxy,
                    df.Pxz,
                    df.Pxy,
                    df.Pyy,
                    df.Pyz,
                    df.Pxz,
                    df.Pyz,
                    df.Pzz,
                ),
                axis=-1,
            )
            .reshape(-1, 3, 3)
            .astype("float64")
        )
        # Rotate pressures from Lammps frame to pyiron frame if necessary
        if _check_ortho_prism(prism=prism):
            rotation_matrix = prism.R.T
            pressures = rotation_matrix.T @ pressures @ rotation_matrix

        df = df.drop(
            columns=df.columns[
                ((df.columns.str.len() == 3) & df.columns.str.startswith("P"))
            ]
        )
        pressure_dict["pressures"] = pressures
    else:
        warnings.warn(
            "LAMMPS warning: log.lammps does not contain the required pressure values."
        )
    if "mean_pressure[1]" in df.columns:
        pressures = (
            np.stack(
                tuple(df[f"mean_pressure[{i}]"] for i in [1, 4, 5, 4, 2, 6, 5, 6, 3]),
                axis=-1,
            )
            .reshape(-1, 3, 3)
            .astype("float64")
        )
        if _check_ortho_prism(prism=prism):
            rotation_matrix = prism.R.T
            pressures = rotation_matrix.T @ pressures @ rotation_matrix
        df = df.drop(
            columns=df.columns[
                (
                    df.columns.str.startswith("mean_pressure")
                    & df.columns.str.endswith("]")
                )
            ]
        )
        pressure_dict["mean_pressures"] = pressures
    generic_keys_lst = list(h5_dict.values())
    return generic_keys_lst, pressure_dict, df


def _check_ortho_prism(
    prism: UnfoldingPrism, rtol: float = 0.0, atol: float = 1e-08
) -> bool:
    """
    Check if the rotation matrix of the UnfoldingPrism object is sufficiently close to a unit matrix

    Args:
        prism (pyiron_atomistics.lammps.structure.UnfoldingPrism): UnfoldingPrism object to check
        rtol (float): relative precision for numpy.isclose()
        atol (float): absolute precision for numpy.isclose()

    Returns:
        boolean: True or False
    """
    return bool(np.isclose(prism.R, np.eye(3), rtol=rtol, atol=atol).all())
