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
) -> Dict[str, Dict[str, Any]]:
    """
    Parse all output files from a finished LAMMPS calculation.

    Looks for a dump file (H5MD format preferred, text format as fallback) and
    a log file in ``working_directory``, converts all quantities from the
    LAMMPS unit system to pyiron/ASE units (Å, eV, ps, …), and returns the
    results in a nested dictionary that can be stored directly in an HDF5 file.

    The function resolves the LAMMPS triclinic-to-orthogonal coordinate
    rotation via :class:`~lammpsparser.structure.UnfoldingPrism` and re-maps
    atom-type indices back to structure indices via ``remap_indices_funct``.

    Args:
        working_directory (str): Directory that contains the LAMMPS output files.
        structure (ase.atoms.Atoms): The input structure used for the calculation.
            Required for index remapping and prism construction.
        potential_elements (numpy.ndarray or list): Ordered list of chemical
            symbols as they appear in the LAMMPS potential definition.  The
            order determines the mapping from LAMMPS integer type IDs to
            element symbols.
        units (str): LAMMPS unit system used in the calculation (e.g.
            ``"metal"``, ``"real"``, ``"si"``, ``"cgs"``, ``"electron"``).
        prism (UnfoldingPrism, optional): Pre-constructed prism object.  If
            ``None`` (default) a new one is built from ``structure.cell``.
        dump_h5_file_name (str): Filename of the H5MD dump inside
            ``working_directory`` (default: ``"dump.h5"``).
        dump_out_file_name (str): Filename of the text dump inside
            ``working_directory`` (default: ``"dump.out"``).
        log_lammps_file_name (str): Filename of the LAMMPS log file inside
            ``working_directory`` (default: ``"log.lammps"``).
        remap_indices_funct (callable): Function used to map LAMMPS integer
            type indices onto structure indices.  Defaults to
            :func:`remap_indices_ase`.

    Returns:
        dict: Nested dictionary with two top-level keys:

        - ``"generic"`` – quantities stored in pyiron/ASE units:
          ``"steps"``, ``"cells"``, ``"positions"``, ``"forces"``,
          ``"velocities"``, ``"indices"``, ``"temperature"``,
          ``"energy_pot"``, ``"energy_tot"``, ``"volume"``,
          ``"pressures"`` (if available), and any per-atom computes.
        - ``"lammps"`` – raw LAMMPS-specific thermo columns that have no
          generic equivalent.

    Raises:
        FileNotFoundError: If neither the H5MD nor the text dump file exists.
    """
    if prism is None:
        prism = UnfoldingPrism(structure.cell)
    dump_dict = _parse_dump(
        dump_h5_full_file_name=os.path.join(working_directory, dump_h5_file_name),
        dump_out_full_file_name=os.path.join(working_directory, dump_out_file_name),
        prism=prism,
        structure=structure,
        potential_elements=potential_elements,
        remap_indices_funct=remap_indices_funct,
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
) -> Dict[str, Any]:
    """
    Dispatch dump parsing to the correct reader (H5MD or text).

    Prefers the H5MD file when it exists.  H5MD files do not require
    coordinate rotation, but the prism must be orthorhombic; a
    :exc:`RuntimeError` is raised when that condition is violated.

    Args:
        dump_h5_full_file_name (str): Absolute path to the H5MD dump file.
        dump_out_full_file_name (str): Absolute path to the text dump file.
        prism (UnfoldingPrism): Prism object for coordinate transformations.
        structure (ase.atoms.Atoms): Input structure for index remapping.
        potential_elements (numpy.ndarray or list): Ordered element list for index remapping.
        remap_indices_funct (callable): Index remapping function.

    Returns:
        dict: Raw dump data dictionary as returned by the underlying parser.

    Raises:
        RuntimeError: If an H5MD file is present but the prism is not orthorhombic.
        FileNotFoundError: If neither dump file exists.
    """
    if os.path.isfile(dump_h5_full_file_name):
        if not _check_ortho_prism(prism=prism):
            raise RuntimeError(
                "The Lammps output will not be mapped back to pyiron correctly."
            )
        return parse_raw_dump_from_h5md(
            file_name=dump_h5_full_file_name,
        )
    elif os.path.exists(dump_out_full_file_name):
        return _collect_dump_from_text(
            file_name=dump_out_full_file_name,
            prism=prism,
            structure=structure,
            potential_elements=potential_elements,
            remap_indices_funct=remap_indices_funct,
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
) -> Dict[str, Any]:
    """
    Post-process a raw text dump dictionary: rotate vectors and remap indices.

    Applies the inverse of the LAMMPS coordinate rotation to forces, velocities,
    and positions so that all vectors are expressed in the original ASE cell
    frame.  Cell matrices are unfolded via
    :meth:`~lammpsparser.structure.UnfoldingPrism.unfold_cell` and atom-type
    indices are remapped from LAMMPS integer IDs to structure indices.

    Args:
        file_name (str): Path to the LAMMPS text dump file.
        prism (UnfoldingPrism): Prism built from the input structure's cell.
        structure (ase.atoms.Atoms): Input structure used for index remapping.
        potential_elements (numpy.ndarray or list): Ordered element list.
        remap_indices_funct (callable): Function mapping LAMMPS type IDs to
            structure indices (default: :func:`remap_indices_ase`).

    Returns:
        dict: Processed dump dictionary with the same keys as
        :func:`~lammpsparser.output_raw.parse_raw_dump_from_text` but with
        all vector quantities rotated to the ASE frame and Cartesian positions
        computed from fractional coordinates.
    """
    rotation_lammps2orig = prism.R.T
    dump_lammps_dict = parse_raw_dump_from_text(file_name=file_name)
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
    Parse the LAMMPS log file and organise thermo data into generic and pressure outputs.

    Renames standard LAMMPS thermo column names to pyiron equivalents
    (e.g. ``Temp`` → ``temperature``, ``PotEng`` → ``energy_pot``).
    If all six independent Voigt pressure components (``Pxx``, ``Pyy``,
    ``Pzz``, ``Pxy``, ``Pxz``, ``Pyz``) are present they are assembled into
    a symmetric 3×3 pressure tensor per frame and rotated to the ASE frame
    when the prism rotation is non-trivial.  Time-averaged pressures from
    ``fix ave/time`` (``mean_pressure[1-6]``) are handled analogously.

    Args:
        file_name (str): Path to the LAMMPS log file.
        prism (UnfoldingPrism): Prism object used to rotate the pressure
            tensor when the simulation cell is triclinic.

    Returns:
        tuple:
            - list[str]: Column names that belong in the ``"generic"`` output
              dictionary.
            - dict: Pressure arrays keyed by ``"pressures"`` and/or
              ``"mean_pressures"``, shape ``(N_frames, 3, 3)``.
            - pandas.DataFrame: Remaining thermo data (pressure columns
              removed) with renamed columns.
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
