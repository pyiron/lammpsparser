"""
Minimal neighbour analysis for the compatibility layer.

This is a reduced port of :mod:`structuretoolkit.analyse.neighbors` (originally by
Joerg Neugebauer and Sam Waseda, Max-Planck-Institut für Eisenforschung GmbH,
distributed under the "New BSD License"). Only the functionality required to
derive bonds and angles for the LAMMPS ``bond`` and ``full`` atom styles is
included.
"""

from __future__ import annotations

import warnings

import numpy as np
from ase.atoms import Atoms
from scipy.spatial import cKDTree


def select_index(structure: Atoms, element: str) -> np.ndarray:
    """
    Select the indices of atoms in the structure that belong to a specific element.

    Args:
        structure (ase.atoms.Atoms): Atomic structure.
        element (str): Chemical symbol.

    Returns:
        numpy.ndarray: Indices of the atoms of the given element.
    """
    return structure.symbols.indices()[element]


def get_vertical_length(structure: Atoms) -> np.ndarray:
    """
    Length of each cell vector projected on the normal of the plane spanned by the
    other two cell vectors.

    Args:
        structure (ase.atoms.Atoms): Atomic structure.

    Returns:
        numpy.ndarray: Projected cell lengths.
    """
    return np.linalg.det(structure.cell) / np.linalg.norm(
        np.cross(
            np.roll(structure.cell, -1, axis=0), np.roll(structure.cell, 1, axis=0)
        ),
        axis=-1,
    )


def get_extended_positions(
    structure: Atoms, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the positions of all atoms in the supercell plus all periodic images within
    ``width`` of the cell boundary.

    Args:
        structure (ase.atoms.Atoms): Atomic structure.
        width (float): Width of the buffer layer on every periodic side of the box.

    Returns:
        tuple: Extended positions and the indices of the corresponding atoms in the
        original structure.
    """
    if width < 0:
        raise ValueError("Invalid width")
    positions = structure.positions
    if width == 0:
        return positions, np.arange(len(positions))
    scaled_width = width / get_vertical_length(structure=structure)
    rep = 2 * np.ceil(scaled_width).astype(int) * structure.pbc + 1
    rep = [np.arange(r) - int(r / 2) for r in rep]
    meshgrid = np.stack(np.meshgrid(rep[0], rep[1], rep[2]), axis=-1).reshape(-1, 3)
    v_repeated = np.einsum("ni,ij->nj", meshgrid, structure.cell)
    v_repeated = v_repeated[:, np.newaxis, :] + positions[np.newaxis, :, :]
    v_repeated = v_repeated.reshape(-1, 3)
    indices = np.tile(np.arange(len(positions)), len(meshgrid))
    dist = v_repeated - np.sum(structure.cell * 0.5, axis=0)
    dist = (
        np.absolute(np.einsum("ni,ij->nj", dist + 1e-8, np.linalg.inv(structure.cell)))
        - 0.5
    )
    check_dist = np.all(dist - scaled_width < 0, axis=-1)
    return v_repeated[check_dist], indices[check_dist] % len(positions)


class Neighbors:
    """
    Neighbourhood information of each atom in a structure.

    Attributes:
        distances (numpy.ndarray): Distances to the neighbours of each atom. Missing
            entries (beyond the cutoff radius) are filled with ``numpy.inf``.
        indices (numpy.ndarray): Indices of the neighbours of each atom. Missing
            entries are filled with ``numpy.iinfo(numpy.int32).max``.
    """

    def __init__(self, structure: Atoms, distances: np.ndarray, indices: np.ndarray):
        self._structure = structure
        self.distances = distances
        self.indices = indices

    def get_bonds(
        self,
        radius: float = np.inf,
        max_shells: int | None = None,
        prec: float = 0.1,
    ) -> list[dict[str, list[list[int]]]]:
        """
        Group the neighbours of each atom into shells and chemical species.

        Args:
            radius (float): Maximum distance for a bond to be considered.
            max_shells (int, optional): Maximum number of shells per species.
            prec (float): Minimum distance gap between two shells; shells closer
                than ``prec`` are merged into one.

        Returns:
            list: For each atom a dictionary mapping the chemical symbol of the
            neighbours to a list of shells, each shell being a list of atom indices.
        """
        el_list = self._structure.get_chemical_symbols()
        ind_shell = []
        for d, i in zip(self.distances, self.indices):
            cond = d < radius
            dist_vec, ind_vec = d[cond], i[cond]
            ind_where = np.where(np.diff(dist_vec) > prec)[0] + 1
            ia_shells_dict: dict[str, list[list[int]]] = {}
            for i_shell_list in [np.sort(g) for g in np.split(ind_vec, ind_where)]:
                ia_shell_dict: dict[str, list[int]] = {}
                for i_s in i_shell_list:
                    ia_shell_dict.setdefault(el_list[i_s], []).append(i_s)
                for el, ia_lst in ia_shell_dict.items():
                    shells = ia_shells_dict.setdefault(el, [])
                    if max_shells is not None and len(shells) + 1 > max_shells:
                        continue
                    shells.append(ia_lst)
            ind_shell.append(ia_shells_dict)
        return ind_shell


def _estimate_width(
    structure: Atoms,
    num_neighbors: int,
    cutoff_radius: float = np.inf,
    width_buffer: float = 1.2,
) -> float:
    """
    Estimate the width of the periodic buffer layer required to find all neighbours.

    Args:
        structure (ase.atoms.Atoms): Atomic structure.
        num_neighbors (int): Number of neighbours.
        cutoff_radius (float): Cutoff radius.
        width_buffer (float): Relative safety margin.

    Returns:
        float: Width of the buffer layer.
    """
    if not any(structure.pbc):
        return 0
    elif cutoff_radius != np.inf:
        return cutoff_radius
    width = np.prod(np.linalg.norm(structure.cell, axis=-1))
    width *= 4 / 3 * np.pi * np.max([num_neighbors, 8]) / len(structure)
    return width_buffer * width ** (1 / 3)


def get_neighbors(
    structure: Atoms,
    num_neighbors: int = 12,
    cutoff_radius: float = np.inf,
    width_buffer: float = 1.2,
) -> Neighbors:
    """
    Get the neighbours of each atom in a structure, taking periodic boundary
    conditions into account.

    Args:
        structure (ase.atoms.Atoms): Atomic structure.
        num_neighbors (int): Maximum number of neighbours per atom.
        cutoff_radius (float): Upper bound of the neighbour distance.
        width_buffer (float): Relative safety margin of the periodic buffer layer.

    Returns:
        Neighbors: Neighbour distances and indices.
    """
    if num_neighbors <= 0:
        raise ValueError("invalid number of neighbors")
    if width_buffer < 0:
        raise ValueError("width_buffer must be a positive float")
    width = _estimate_width(
        structure=structure,
        num_neighbors=num_neighbors,
        cutoff_radius=cutoff_radius,
        width_buffer=width_buffer,
    )
    extended_positions, wrapped_indices = get_extended_positions(
        structure=structure, width=width
    )
    if len(extended_positions) < num_neighbors + 1 and cutoff_radius == np.inf:
        raise ValueError(
            "num_neighbors too large - make width_buffer larger and/or make "
            + "num_neighbors smaller"
        )
    distances, indices = cKDTree(extended_positions).query(
        structure.positions,
        k=num_neighbors + 1,
        distance_upper_bound=cutoff_radius,
    )
    distances = np.asarray(distances).reshape(len(structure), num_neighbors + 1)
    indices = np.asarray(indices).reshape(len(structure), num_neighbors + 1)
    if cutoff_radius < np.inf and np.any(distances[:, -1] < np.inf):
        warnings.warn(
            "Number of neighbors found within the cutoff_radius is equal to "
            + "num_neighbors. Increase num_neighbors to find all neighbors within "
            + "cutoff_radius.",
            stacklevel=2,
        )
    valid = distances < np.inf
    indices[valid] = wrapped_indices[indices[valid]]
    indices[~valid] = np.iinfo(np.int32).max
    # The first column is the atom itself
    max_column = np.sum(valid, axis=-1).max()
    return Neighbors(
        structure=structure,
        distances=distances[:, 1:max_column],
        indices=indices[:, 1:max_column],
    )
