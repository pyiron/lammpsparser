# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import annotations

import decimal as dec
import posixpath
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_masses, atomic_numbers

from lammpsparser.units import UnitConverter

try:
    from ase.calculators.lammps import Prism as PrismBase
except ImportError:
    try:
        from ase.calculators.lammpsrun import Prism as PrismBase
    except ImportError:
        from ase.calculators.lammpsrun import (
            prism as PrismBase,  # type: ignore[attr-defined,no-redef]
        )

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Yury Lysogorskiy, Jan Janssen, Markus Tautschnig"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class UnfoldingPrism(PrismBase):
    """
    Create a lammps-style triclinic prism object from a cell

    The main purpose of the prism-object is to create suitable
    string representations of prism limits and atom positions
    within the prism.
    When creating the object, the digits parameter (default set to 10)
    specify the precision to use.
    lammps is picky about stuff being within semi-open intervals,
    e.g. for atom positions (when using create_atom in the in-file),
    x must be within [xlo, xhi).

    Args:
        cell:
        pbc:
        digits:
    """

    def __init__(
        self,
        cell: np.ndarray,
        pbc: Union[bool, tuple[bool, bool, bool]] = (True, True, True),
        digits: int = 10,
    ):
        # Temporary fix. Since the arguments for the constructor have changed, try to see if it is compatible with
        # the latest ase. If not, revert to the old __init__ parameters.
        if isinstance(pbc, bool):
            pbc = (pbc, pbc, pbc)
        try:
            super(UnfoldingPrism, self).__init__(
                cell, pbc=np.array(pbc), tolerance=float("1e-{}".format(digits))
            )
        except TypeError:
            super(UnfoldingPrism, self).__init__(cell, pbc=np.array(pbc), digits=digits)  # type: ignore[call-arg]
        a, b, c = cell
        an, bn, cn = [np.linalg.norm(v) for v in cell]

        alpha = np.arccos(np.dot(b, c) / (bn * cn))
        beta = np.arccos(np.dot(a, c) / (an * cn))
        gamma = np.arccos(np.dot(a, b) / (an * bn))

        xhi = an
        xyp = np.cos(gamma) * bn
        yhi = np.sin(gamma) * bn
        xzp = np.cos(beta) * cn
        yzp = (bn * cn * np.cos(alpha) - xyp * xzp) / yhi
        zhi = np.sqrt(cn**2 - xzp**2 - yzp**2)

        # Set precision
        self.car_prec = dec.Decimal("10.0") ** int(
            np.floor(np.log10(max((xhi, yhi, zhi)))) - digits
        )
        self.dir_prec = dec.Decimal("10.0") ** (-digits)
        self.acc = float(self.car_prec)
        self.eps = np.finfo(xhi).eps

        # For rotating positions from ase to lammps
        apre = np.array(((xhi, 0, 0), (xyp, yhi, 0), (xzp, yzp, zhi)))
        # np.linalg.inv(cell) ?= np.array([np.cross(b, c), np.cross(c, a), np.cross(a, b)]).T / np.linalg.det(cell)
        self.R = np.dot(np.linalg.inv(cell), apre)

        def fold(
            vec: np.ndarray, pvec: np.ndarray, i: int
        ) -> Tuple[List[float], float]:
            p = pvec[i]
            x = vec[i] + 0.5 * p
            n = (np.mod(x, p) - x) / p
            return [float(self.f2qdec(vec_a)) for vec_a in (vec + n * pvec)], n

        apre[1, :], n1 = fold(apre[1, :], apre[0, :], 0)
        if np.abs(apre[1, 0] / apre[0, 0]) > 0.5:
            apre[1, 0] -= np.sign(n1) * apre[0, 0]
            n1 -= np.sign(n1)

        apre[2, :], n2 = fold(apre[2, :], apre[1, :], 1)
        if np.abs(apre[2, 1] / apre[1, 1]) > 0.5:
            apre[2, 1] -= np.sign(n2) * apre[1, 1]
            n2 -= np.sign(n2)

        apre[2, :], n3 = fold(apre[2, :], apre[0, :], 0)
        if np.abs(apre[2, 0] / apre[0, 0]) > 0.5:
            apre[2, 0] -= np.sign(n3) * apre[0, 0]
            n3 -= np.sign(n3)
        self.ns = [n1, n2, n3]

        d_a = apre[0, 0] / 2 - apre[1, 0]
        if np.abs(d_a) < self.acc:
            if d_a < 0:
                print("debug: apply shift")
                apre[1, 0] += 2 * d_a
                apre[2, 0] += 2 * d_a

        self.A = apre

        if self.is_skewed() and (not (pbc[0] and pbc[1] and pbc[2])):
            warnings.warn(
                "Skewed lammps cells should have PBC == True in all directions!"
            )

    def unfold_cell(self, cell: np.ndarray) -> np.ndarray:
        """
        Unfold LAMMPS cell to original

        Let C be the pyiron_atomistics cell and A be the Lammps cell, then define (in init) the rotation matrix between them as
            R := C^inv.A
        And recall that rotation matrices have the property
            R^T == R^inv
        Then left multiply the definition of R by C, and right multiply by R.T to get
            C.R.R^T = C.C^inv.A.R^T
        Then
            C = A.R^T

        After that, account for the folding process.

        Args:
            cell: LAMMPS cell,

        Returns:
            unfolded cell
        """
        # Rotation
        ucell = np.dot(cell, self.R.T)
        # Folding
        a = ucell[0]
        bp = ucell[1]
        cpp = ucell[2]
        n1, n2, n3 = self.ns
        b = bp - n1 * a
        c = cpp - n2 * bp - n3 * a
        return np.array([a, b, c])

    def pos_to_lammps(self, position: np.ndarray) -> Tuple[float, float, float]:
        """
        Rotate an ase-cell position to the lammps cell orientation

        Args:
            position:

        Returns:
            tuple of float.
        """
        return tuple([x for x in np.dot(position, self.R)])

    def f2qdec(self, f: float) -> dec.Decimal:
        """Round ``f`` down to the Cartesian precision of the prism."""
        return dec.Decimal(str(f)).quantize(self.car_prec, dec.ROUND_DOWN)

    def f2s(self, f: float) -> str:
        """Convert ``f`` to a string rounded to the Cartesian precision of the prism."""
        return str(dec.Decimal(str(f)).quantize(self.car_prec, dec.ROUND_HALF_EVEN))

    def get_lammps_prism_str(self) -> Tuple[str, ...]:
        """Return a tuple of strings"""
        p = self.get_lammps_prism()
        return tuple([self.f2s(x) for x in p])


class LammpsStructure:
    """
    Generate LAMMPS data file content from an ASE :class:`~ase.atoms.Atoms` structure.

    After construction, set :attr:`el_eam_lst` to the ordered list of element
    symbols used by the interatomic potential and assign an
    :class:`~ase.atoms.Atoms` object to :attr:`structure`.  The assignment
    triggers the serialisation of the structure into LAMMPS data format.  Call
    :meth:`write_file` to persist the result.

    Supported ``atom_style`` modes are ``"atomic"`` (default) and ``"charge"``.

    Args:
        bond_dict (dict, optional): Bond topology dictionary (used by derived
            classes for ``"bond"`` and ``"full"`` atom styles).
        units (str): LAMMPS unit system (default: ``"metal"``).
        atom_type (str): LAMMPS ``atom_style`` (default: ``"atomic"``).
    """

    def __init__(
        self,
        bond_dict: Optional[Dict] = None,
        units: str = "metal",
        atom_type: str = "atomic",
    ):
        self._string_input: str = ""
        self._structure: Optional[Atoms] = None
        self._potential: Optional[Any] = None
        self._el_eam_lst: List[str] = []
        self.atom_type: Optional[str] = None
        self.cutoff_radius: Optional[float] = None
        self.digits: int = 10
        self._bond_dict: Optional[Dict] = bond_dict
        self._force_skewed: bool = False
        self._units: str = units
        self._atom_type: str = atom_type

    @property
    def potential(self) -> Any:
        return self._potential

    @potential.setter
    def potential(self, val: Any):
        self._potential = val

    @property
    def structure(self) -> Optional[Atoms]:
        """The ASE :class:`~ase.atoms.Atoms` object associated with this instance."""
        return self._structure

    @structure.setter
    def structure(self, structure: Atoms):
        """
        Set the atomic structure and serialise it to the internal LAMMPS data string.

        The setter immediately builds the complete LAMMPS data file content
        (header, cell, masses, atom positions, and optionally velocities) and
        stores it in ``_string_input``.

        Args:
            structure (ase.atoms.Atoms): The structure to serialise.
        """
        self._structure = structure
        if self._atom_type == "charge":
            input_str = self.structure_charge()
        else:  # self.atom_type == 'atomic'
            input_str = self.structure_atomic()
        self._string_input = input_str + self._get_velocities_input_string()

    def _get_velocities_input_string(self) -> str:
        input_str = ""
        if self._structure is not None and not np.all(
            np.isclose(
                self._structure.get_velocities(),
                np.array([[0.0, 0.0, 0.0]] * len(self._structure)),
            )
        ):
            uc = UnitConverter(self._units)
            self._structure.set_velocities(
                self._structure.get_velocities() * uc.pyiron_to_lammps("velocity")
            )
            vels = self.rotate_velocities(self._structure)
            input_str += "Velocities\n\n"
            if len(self._structure.positions[0]) == 3:
                format_str = "{0:d} {1:f} {2:f} {3:f}\n"
                for id_atom, (x, y, z) in enumerate(vels, start=1):
                    input_str += format_str.format(id_atom, x, y, z)
            elif len(self._structure.positions[0]) == 2:
                format_str = "{0:d} {1:f} {2:f}\n"
                for id_atom, velocity in enumerate(vels, start=1):
                    input_str += format_str.format(id_atom, velocity[0], velocity[1])
        return input_str

    @property
    def el_eam_lst(self) -> List[str]:
        """Ordered list of element symbols as defined in the LAMMPS potential file."""
        return self._el_eam_lst

    @el_eam_lst.setter
    def el_eam_lst(self, el_eam_lst: List[str]):
        """
        Set the ordered element list used to assign LAMMPS integer type IDs.

        Args:
            el_eam_lst (list[str]): Chemical symbols in the same order as they
                appear in the ``pair_coeff`` or ``pair_style`` commands of the
                LAMMPS input script.
        """
        self._el_eam_lst = el_eam_lst

    @staticmethod
    def get_lammps_id_dict(el_eam_lst: List[str]) -> Dict[str, int]:
        """
        Build a mapping from element symbol to LAMMPS integer type ID (1-based).

        The order of ``el_eam_lst`` determines the mapping, which must match
        the order of species in the LAMMPS potential definition.

        Args:
            el_eam_lst (list[str]): Ordered list of element symbols.

        Returns:
            dict: Mapping ``{symbol: lammps_type_id}`` with IDs starting at 1.

        Raises:
            ValueError: If ``el_eam_lst`` is empty.
        """
        if len(el_eam_lst) == 0:
            raise ValueError("el_eam_list is empty. Can not determine order of species")
        return {el: idx + 1 for idx, el in enumerate(el_eam_lst)}

    @staticmethod
    def lammps_header(
        structure: Atoms,
        cell_dimensions: str,
        species_lammps_id_dict: Dict[str, int],
        nbonds: Optional[int] = None,
        nangles: Optional[int] = None,
        nbond_types: Optional[int] = None,
        nangle_types: Optional[int] = None,
    ) -> str:
        """
        Generate the header section of a LAMMPS data file.

        Produces the counts block (number of atoms, atom types, bonds, angles)
        followed by the cell dimensions and the ``Masses`` section.

        Args:
            structure (ase.atoms.Atoms): Structure whose atom count and species
                masses are written.
            cell_dimensions (str): Pre-formatted string with ``xlo``/``xhi``/…
                lines as returned by :meth:`simulation_cell`.
            species_lammps_id_dict (dict): Mapping ``{symbol: lammps_type_id}``
                as returned by :meth:`get_lammps_id_dict`.
            nbonds (int, optional): Total number of bonds (omitted when ``None``).
            nangles (int, optional): Total number of angles (omitted when ``None``).
            nbond_types (int, optional): Number of distinct bond types.
            nangle_types (int, optional): Number of distinct angle types.

        Returns:
            str: Header string ready to be prepended to the data file.
        """
        atomtypes = (
            "Start File for LAMMPS \n"
            + "{0:d} atoms".format(len(structure))
            + " \n"
            + "{0} atom types".format(len(species_lammps_id_dict.keys()))
            + " \n"
        )  # '{0} atom types'.format(structure.get_number_of_species()) + ' \n'
        if nbonds is not None:
            atomtypes += "{0:d} bonds\n".format(nbonds)
        if nangles is not None:
            atomtypes += "{0:d} angles\n".format(nangles)
        if nbond_types is not None:
            atomtypes += "{0:d} bond types\n".format(nbond_types)
        if nangle_types is not None:
            atomtypes += "{0:d} angle types\n".format(nangle_types)

        masses = "Masses\n\n"
        for el, idx in species_lammps_id_dict.items():
            mass = atomic_masses[atomic_numbers[el]]
            masses += "{0:3d} {1:f}  # ({2}) \n".format(idx, mass, el)

        return atomtypes + "\n" + cell_dimensions + "\n" + masses + "\n"

    def simulation_cell(self) -> str:
        """
        Format the simulation cell dimensions as a LAMMPS data file box block.

        Builds an :class:`UnfoldingPrism` from the current structure's cell and
        writes the ``xlo xhi``, ``ylo yhi``, ``zlo zhi`` lines (and ``xy xz yz``
        for skewed cells) in the high-precision format required by LAMMPS.

        Returns:
            str: Multi-line string with the box bounds block.

        Raises:
            ValueError: If no structure has been assigned.
        """
        if self._structure is None:
            raise ValueError("Structure not set")
        self.prism = UnfoldingPrism(self._structure.cell, digits=15)
        xhi, yhi, zhi, xy, xz, yz = self.prism.get_lammps_prism_str()
        # Please, be carefull and not round xhi, yhi,..., otherwise you will get too skew cell from LAMMPS.
        # These values are already checked in UnfoldingPrism to fullfill LAMMPS skewness criteria
        simulation_cell = (
            "0. {} xlo xhi\n".format(xhi)
            + "0. {} ylo yhi\n".format(yhi)
            + "0. {} zlo zhi\n".format(zhi)
        )

        if is_skewed(self._structure) or self._force_skewed:
            simulation_cell += "{0} {1} {2} xy xz yz\n".format(xy, xz, yz)

        return simulation_cell

    def structure_atomic(self) -> str:
        """
        Serialise the structure for LAMMPS ``atom_style atomic``.

        Writes the header followed by an ``Atoms`` block where each line has
        the format ``atom-ID atom-type x y z``.  Positions are rotated to the
        LAMMPS upper-triangular cell frame.

        Returns:
            str: Complete LAMMPS data file content (header + atoms).

        Raises:
            ValueError: If no structure or element list has been set.
        """
        if self._structure is None:
            raise ValueError("Structure not set")
        species_lammps_id_dict = self.get_lammps_id_dict(self.el_eam_lst)
        atoms = "Atoms\n\n"
        coords = self.rotate_positions(self._structure)

        el_lst = self._structure.get_chemical_symbols()
        for id_atom, (el, coord) in enumerate(zip(el_lst, coords)):
            dim = len(self._structure.positions[0])
            c = np.zeros(3)
            c[:dim] = coord
            atoms += (
                "{0:d} {1:d} {2:.15f} {3:.15f} {4:.15f}".format(
                    id_atom + 1, species_lammps_id_dict[el], c[0], c[1], c[2]
                )
                + "\n"
            )
        return (
            self.lammps_header(
                structure=self._structure,
                cell_dimensions=self.simulation_cell(),
                species_lammps_id_dict=species_lammps_id_dict,
            )
            + atoms
            + "\n"
        )

    def structure_charge(self):
        """
        Serialise the structure for LAMMPS ``atom_style charge``.

        Writes the header followed by an ``Atoms`` block where each line has
        the format ``atom-ID atom-type charge x y z``.  The per-atom charges
        are read from :meth:`ase.atoms.Atoms.get_initial_charges`.  By
        convention, LAMMPS atom type numbers are assigned alphabetically for
        each chemical species.  Positions are rotated to the LAMMPS
        upper-triangular cell frame.

        Returns:
            str: Complete LAMMPS data file content (header + atoms).
        """
        species_lammps_id_dict = self.get_lammps_id_dict(self.el_eam_lst)
        atoms = "Atoms\n\n"
        coords = self.rotate_positions(self._structure)
        el_charge_lst = self._structure.get_initial_charges()
        el_lst = self._structure.get_chemical_symbols()
        for id_atom, (el, coord) in enumerate(zip(el_lst, coords)):
            dim = len(self._structure.positions[0])
            c = np.zeros(3)
            c[:dim] = coord
            atoms += (
                "{0:d} {1:d} {2:f} {3:.15f} {4:.15f} {5:.15f}".format(
                    id_atom + 1,
                    species_lammps_id_dict[el],
                    el_charge_lst[id_atom],
                    c[0],
                    c[1],
                    c[2],
                )
                + "\n"
            )
        return (
            self.lammps_header(
                structure=self.structure,
                cell_dimensions=self.simulation_cell(),
                species_lammps_id_dict=species_lammps_id_dict,
            )
            + atoms
            + "\n"
        )

    def rotate_positions(self, structure: Atoms) -> List[Tuple[float, float, float]]:
        """
        Rotate all atomic positions in given structure according to new Prism cell

        Args:
            structure: Atoms-like object. Should has .positions attribute

        Returns:
            (list): List of rotated coordinates
        """
        if self._structure is None:
            raise ValueError("Structure not set")
        prism = UnfoldingPrism(self._structure.cell)
        coords = [prism.pos_to_lammps(position) for position in structure.positions]
        return coords

    def rotate_velocities(self, structure: Atoms) -> List[Tuple[float, float, float]]:
        """
        Rotate all atomic velocities in given structure according to new Prism cell

        Args:
            structure: Atoms-like object. Should have .velocities attribute.

        Returns:
            (list): List of rotated velocities
        """
        if self._structure is None:
            raise ValueError("Structure not set")
        prism = UnfoldingPrism(self._structure.cell)
        vels = [prism.pos_to_lammps(vel) for vel in structure.get_velocities()]
        return vels

    def write_file(self, file_name: str, cwd: Optional[str] = None):
        """
        Write GenericParameters to input file

        Args:
            file_name (str): name of the file, either absolute (then cwd must be None) or relative
            cwd (str): path name (default: None)
        """
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)

        with open(file_name, "w") as f:
            for line in self._string_input:
                f.write(line)


def is_skewed(structure: Atoms, tolerance: float = 1.0e-8) -> bool:
    """
    Check whether the simulation box is skewed/sheared. The algorithm compares the box volume
    and the product of the box length in each direction. If these numbers do not match, the box
    is considered to be skewed and the function returns True

    Args:
        tolerance (float): Relative tolerance above which the structure is considered as skewed

    Returns:
        (bool): Whether the box is skewed or not.
    """
    volume = structure.get_volume()
    prod = np.linalg.norm(structure.cell, axis=-1).prod()
    if volume > 0:
        if abs(volume - prod) / volume < tolerance:
            return False
    return True


def write_lammps_datafile(
    structure: Atoms,
    potential_elements: Union[np.ndarray, list[str]],
    bond_dict: Optional[Dict] = None,
    units: str = "metal",
    file_name: str = "lammps.data",
    working_directory: Optional[str] = None,
    atom_type: str = "atomic",
) -> None:
    """
    Write an ASE structure to a LAMMPS data file.

    Convenience wrapper around :class:`LammpsStructure` that handles the full
    workflow: build the data-file string, optionally combine with a
    ``working_directory``, and write to disk.  This is the file loaded by
    LAMMPS via ``read_data`` in the input script.

    Args:
        structure (ase.atoms.Atoms): Atomic structure to write.
        potential_elements (numpy.ndarray or list[str]): Ordered list of
            element symbols matching the potential definition.  Determines the
            LAMMPS integer type IDs.
        bond_dict (dict, optional): Bond topology used for ``"bond"`` and
            ``"full"`` atom styles (pass ``None`` for ``"atomic"`` or
            ``"charge"``).
        units (str): LAMMPS unit system (default: ``"metal"``).
        file_name (str): Output filename (default: ``"lammps.data"``).
        working_directory (str, optional): If given, the file is written to
            ``<working_directory>/<file_name>``.
        atom_type (str): LAMMPS ``atom_style`` keyword; one of ``"atomic"``
            (default), ``"charge"``, ``"bond"``, or ``"full"``.
    """
    lammps_str = LammpsStructure(bond_dict=bond_dict, units=units, atom_type=atom_type)
    lammps_str.el_eam_lst = cast(List[str], list(potential_elements))
    lammps_str.structure = structure
    lammps_str.write_file(file_name=file_name, cwd=working_directory)


def structure_to_lammps(structure: Atoms) -> Atoms:
    """
    Converts a structure to the Lammps coordinate frame

    Args:
        structure (pyiron.atomistics.structure.atoms.Atoms): Structure to convert.

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: Structure with the LAMMPS coordinate frame.
    """
    prism = UnfoldingPrism(structure.cell)
    lammps_structure = structure.copy()
    lammps_structure.set_cell(prism.A)
    lammps_structure.positions = np.matmul(structure.positions, prism.R)
    if not np.all(
        np.isclose(
            structure.get_velocities(), np.array([[0.0, 0.0, 0.0]] * len(structure))
        )
    ):
        lammps_structure.set_velocities(np.matmul(structure.get_velocities(), prism.R))
    return lammps_structure
