from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
from ase.atoms import Atoms
from structuretoolkit.analyse import get_neighbors
from structuretoolkit.common import select_index

from lammpsparser.structure import LammpsStructure


class LammpsStructureCompatibility(LammpsStructure):
    def __init__(
        self,
        bond_dict: Optional[Dict] = None,
        units: str = "metal",
        atom_type: str = "atomic",
    ):
        super().__init__(bond_dict=bond_dict, units=units, atom_type=atom_type)
        self._molecule_ids = []

    @property
    def structure(self) -> Optional[Atoms]:
        """

        Returns:

        """
        return self._structure

    @property
    def molecule_ids(self):
        if self._molecule_ids is None or len(self._molecule_ids) == 0:
            return np.ones(len(self.structure), dtype=int)
        return self._molecule_ids

    @molecule_ids.setter
    def molecule_ids(self, val):
        self._molecule_ids = val

    @structure.setter
    def structure(self, structure):
        """

        Args:
            structure:

        Returns:

        """
        self._structure = structure
        if self._atom_type == "full":
            input_str = self.structure_full()
        elif self._atom_type == "bond":
            input_str = self.structure_bond()
        elif self._atom_type == "charge":
            input_str = self.structure_charge()
        else:  # self.atom_type == 'atomic'
            input_str = self.structure_atomic()

        self._string_input = input_str + self._get_velocities_input_string()

    def structure_bond(self):
        """

        Returns:

        """
        species_lammps_id_dict = self.get_lammps_id_dict(self.el_eam_lst)
        # analyze structure to get molecule_ids, bonds, angles etc
        coords = self.rotate_positions(self._structure)

        elements = self._structure.get_chemical_symbols()

        ## Standard atoms stuff
        atoms = "Atoms \n\n"
        # atom_style bond
        # format: atom-ID, molecule-ID, atom_type, x, y, z
        format_str = "{0:d} {1:d} {2:d} {3:f} {4:f} {5:f} "
        if len(self._structure.positions[0]) == 3:
            for id_atom, (x, y, z) in enumerate(coords):
                id_mol = self.molecule_ids[id_atom]
                atoms += (
                    format_str.format(
                        id_atom + 1,
                        id_mol,
                        species_lammps_id_dict[elements[id_atom]],
                        x,
                        y,
                        z,
                    )
                    + "\n"
                )
        elif len(self._structure.positions[0]) == 2:
            for id_atom, (x, y) in enumerate(coords):
                id_mol = self.molecule_ids[id_atom]
                atoms += (
                    format_str.format(
                        id_atom + 1,
                        id_mol,
                        species_lammps_id_dict[elements[id_atom]],
                        x,
                        y,
                        0.0,
                    )
                    + "\n"
                )
        else:
            raise ValueError("dimension 1 not yet implemented")

        ## Bond related.
        # This seems independent from the lammps atom type ids, because bonds only use atom ids
        el_list = set(self._structure.get_chemical_symbols())
        el_dict = OrderedDict()
        for object_id, el in enumerate(el_list):
            el_dict[el] = object_id

        n_s = len(el_list)
        bond_type = np.ones([n_s, n_s], dtype=int)
        count = 0
        for i in range(n_s):
            for j in range(i, n_s):
                count += 1
                bond_type[i, j] = count
                bond_type[j, i] = count

        if getattr(self.structure, "bonds", None) is None:
            if self.cutoff_radius is None:
                bonds_lst = get_bonds(structure=self.structure, max_shells=1)
            else:
                bonds_lst = get_bonds(
                    structure=self.structure, radius=self.cutoff_radius
                )
            bonds = []

            for ia, i_bonds in enumerate(bonds_lst):
                el_i = el_dict[elements[ia]]
                for el_j, b_lst in i_bonds.items():
                    b_type = bond_type[el_i][el_dict[el_j]]
                    for i_shell, ib_shell_lst in enumerate(b_lst):
                        for ib in np.unique(ib_shell_lst):
                            if ia < ib:  # avoid double counting of bonds
                                bonds.append([ia + 1, ib + 1, b_type])

            self.structure.bonds = np.array(bonds)
        bonds = self.structure.bonds

        bonds_str = "Bonds \n\n"
        for i_bond, (i_a, i_b, b_type) in enumerate(bonds):
            bonds_str += (
                "{0:d} {1:d} {2:d} {3:d}".format(i_bond + 1, b_type, i_a, i_b) + "\n"
            )

        return (
            self.lammps_header(
                structure=self.structure,
                cell_dimensions=self.simulation_cell(),
                species_lammps_id_dict=species_lammps_id_dict,
                nbonds=len(bonds),
                nbond_types=np.max(np.array(bonds)[:, 2]),
            )
            + "\n"
            + atoms
            + "\n"
            + bonds_str
            + "\n"
        )

    def structure_full(self):
        """
        Write routine to create atom structure static file for atom_type='full' that can be loaded by LAMMPS

        Returns:

        """
        species_lammps_id_dict = self.get_lammps_id_dict(self.el_eam_lst)
        coords = self.rotate_positions(self._structure)

        # extract electric charges from potential file
        if self.potential is not None:
            q_dict = {
                species_name: self.potential.get_charge(species_name)
                for species_name in set(self.structure.get_chemical_symbols())
            }
        else:
            q_dict = {
                species_name: 0.0
                for species_name in set(self.structure.get_chemical_symbols())
            }

        bonds_lst, angles_lst = [], []
        bond_type_lst, angle_type_lst = [], []
        # Using a cutoff distance to draw the bonds instead of the number of neighbors
        # Only if any bonds are defined
        if self._bond_dict is not None and len(self._bond_dict.keys()) > 0:
            cutoff_list = list()
            for val in self._bond_dict.values():
                cutoff_list.append(np.max(val["cutoff_list"]))
            max_cutoff = np.max(cutoff_list)
            # Calculate neighbors only once
            neighbors = get_neighbors(
                structure=self.structure, cutoff_radius=max_cutoff
            )

            # Draw bonds between atoms is defined in self._bond_dict
            # Go through all elements for which bonds are defined
            for element, val in self._bond_dict.items():
                el_1_list = select_index(structure=self._structure, element=element)
                if el_1_list is not None:
                    if len(el_1_list) > 0:
                        for i, v in enumerate(val["element_list"]):
                            el_2_list = select_index(
                                structure=self._structure, element=v
                            )
                            cutoff_dist = val["cutoff_list"][i]
                            for j, ind in enumerate(
                                np.array(neighbors.indices, dtype=object)[el_1_list]
                            ):
                                # Only chose those indices within the cutoff distance and which belong
                                # to the species defined in the element_list
                                # i is the index of each bond type, and j is the element index
                                id_el = el_1_list[j]
                                bool_1 = (
                                    np.array(neighbors.distances, dtype=object)[id_el]
                                    <= cutoff_dist
                                )
                                act_ind = ind[bool_1]
                                bool_2 = np.isin(act_ind, el_2_list)
                                final_ind = act_ind[bool_2]
                                # Get the bond and angle type
                                bond_type = val["bond_type_list"][i]
                                angle_type = val["angle_type_list"][i]
                                # Draw only maximum allowed bonds
                                final_ind = final_ind[: val["max_bond_list"][i]]
                                for fi in final_ind:
                                    bonds_lst.append([id_el + 1, fi + 1])
                                    bond_type_lst.append(bond_type)
                                # Draw angles if at least 2 bonds are present and if an angle type is defined for this
                                # particular set of bonds
                                if (
                                    len(final_ind) >= 2
                                    and val["angle_type_list"][i] is not None
                                ):
                                    angles_lst.append(
                                        [final_ind[0] + 1, id_el + 1, final_ind[1] + 1]
                                    )
                                    angle_type_lst.append(angle_type)

        if len(bond_type_lst) == 0:
            num_bond_types = 0
        else:
            num_bond_types = int(np.max(bond_type_lst))
        if len(angle_type_lst) == 0:
            num_angle_types = 0
        else:
            num_angle_types = int(np.max(angle_type_lst))

        atoms = "Atoms \n\n"

        # format: atom-ID, molecule-ID, atom_type, q, x, y, z
        format_str = "{0:d} {1:d} {2:d} {3:f} {4:f} {5:f} {6:f}"
        el_lst = self.structure.get_chemical_symbols()
        for id_atom, (el, coord) in enumerate(zip(el_lst, coords)):
            atoms += (
                format_str.format(
                    id_atom + 1,
                    self.molecule_ids[id_atom],
                    species_lammps_id_dict[el],
                    q_dict[el],
                    coord[0],
                    coord[1],
                    coord[2],
                )
                + "\n"
            )

        if len(bonds_lst) > 0:
            bonds_str = "Bonds \n\n"
            for i_bond, id_vec in enumerate(bonds_lst):
                bonds_str += (
                    "{0:d} {1:d} {2:d} {3:d}".format(
                        i_bond + 1, bond_type_lst[i_bond], id_vec[0], id_vec[1]
                    )
                    + "\n"
                )
        else:
            bonds_str = "\n"

        if len(angles_lst) > 0:
            angles_str = "Angles \n\n"
            for i_angle, id_vec in enumerate(angles_lst):
                angles_str += (
                    "{0:d} {1:d} {2:d} {3:d} {4:d}".format(
                        i_angle + 1,
                        angle_type_lst[i_angle],
                        id_vec[0],
                        id_vec[1],
                        id_vec[2],
                    )
                    + "\n"
                )
        else:
            angles_str = "\n"
        return (
            self.lammps_header(
                structure=self.structure,
                cell_dimensions=self.simulation_cell(),
                species_lammps_id_dict=species_lammps_id_dict,
                nbonds=len(bonds_lst),
                nangles=len(angles_lst),
                nbond_types=num_bond_types,
                nangle_types=num_angle_types,
            )
            + " \n"
            + atoms
            + "\n"
            + bonds_str
            + "\n"
            + angles_str
            + "\n"
        )


def get_bonds(
    structure: Atoms, radius=np.inf, max_shells=None, prec=0.1, num_neighbors=20
):
    """

    Args:
        structure (ase.Atoms):
        radius (float):
        max_shells:
        prec: minimum distance between any two clusters (if smaller considered to be single cluster)
        num_neighbors:

    Returns:

    """
    neighbors = get_neighbors(
        structure=structure,
        cutoff_radius=radius,
        num_neighbors=num_neighbors,
        tolerance=2,
        id_list=None,
        width_buffer=1.2,
        mode="ragged",
        norm_order=2,
    )
    return neighbors.get_bonds(radius=radius, max_shells=max_shells, prec=prec)
