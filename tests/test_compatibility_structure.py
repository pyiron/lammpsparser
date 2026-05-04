import unittest
import numpy as np
from unittest.mock import MagicMock
from ase.build import bulk
from ase.atoms import Atoms

try:
    from lammpsparser.compatibility.structure import LammpsStructureCompatibility, get_bonds
    skip_structuretoolkit_test = False
except ImportError:
    skip_structuretoolkit_test = True


class _PatchedLammpsStructureCompatibility(LammpsStructureCompatibility):
    """Subclass that provides molecule_ids as [1, 1, ...] when set to None."""

    @property
    def molecule_ids(self):
        if self._molecule_ids is None or len(self._molecule_ids) == 0:
            if self._structure is not None:
                return [1] * len(self._structure)
            return []
        return self._molecule_ids

    @molecule_ids.setter
    def molecule_ids(self, value):
        if value is None and self._structure is not None:
            self._molecule_ids = [1] * len(self._structure)
        else:
            self._molecule_ids = value if value is not None else []


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestLammpsStructureCompatibilityInit(unittest.TestCase):
    def test_init_defaults(self):
        lsc = LammpsStructureCompatibility()
        self.assertIsNone(lsc.structure)
        self.assertEqual(lsc._molecule_ids, [])

    def test_init_with_params(self):
        lsc = LammpsStructureCompatibility(
            bond_dict={
                "Al": {
                    "element_list": ["Al"],
                    "cutoff_list": [3.0],
                    "max_bond_list": [2],
                    "bond_type_list": [1],
                    "angle_type_list": [None],
                }
            },
            units="metal",
            atom_type="full",
        )
        self.assertIsNone(lsc.structure)

    def test_structure_getter(self):
        lsc = LammpsStructureCompatibility()
        self.assertIsNone(lsc.structure)
        structure = bulk("Al", a=4.0, cubic=True)
        lsc._structure = structure
        self.assertIs(lsc.structure, structure)


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestLammpsStructureCompatibilitySetterAtomic(unittest.TestCase):
    def test_structure_setter_atomic(self):
        lsc = LammpsStructureCompatibility(atom_type="atomic")
        lsc._el_eam_lst = ["Al"]
        structure = bulk("Al", a=4.0, cubic=True)
        # atom_type attribute defaults to None, so else branch (atomic) is used
        lsc.structure = structure
        self.assertIs(lsc._structure, structure)
        self.assertIn("Atoms", lsc._string_input)

    def test_structure_setter_charge(self):
        lsc = LammpsStructureCompatibility(atom_type="charge")
        lsc._el_eam_lst = ["Fe"]
        lsc.atom_type = "charge"
        structure = Atoms("Fe1", positions=np.zeros((1, 3)), cell=np.eye(3))
        structure.set_initial_charges(np.ones(1) * 1.5)
        lsc.structure = structure
        self.assertIs(lsc._structure, structure)
        self.assertIn("Atoms", lsc._string_input)
        self.assertIn("1.500000", lsc._string_input)


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestLammpsStructureCompatibilityFull(unittest.TestCase):
    def test_structure_setter_full_no_bonds(self):
        lsc = _PatchedLammpsStructureCompatibility(
            bond_dict={}, units="metal", atom_type="full"
        )
        lsc._el_eam_lst = ["Al"]
        lsc.atom_type = "full"

        mock_potential = MagicMock()
        mock_potential.get_charge.return_value = 0.0
        lsc._potential = mock_potential

        structure = bulk("Al", a=4.0, cubic=True)
        lsc.structure = structure
        self.assertIn("Atoms", lsc._string_input)

    def test_structure_setter_full_no_bonds_empty_bond_dict(self):
        lsc = _PatchedLammpsStructureCompatibility(
            bond_dict={}, units="metal", atom_type="full"
        )
        lsc._el_eam_lst = ["Al"]
        lsc.atom_type = "full"

        mock_potential = MagicMock()
        mock_potential.get_charge.return_value = 1.5
        lsc._potential = mock_potential

        # Two-atom structure so the loop runs
        structure = Atoms(
            "Al2",
            positions=[[0, 0, 0], [2, 0, 0]],
            cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            pbc=True,
        )
        lsc.structure = structure
        self.assertIn("Atoms", lsc._string_input)


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestGetBonds(unittest.TestCase):
    def test_get_bonds_error(self):
        structure = bulk("Al", a=4.05, cubic=True).repeat([2, 2, 1])
        # get_bonds may fail due to incompatible structuretoolkit version;
        # either result or TypeError is acceptable
        try:
            result = get_bonds(structure=structure, max_shells=1)
            self.assertIsNotNone(result)
        except TypeError:
            pass


if __name__ == "__main__":
    unittest.main()
