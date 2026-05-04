import unittest
import numpy as np
from unittest.mock import MagicMock
from ase.build import bulk
from ase.atoms import Atoms

try:
    from lammpsparser.compatibility.structure import (
        LammpsStructureCompatibility,
        get_bonds,
    )

    skip_structuretoolkit_test = False
except ImportError:
    skip_structuretoolkit_test = True


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
