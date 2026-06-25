import unittest
from unittest.mock import patch
import numpy as np
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

    def test_molecule_ids_default(self):
        lsc = LammpsStructureCompatibility()
        lsc._structure = bulk("Al", a=4.0, cubic=True)
        np.testing.assert_array_equal(lsc.molecule_ids, np.ones(4, dtype=int))

    def test_molecule_ids_explicit(self):
        lsc = LammpsStructureCompatibility()
        lsc._structure = bulk("Al", a=4.0, cubic=True)
        lsc.molecule_ids = [1, 1, 2, 2]
        self.assertEqual(lsc.molecule_ids, [1, 1, 2, 2])


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
class TestLammpsStructureCompatibilitySetterBond(unittest.TestCase):
    def test_structure_setter_bond(self):
        lsc = LammpsStructureCompatibility(atom_type="bond")
        lsc.atom_type = "bond"
        lsc.el_eam_lst = ["Al"]
        structure = bulk("Al", a=4.05, cubic=True).repeat([2, 2, 2])
        lsc.structure = structure
        self.assertIs(lsc._structure, structure)
        self.assertIn("Bonds", lsc._string_input)

    def test_structure_setter_bond_cutoff_radius(self):
        lsc = LammpsStructureCompatibility(atom_type="bond")
        lsc.atom_type = "bond"
        lsc.el_eam_lst = ["Al"]
        lsc.cutoff_radius = 3.0
        structure = bulk("Al", a=4.05, cubic=True).repeat([2, 2, 2])
        lsc.structure = structure
        self.assertIn("Bonds", lsc._string_input)

    def test_structure_setter_bond_2d(self):
        # ASE Atoms always stores 3-column positions, so the 2D branch is
        # only reachable via a duck-typed structure (mirrors the pattern in
        # test_structure.py's two-dimensional-positions test).
        class _Dummy2D:
            def __init__(self):
                self.positions = np.zeros((2, 2))
                self.cell = np.eye(3) * 5.0
                self.bonds = np.array([[1, 2, 1]])

            def get_chemical_symbols(self):
                return ["Fe", "Fe"]

            def get_volume(self):
                return float(np.linalg.det(self.cell))

            def __len__(self):
                return 2

        lsc = LammpsStructureCompatibility(atom_type="bond")
        lsc.el_eam_lst = ["Fe"]
        lsc._structure = _Dummy2D()
        with patch.object(
            LammpsStructureCompatibility,
            "rotate_positions",
            return_value=[(0.0, 0.0), (1.0, 1.0)],
        ):
            result = lsc.structure_bond()
        self.assertIn("Bonds", result)

    def test_structure_setter_bond_1d_raises(self):
        class _Dummy1D:
            def __init__(self):
                self.positions = np.zeros((2, 1))

            def get_chemical_symbols(self):
                return ["Fe", "Fe"]

        lsc = LammpsStructureCompatibility(atom_type="bond")
        lsc.el_eam_lst = ["Fe"]
        lsc._structure = _Dummy1D()
        with patch.object(
            LammpsStructureCompatibility,
            "rotate_positions",
            return_value=[(0.0,), (1.0,)],
        ):
            with self.assertRaises(ValueError):
                lsc.structure_bond()


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestLammpsStructureCompatibilitySetterFull(unittest.TestCase):
    def test_structure_setter_full_no_potential(self):
        lsc = LammpsStructureCompatibility(atom_type="full")
        lsc.atom_type = "full"
        lsc.el_eam_lst = ["Al"]
        structure = bulk("Al", a=4.05, cubic=True)
        lsc.structure = structure
        self.assertIs(lsc._structure, structure)
        self.assertIn("Atoms", lsc._string_input)

    def test_structure_setter_full_with_potential_bonds_angles(self):
        class _DummyPotential:
            def get_charge(self, species_name):
                return 1.5

        lsc = LammpsStructureCompatibility(
            bond_dict={
                "Al": {
                    "element_list": ["Al"],
                    "cutoff_list": [3.0],
                    "max_bond_list": [2],
                    "bond_type_list": [1],
                    "angle_type_list": [2],
                }
            },
            atom_type="full",
        )
        lsc.atom_type = "full"
        lsc.el_eam_lst = ["Al"]
        lsc.potential = _DummyPotential()
        structure = bulk("Al", a=4.05, cubic=True).repeat([2, 2, 2])
        lsc.structure = structure
        self.assertIn("Bonds", lsc._string_input)
        self.assertIn("Angles", lsc._string_input)
        self.assertIn("1.500000", lsc._string_input)


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestGetBonds(unittest.TestCase):
    def test_get_bonds_default(self):
        structure = bulk("Al", a=4.05, cubic=True).repeat([2, 2, 1])
        result = get_bonds(structure=structure, max_shells=1)
        self.assertEqual(len(result), len(structure))

    def test_get_bonds_radius(self):
        structure = bulk("Al", a=4.05, cubic=True).repeat([2, 2, 1])
        result = get_bonds(structure=structure, radius=3.0)
        self.assertEqual(len(result), len(structure))


if __name__ == "__main__":
    unittest.main()
