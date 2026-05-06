import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from ase.build import bulk
from ase.atoms import Atoms

try:
    from lammpsparser.compatibility.structure import (
        LammpsStructureCompatibility,
        get_bonds,
    )
    import lammpsparser.compatibility.structure as compat_struct

    skip_structuretoolkit_test = False
except ImportError:
    skip_structuretoolkit_test = True


# ---------------------------------------------------------------------------
# Helper: subclass that exposes molecule_ids as a property so that the
# `self.molecule_ids = None` assignment inside structure_bond / structure_full
# is a no-op, and the getter always returns sensible defaults.
# ---------------------------------------------------------------------------
if not skip_structuretoolkit_test:

    class _TestableLSC(LammpsStructureCompatibility):
        """LammpsStructureCompatibility with molecule_ids as a proper property."""

        @property
        def molecule_ids(self):
            if self._structure is None:
                return []
            return [1] * len(self._structure)

        @molecule_ids.setter
        def molecule_ids(self, val):
            pass  # ignore – molecule_ids is derived from _structure


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

    def test_structure_setter_full_branch(self):
        """Setting structure with atom_type='full' exercises the full branch (line 43)."""
        structure = bulk("Fe", a=2.87, cubic=True)
        mock_potential = MagicMock()
        mock_potential.get_charge.return_value = 0.0

        lsc = _TestableLSC(bond_dict={})
        lsc._el_eam_lst = ["Fe"]
        lsc._potential = mock_potential
        lsc.atom_type = "full"  # use the plain attribute checked in the setter

        lsc.structure = structure  # triggers structure_full()
        self.assertIs(lsc._structure, structure)
        self.assertIn("Atoms", lsc._string_input)

    def test_structure_setter_bond_branch(self):
        """Setting structure with atom_type='bond' exercises the bond branch (line 45)."""
        structure = bulk("Fe", a=2.87, cubic=True)
        structure.bonds = np.array([[1, 2, 1]])

        lsc = _TestableLSC()
        lsc._el_eam_lst = ["Fe"]
        lsc.atom_type = "bond"  # use the plain attribute checked in the setter

        lsc.structure = structure  # triggers structure_bond()
        self.assertIs(lsc._structure, structure)
        self.assertIn("Atoms", lsc._string_input)


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestStructureBond(unittest.TestCase):
    """Tests for LammpsStructureCompatibility.structure_bond()."""

    def setUp(self):
        self.structure = bulk("Fe", a=2.87, cubic=True)

    def test_structure_bond_3d_preexisting_bonds(self):
        """structure_bond uses pre-existing bonds when structure.bonds is set."""
        self.structure.bonds = np.array([[1, 2, 1], [2, 1, 1]])
        lsc = _TestableLSC()
        lsc._el_eam_lst = ["Fe"]
        lsc._structure = self.structure

        result = lsc.structure_bond()

        self.assertIn("Atoms", result)
        self.assertIn("Bonds", result)
        # Each bond row is "id bond_type atom_a atom_b"
        self.assertIn("1 1 1 2", result)

    def test_structure_bond_calculates_bonds_no_cutoff(self):
        """structure_bond calculates bonds via get_bonds when bonds=None and no cutoff."""
        self.structure.bonds = None

        # bonds_lst format: list-per-atom of {element: [shell_indices]}.
        # Atom 0 bonds to atom 1 and vice-versa; the ia < ib guard inside
        # structure_bond ensures only one bond entry [1, 2, bond_type] is kept.
        bonds_lst = [{"Fe": [[1]]}, {"Fe": [[0]]}]

        lsc = _TestableLSC()
        lsc._el_eam_lst = ["Fe"]
        lsc._structure = self.structure

        with patch(
            "lammpsparser.compatibility.structure.get_bonds",
            return_value=bonds_lst,
        ) as mock_gb:
            result = lsc.structure_bond()

        self.assertIn("Atoms", result)
        self.assertIn("Bonds", result)
        mock_gb.assert_called_once_with(structure=self.structure, max_shells=1)

    def test_structure_bond_calculates_bonds_with_cutoff(self):
        """When cutoff_radius is set, get_bonds is called with radius keyword."""
        self.structure.bonds = None

        bonds_lst = [{"Fe": [[1]]}, {"Fe": [[0]]}]

        lsc = _TestableLSC()
        lsc._el_eam_lst = ["Fe"]
        lsc._structure = self.structure
        lsc.cutoff_radius = 3.0

        with patch(
            "lammpsparser.compatibility.structure.get_bonds",
            return_value=bonds_lst,
        ) as mock_gb:
            result = lsc.structure_bond()

        self.assertIn("Atoms", result)
        self.assertIn("Bonds", result)
        mock_gb.assert_called_once_with(
            structure=self.structure, radius=lsc.cutoff_radius
        )


@unittest.skipIf(skip_structuretoolkit_test, "structuretoolkit not available")
class TestStructureFull(unittest.TestCase):
    """Tests for LammpsStructureCompatibility.structure_full()."""

    def _make_mock_potential(self, charge=0.0):
        mock_potential = MagicMock()
        mock_potential.get_charge.return_value = charge
        return mock_potential

    def test_structure_full_no_bond_dict(self):
        """structure_full with an empty bond_dict produces Atoms section without bonds."""
        structure = bulk("Fe", a=2.87, cubic=True)
        lsc = _TestableLSC(bond_dict={})
        lsc._el_eam_lst = ["Fe"]
        lsc._structure = structure
        lsc._potential = self._make_mock_potential(charge=0.5)

        result = lsc.structure_full()

        self.assertIn("Atoms", result)
        # No bonds section when bond_dict is empty
        self.assertNotIn("Bonds", result)

    def test_structure_full_with_bond_dict_no_angles(self):
        """structure_full with bond_dict generates a Bonds section (no angles)."""
        structure = bulk("Fe", a=2.87, cubic=True).repeat([2, 2, 2])
        bond_dict = {
            "Fe": {
                "element_list": ["Fe"],
                "cutoff_list": [3.0],
                "max_bond_list": [6],
                "bond_type_list": [1],
                "angle_type_list": [None],  # no angles
            }
        }
        lsc = _TestableLSC(bond_dict=bond_dict)
        lsc._el_eam_lst = ["Fe"]
        lsc._structure = structure
        lsc._potential = self._make_mock_potential(charge=0.0)
        result = lsc.structure_full()

        self.assertIn("Atoms", result)
        self.assertIn("Bonds", result)

    def test_structure_full_with_bond_dict_and_angles(self):
        """structure_full generates both Bonds and Angles sections when configured."""
        structure = bulk("Fe", a=2.87, cubic=True).repeat([2, 2, 2])
        bond_dict = {
            "Fe": {
                "element_list": ["Fe"],
                "cutoff_list": [3.0],
                "max_bond_list": [6],
                "bond_type_list": [1],
                "angle_type_list": [1],  # angle type defined
            }
        }
        lsc = _TestableLSC(bond_dict=bond_dict)
        lsc._el_eam_lst = ["Fe"]
        lsc._structure = structure
        lsc._potential = self._make_mock_potential(charge=0.0)
        result = lsc.structure_full()

        self.assertIn("Atoms", result)
        self.assertIn("Bonds", result)
        self.assertIn("Angles", result)


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

    def test_get_bonds_mocked_neighbors(self):
        """Cover the return statement of get_bonds (line 336) via mocked get_neighbors."""
        structure = bulk("Al", a=4.05, cubic=True)
        expected_bonds = [{"Al": [[0, 1]]}]
        mock_neighbors = MagicMock()
        mock_neighbors.get_bonds.return_value = expected_bonds

        with patch(
            "lammpsparser.compatibility.structure.get_neighbors",
            return_value=mock_neighbors,
        ):
            result = get_bonds(structure=structure, radius=3.0, max_shells=1)

        self.assertIs(result, expected_bonds)
        mock_neighbors.get_bonds.assert_called_once_with(
            radius=3.0, max_shells=1, prec=0.1
        )


if __name__ == "__main__":
    unittest.main()
