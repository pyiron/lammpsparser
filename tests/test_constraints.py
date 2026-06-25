import unittest

from ase.build import bulk
from ase.constraints import FixAtoms, FixCom, FixedPlane

from lammpsparser.compatibility.constraints import (
    _get_fixed_atom_boolean_vector,
    set_selective_dynamics,
)


class TestConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        structure = bulk("Cu", cubic=True)
        structure.symbols[2:] = "Al"
        cls.structure = structure

    def test_selective_dynamics_mixed_calcmd_x(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [1, 0, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 6)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintxyz"], "set 0.0 0.0 0.0")
        self.assertTrue(control_dict["group constraintx"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintx"], "constraintx setforce NULL NULL 0.0"
        )
        self.assertTrue(control_dict["velocity constraintx"], "set NULL NULL 0.0")

    def test_selective_dynamics_mixed_calcmd_y(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [0, 1, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 6)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintxyz"], "set 0.0 0.0 0.0")
        self.assertTrue(control_dict["group constrainty"], "id 3 4")
        self.assertTrue(
            control_dict["fix constrainty"], "constrainty setforce NULL 0.0 NULL"
        )
        self.assertTrue(control_dict["velocity constrainty"], "set NULL 0.0 NULL")

    def test_selective_dynamics_mixed_calcmd_z(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [0, 0, 1],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 6)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintxyz"], "set 0.0 0.0 0.0")
        self.assertTrue(control_dict["group constraintz"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintz"], "constraintz setforce NULL NULL 0.0"
        )
        self.assertTrue(control_dict["velocity constraintz"], "set NULL NULL 0.0")

    def test_selective_dynamics_calcmd_xy(self):
        atoms = self.structure.copy()
        c1 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [1, 1, 0],
        )
        atoms.set_constraint([c1])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 3)
        self.assertTrue(control_dict["group constraintxy"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintxy"], "constraintxy setforce NULL 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintxy"], "set NULL 0.0 0.0")

    def test_selective_dynamics_calcmd_xz(self):
        atoms = self.structure.copy()
        c1 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [1, 0, 1],
        )
        atoms.set_constraint([c1])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 3)
        self.assertTrue(control_dict["group constraintxz"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintxz"], "constraintxz setforce NULL 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintxz"], "set NULL 0.0 0.0")

    def test_selective_dynamics_calcmd_yz(self):
        atoms = self.structure.copy()
        c1 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [0, 1, 1],
        )
        atoms.set_constraint([c1])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 3)
        self.assertTrue(control_dict["group constraintyz"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintyz"], "constraintyz setforce NULL 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintyz"], "set NULL 0.0 0.0")

    def test_selective_dynamics_mixed_x(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [1, 0, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 4)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["group constraintx"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintx"], "constraintx setforce 0.0 NULL NULL"
        )

    def test_selective_dynamics_single_fix(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        atoms.set_constraint(c1)
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 2)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )

    def test_selective_dynamics_errors(self):
        atoms = self.structure.copy()
        atoms.set_constraint(FixCom())
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)

    def test_selective_dynamics_wrong_plane(self):
        atoms = self.structure.copy()
        atoms.set_constraint(
            FixedPlane(
                [atom.index for atom in atoms if atom.symbol == "Al"],
                [2, 1, 0],
            )
        )
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)

    def test_fixed_plane_legacy_a_kwarg(self):
        # Older ASE releases named the FixedPlane constructor argument "a"
        # instead of "indices"; todict() for such an object would report
        # the constraint under the "a" key. Exercise that branch directly
        # with a duck-typed constraint, since the installed ASE version no
        # longer produces this shape.
        class _LegacyFixedPlane:
            def todict(self):
                return {
                    "name": "FixedPlane",
                    "kwargs": {"a": 2, "direction": [1, 0, 0]},
                }

        atoms = self.structure.copy()
        atoms.constraints = [_LegacyFixedPlane()]
        fixed_atom_vector = _get_fixed_atom_boolean_vector(atoms)
        self.assertTrue(fixed_atom_vector[2][0])
        self.assertFalse(fixed_atom_vector[2][1])
        self.assertFalse(fixed_atom_vector[2][2])
