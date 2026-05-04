import unittest
import warnings
import numpy as np
from ase.build import bulk

from lammpsparser.compatibility.calculate import (
    calc_md,
    calc_minimize,
    _set_initial_velocity,
    _pressure_to_lammps,
    _get_rotation_matrix,
    _modify_structure_to_allow_requested_deformation,
    _is_isotropic_hydrostatic,
)


class TestCalcMd(unittest.TestCase):
    def test_delta_temp_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_md(temperature=500.0, delta_temp=0.1)
        self.assertTrue(any("delta_temp" in str(warning.message) for warning in w))
        self.assertIsInstance(result, list)

    def test_delta_press_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_md(temperature=500.0, pressure=0.0, delta_press=1.0)
        self.assertTrue(any("delta_press" in str(warning.message) for warning in w))
        self.assertIsInstance(result, list)

    def test_too_many_temperatures_raises(self):
        with self.assertRaises(ValueError):
            calc_md(temperature=[100.0, 200.0, 300.0])

    def test_npt_temperature_zero_raises(self):
        with self.assertRaises(ValueError):
            calc_md(temperature=0.0, pressure=0.0)

    def test_npt_list_pressure(self):
        result = calc_md(
            temperature=500.0,
            pressure=[0.0, 0.0, 0.0, None, None, None],
        )
        self.assertIsInstance(result, list)
        self.assertTrue(any("npt" in line for line in result))

    def test_npt_list_pressure_with_shear(self):
        result = calc_md(
            temperature=500.0,
            pressure=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        self.assertIsInstance(result, list)

    def test_nvt_temperature_zero_raises(self):
        with self.assertRaises(ValueError):
            calc_md(temperature=0.0)

    def test_tloop(self):
        result = calc_md(temperature=500.0, tloop=5)
        self.assertTrue(any("tloop 5" in line for line in result))

    def test_nve_with_langevin_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_md(langevin=True)
        self.assertTrue(
            any("Langevin" in str(warning.message) for warning in w)
        )
        self.assertIsInstance(result, list)

    def test_npt_langevin(self):
        result = calc_md(
            temperature=500.0,
            pressure=0.0,
            langevin=True,
        )
        self.assertIsInstance(result, list)
        self.assertTrue(any("nph" in line for line in result))
        self.assertTrue(any("langevin" in line for line in result))

    def test_nvt_langevin(self):
        result = calc_md(
            temperature=500.0,
            langevin=True,
        )
        self.assertIsInstance(result, list)
        self.assertTrue(any("langevin" in line for line in result))

    def test_invalid_units(self):
        with self.assertRaises(NotImplementedError):
            calc_md(temperature=500.0, units="invalid_units")

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            calc_md(temperature=500.0, seed=-1)

    def test_two_temperatures(self):
        result = calc_md(temperature=[300.0, 600.0])
        self.assertIsInstance(result, list)


class TestCalcMinimize(unittest.TestCase):
    def test_minimize_no_structure_with_pressure_raises(self):
        with self.assertRaises(ValueError):
            calc_minimize(structure=None, pressure=0.0)

    def test_minimize_with_structure_pressure(self):
        structure = bulk("Al", a=4.0, cubic=True)
        result, _ = calc_minimize(structure=structure, pressure=0.0)
        self.assertIsInstance(result, list)

    def test_minimize_with_list_pressure(self):
        structure = bulk("Al", a=4.0, cubic=True)
        result, _ = calc_minimize(
            structure=structure, pressure=[0.0, 0.0, 0.0, None, None, None]
        )
        self.assertIsInstance(result, list)


class TestSetInitialVelocity(unittest.TestCase):
    def test_basic(self):
        result = _set_initial_velocity(temperature=300.0)
        self.assertIn("velocity all create", result)
        self.assertNotIn("dist gaussian", result)
        self.assertNotIn("sum yes", result)
        self.assertNotIn("mom no", result)
        self.assertNotIn("rot no", result)

    def test_gaussian(self):
        result = _set_initial_velocity(temperature=300.0, gaussian=True)
        self.assertIn("dist gaussian", result)

    def test_append_value(self):
        result = _set_initial_velocity(temperature=300.0, append_value=True)
        self.assertIn("sum yes", result)

    def test_no_lin_momentum(self):
        result = _set_initial_velocity(temperature=300.0, zero_lin_momentum=False)
        self.assertIn("mom no", result)

    def test_no_rot_momentum(self):
        result = _set_initial_velocity(temperature=300.0, zero_rot_momentum=False)
        self.assertIn("rot no", result)

    def test_all_options(self):
        result = _set_initial_velocity(
            temperature=300.0,
            append_value=True,
            zero_lin_momentum=False,
            zero_rot_momentum=False,
        )
        self.assertIn("sum yes", result)
        self.assertIn("mom no", result)
        self.assertIn("rot no", result)


class TestPressureToLammps(unittest.TestCase):
    def test_scalar(self):
        result = _pressure_to_lammps(pressure=1.0, rotation_matrix=None)
        self.assertIsInstance(result, float)

    def test_too_long_raises(self):
        with self.assertRaises(ValueError):
            _pressure_to_lammps(
                pressure=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                rotation_matrix=None,
            )

    def test_all_none_raises(self):
        with self.assertRaises(ValueError):
            _pressure_to_lammps(
                pressure=[None, None],
                rotation_matrix=None,
            )

    def test_ortho_rotation_no_rotation_applied(self):
        result = _pressure_to_lammps(
            pressure=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            rotation_matrix=np.eye(3),
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)

    def test_non_ortho_rotation_with_none_raises(self):
        rotation_matrix = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            _pressure_to_lammps(
                pressure=[1.0, 2.0, 3.0, None, None, None],
                rotation_matrix=rotation_matrix,
            )

    def test_non_ortho_rotation_applied(self):
        rotation_matrix = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = _pressure_to_lammps(
            pressure=[1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            rotation_matrix=rotation_matrix,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)

    def test_list_with_none(self):
        result = _pressure_to_lammps(
            pressure=[1.0, 1.0, 1.0, None, None, None],
            rotation_matrix=None,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)


class TestIsIsotropicHydrostatic(unittest.TestCase):
    def test_isotropic_shear_none(self):
        self.assertTrue(
            _is_isotropic_hydrostatic([1.0, 1.0, 1.0, None, None, None])
        )

    def test_isotropic_shear_zero(self):
        self.assertTrue(
            _is_isotropic_hydrostatic([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        )

    def test_non_isotropic(self):
        self.assertFalse(
            _is_isotropic_hydrostatic([1.0, 2.0, 3.0, None, None, None])
        )

    def test_non_isotropic_shear(self):
        self.assertFalse(
            _is_isotropic_hydrostatic([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        )


class TestGetRotationMatrix(unittest.TestCase):
    def test_no_structure_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rotation_matrix, structure = _get_rotation_matrix(
                structure=None, pressure=[1.0, 1.0, 1.0]
            )
        self.assertIsNone(rotation_matrix)
        self.assertIsNone(structure)
        self.assertTrue(
            any("No structure set" in str(warning.message) for warning in w)
        )

    def test_with_structure(self):
        structure = bulk("Al", a=4.0, cubic=True)
        rotation_matrix, _ = _get_rotation_matrix(
            structure=structure, pressure=0.0
        )
        self.assertIsNotNone(rotation_matrix)


class TestModifyStructure(unittest.TestCase):
    def test_scalar_pressure_unchanged(self):
        structure = bulk("Al", a=4.0, cubic=True)
        result = _modify_structure_to_allow_requested_deformation(
            structure=structure, pressure=0.0
        )
        self.assertIs(result, structure)

    def test_diagonal_pressure_unchanged(self):
        structure = bulk("Al", a=4.0, cubic=True)
        result = _modify_structure_to_allow_requested_deformation(
            structure=structure, pressure=[0.0, 0.0, 0.0, None, None, None]
        )
        self.assertIs(result, structure)

    def test_non_diagonal_pressure_skews_cell(self):
        structure = bulk("Al", a=4.0, cubic=True)
        result = _modify_structure_to_allow_requested_deformation(
            structure=structure, pressure=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        )
        self.assertIsNot(result, structure)

    def test_non_diagonal_pressure_already_skewed(self):
        structure = bulk("Al")
        cell = structure.cell.array.copy()
        cell[0, 1] = cell[0, 0] * 0.6
        structure.set_cell(cell, scale_atoms=False)
        result = _modify_structure_to_allow_requested_deformation(
            structure=structure, pressure=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        )
        self.assertIsNotNone(result)

    def test_non_diagonal_prism_attr_error_warns(self):
        from unittest.mock import MagicMock

        structure = bulk("Al", a=4.0, cubic=True)
        mock_prism = MagicMock()
        mock_prism.is_skewed.side_effect = AttributeError("no is_skewed")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _modify_structure_to_allow_requested_deformation(
                structure=structure,
                pressure=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                prism=mock_prism,
            )
        self.assertIs(result, structure)
        self.assertTrue(
            any("constraining" in str(warning.message) for warning in w)
        )


if __name__ == "__main__":
    unittest.main()
