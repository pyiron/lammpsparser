import os
import shutil
import tempfile
import unittest

import numpy as np

from lammpsparser.compatibility.mlip import (
    MlipConfiguration,
    check_mlip_convergence,
    get_mlip_selected_structures,
    load_mlip_cfgs,
    write_mlip_input_file,
)

STATIC_MLIP_DIR = os.path.join(os.path.dirname(__file__), "static", "mlip")


class TestWriteMlipInputFile(unittest.TestCase):
    def setUp(self):
        self.working_directory = os.path.abspath("mlip_working_directory")

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_creates_working_directory(self):
        self.assertFalse(os.path.exists(self.working_directory))
        write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
        )
        self.assertTrue(os.path.exists(self.working_directory))

    def test_default_content(self):
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
        )
        self.assertEqual(file_path, os.path.join(self.working_directory, "mlip.ini"))
        with open(file_path) as f:
            content = f.read()
        self.assertEqual(
            content,
            "mtp-filename /abs/path/to/pot.mtp\nselect FALSE\n",
        )

    def test_active_learning_content(self):
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
            active_learning=True,
            threshold=2.5,
            threshold_break=6.0,
        )
        with open(file_path) as f:
            content = f.read()
        self.assertEqual(
            content,
            "mtp-filename /abs/path/to/pot.mtp\n"
            "calculate-efs TRUE\n"
            "select TRUE\n"
            "select:threshold 2.5\n"
            "select:threshold-break 6.0\n"
            "select:save-selected selected.cfg\n"
            "select:load-state state.mvs\n"
            "select:log selection.log\n"
            "write-cfgs:skip 0\n",
        )

    def test_active_learning_custom_parameters(self):
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
            active_learning=True,
            threshold=3.0,
            threshold_break=7.5,
            save_selected="my_selected.cfg",
            load_state="my_state.mvs",
            log="my_selection.log",
        )
        with open(file_path) as f:
            content = f.read()
        self.assertEqual(
            content,
            "mtp-filename /abs/path/to/pot.mtp\n"
            "calculate-efs TRUE\n"
            "select TRUE\n"
            "select:threshold 3.0\n"
            "select:threshold-break 7.5\n"
            "select:save-selected my_selected.cfg\n"
            "select:load-state my_state.mvs\n"
            "select:log my_selection.log\n"
            "write-cfgs:skip 0\n",
        )

    def test_custom_file_name(self):
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
            file_name="custom_mlip.ini",
        )
        self.assertEqual(
            file_path, os.path.join(self.working_directory, "custom_mlip.ini")
        )
        self.assertTrue(os.path.exists(file_path))

    def test_existing_working_directory(self):
        os.makedirs(self.working_directory, exist_ok=True)
        file_path = write_mlip_input_file(
            working_directory=self.working_directory,
            mtp_filename="/abs/path/to/pot.mtp",
        )
        self.assertTrue(os.path.exists(file_path))


class TestCheckMlipConvergence(unittest.TestCase):
    def setUp(self):
        self.working_directory = os.path.abspath("mlip_convergence_working_directory")
        os.makedirs(self.working_directory, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_no_error_file_is_converged(self):
        self.assertTrue(
            check_mlip_convergence(working_directory=self.working_directory)
        )

    def test_error_file_without_breaking_line_is_converged(self):
        with open(os.path.join(self.working_directory, "error.out"), "w") as f:
            f.write("Some unrelated warning\n")
        self.assertTrue(
            check_mlip_convergence(working_directory=self.working_directory)
        )

    def test_error_file_with_breaking_line_is_not_converged(self):
        with open(os.path.join(self.working_directory, "error.out"), "w") as f:
            f.write("MLIP: Breaking threshold exceeded, stopping\n")
        self.assertFalse(
            check_mlip_convergence(working_directory=self.working_directory)
        )

    def test_custom_error_file_name_not_present_is_converged(self):
        self.assertTrue(
            check_mlip_convergence(
                working_directory=self.working_directory,
                error_file_name="custom_error.out",
            )
        )

    def test_custom_error_file_name_with_breaking_line_is_not_converged(self):
        with open(os.path.join(self.working_directory, "custom_error.out"), "w") as f:
            f.write("MLIP: Breaking threshold exceeded, stopping\n")
        self.assertFalse(
            check_mlip_convergence(
                working_directory=self.working_directory,
                error_file_name="custom_error.out",
            )
        )

    def test_breaking_line_in_middle_of_file_is_not_converged(self):
        with open(os.path.join(self.working_directory, "error.out"), "w") as f:
            f.write("Some preamble\n")
            f.write("MLIP: Breaking threshold exceeded, stopping\n")
            f.write("Some trailing line\n")
        self.assertFalse(
            check_mlip_convergence(working_directory=self.working_directory)
        )


class TestLoadMlipCfgs(unittest.TestCase):
    def test_parses_two_configurations(self):
        configurations = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "selected.cfg"))
        self.assertEqual(len(configurations), 2)

    def test_first_configuration_has_full_data(self):
        cfg = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "selected.cfg"))[0]
        self.assertIsInstance(cfg, MlipConfiguration)
        np.testing.assert_allclose(cfg.cell, np.eye(3) * 5.680600067138671875)
        np.testing.assert_allclose(
            cfg.positions,
            [
                [0.0, 0.0, 0.0],
                [2.840300033569336, 2.840300033569336, 2.840300033569336],
            ],
        )
        np.testing.assert_array_equal(cfg.types, [0, 1])
        np.testing.assert_allclose(
            cfg.forces, [[-0.01, -0.02, -0.03], [0.01, 0.02, 0.03]]
        )
        self.assertAlmostEqual(cfg.energy, -14.493988037109375)
        np.testing.assert_allclose(
            cfg.stress,
            [
                -0.239923946063382,
                -0.239923946063382,
                -0.239923946063382,
                0.0,
                0.0,
                0.0,
            ],
        )
        self.assertAlmostEqual(cfg.grade, 3.19183351)

    def test_second_configuration_has_only_required_data(self):
        cfg = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "selected.cfg"))[1]
        np.testing.assert_allclose(cfg.cell, np.eye(3) * 4.0)
        np.testing.assert_allclose(cfg.positions, [[1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(cfg.types, [0])
        self.assertIsNone(cfg.forces)
        self.assertIsNone(cfg.energy)
        self.assertIsNone(cfg.stress)
        self.assertIsNone(cfg.grade)

    def test_direct_coordinates_converted_to_cartesian(self):
        cfg = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "direct_coords.cfg"))[0]
        cell = np.eye(3) * 4.0
        np.testing.assert_allclose(cfg.cell, cell)
        np.testing.assert_allclose(
            cfg.positions,
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        )
        np.testing.assert_array_equal(cfg.types, [0, 1])
        np.testing.assert_allclose(cfg.forces, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        self.assertAlmostEqual(cfg.energy, -10.0)

    def test_reordered_plus_stress_parsed_to_voigt_order(self):
        cfg = load_mlip_cfgs(os.path.join(STATIC_MLIP_DIR, "reordered_stress.cfg"))[0]
        # File has: xy=0.6, xz=0.5, yz=0.4, zz=0.3, yy=0.2, xx=0.1
        # Expected Voigt order: (xx, yy, zz, yz, xz, xy)
        np.testing.assert_allclose(cfg.stress, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_end_cfg_without_begin_cfg_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as tmp:
            tmp.write("END_CFG\n")
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                load_mlip_cfgs(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_missing_supercell_raises(self):
        content = (
            "BEGIN_CFG\n"
            " Size\n"
            "    1\n"
            " AtomData:  id type       cartes_x      cartes_y      cartes_z\n"
            "             1    0      0.0      0.0      0.0\n"
            "END_CFG\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                load_mlip_cfgs(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_missing_atomdata_raises(self):
        content = (
            "BEGIN_CFG\n"
            " Size\n"
            "    1\n"
            " Supercell\n"
            "         4.0         0.0         0.0\n"
            "         0.0         4.0         0.0\n"
            "         0.0         0.0         4.0\n"
            "END_CFG\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                load_mlip_cfgs(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_atomdata_before_size_raises(self):
        content = (
            "BEGIN_CFG\n"
            " Supercell\n"
            "         4.0         0.0         0.0\n"
            "         0.0         4.0         0.0\n"
            "         0.0         0.0         4.0\n"
            " AtomData:  id type       cartes_x      cartes_y      cartes_z\n"
            "             1    0      0.0      0.0      0.0\n"
            "END_CFG\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                load_mlip_cfgs(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as tmp:
            tmp.write("")
            tmp_path = tmp.name
        try:
            self.assertEqual(load_mlip_cfgs(tmp_path), [])
        finally:
            os.unlink(tmp_path)


class TestGetMlipSelectedStructures(unittest.TestCase):
    def test_builds_atoms_with_species_and_metadata(self):
        structures = get_mlip_selected_structures(
            file_name=os.path.join(STATIC_MLIP_DIR, "selected.cfg"),
            species=["Al", "Ni"],
        )
        self.assertEqual(len(structures), 2)

        first = structures[0]
        self.assertEqual(first.get_chemical_symbols(), ["Al", "Ni"])
        np.testing.assert_allclose(
            first.arrays["forces"], [[-0.01, -0.02, -0.03], [0.01, 0.02, 0.03]]
        )
        self.assertAlmostEqual(first.info["energy"], -14.493988037109375)
        np.testing.assert_allclose(
            first.info["stress"],
            [
                -0.239923946063382,
                -0.239923946063382,
                -0.239923946063382,
                0.0,
                0.0,
                0.0,
            ],
        )
        self.assertAlmostEqual(first.info["mv_grade"], 3.19183351)

        second = structures[1]
        self.assertEqual(second.get_chemical_symbols(), ["Al"])
        self.assertNotIn("forces", second.arrays)
        self.assertNotIn("energy", second.info)
        self.assertNotIn("stress", second.info)
        self.assertNotIn("mv_grade", second.info)

    def test_pbc_is_true(self):
        structures = get_mlip_selected_structures(
            file_name=os.path.join(STATIC_MLIP_DIR, "selected.cfg"),
            species=["Al", "Ni"],
        )
        for atoms in structures:
            self.assertTrue(all(atoms.pbc))

    def test_cell_is_set(self):
        structures = get_mlip_selected_structures(
            file_name=os.path.join(STATIC_MLIP_DIR, "selected.cfg"),
            species=["Al", "Ni"],
        )
        np.testing.assert_allclose(
            structures[0].cell[:], np.eye(3) * 5.680600067138671875
        )

    def test_species_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            get_mlip_selected_structures(
                file_name=os.path.join(STATIC_MLIP_DIR, "selected.cfg"),
                species=["Al"],
            )


class TestMlipConfiguration(unittest.TestCase):
    def test_required_fields_only(self):
        cell = np.eye(3) * 3.0
        positions = np.array([[0.0, 0.0, 0.0]])
        types = np.array([0])
        cfg = MlipConfiguration(cell=cell, positions=positions, types=types)
        np.testing.assert_array_equal(cfg.cell, cell)
        np.testing.assert_array_equal(cfg.positions, positions)
        np.testing.assert_array_equal(cfg.types, types)
        self.assertIsNone(cfg.forces)
        self.assertIsNone(cfg.energy)
        self.assertIsNone(cfg.stress)
        self.assertIsNone(cfg.grade)

    def test_all_fields(self):
        cell = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]])
        types = np.array([0, 1])
        forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        stress = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
        cfg = MlipConfiguration(
            cell=cell,
            positions=positions,
            types=types,
            forces=forces,
            energy=-10.0,
            stress=stress,
            grade=2.5,
        )
        np.testing.assert_array_equal(cfg.forces, forces)
        self.assertAlmostEqual(cfg.energy, -10.0)
        np.testing.assert_array_equal(cfg.stress, stress)
        self.assertAlmostEqual(cfg.grade, 2.5)


if __name__ == "__main__":
    unittest.main()
