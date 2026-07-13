import os
import shutil
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


if __name__ == "__main__":
    unittest.main()
