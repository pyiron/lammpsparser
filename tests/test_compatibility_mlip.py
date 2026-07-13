import os
import shutil
import unittest

from lammpsparser.compatibility.mlip import (
    check_mlip_convergence,
    write_mlip_input_file,
)


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
        self.working_directory = os.path.abspath(
            "mlip_convergence_working_directory"
        )
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


if __name__ == "__main__":
    unittest.main()
