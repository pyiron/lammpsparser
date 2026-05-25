import unittest
from src.lammpsparser.output_raw import (
    to_amat,
    parse_raw_dump_from_text,
    parse_raw_lammps_log,
)


class TestOutputRaw(unittest.TestCase):
    def test_to_amat_9_values(self):
        result = to_amat([0, 1, 0, 0, 1, 0, 0, 1, 0])
        self.assertEqual(result, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_to_amat_6_values(self):
        result = to_amat([0, 1, 0, 1, 0, 1])
        self.assertEqual(result, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_to_amat_invalid_values(self):
        with self.assertRaises(ValueError):
            to_amat([0, 1, 2, 3])

    def test_parse_raw_dump_from_text_jagged(self):
        data = parse_raw_dump_from_text("tests/static/jagged_dump/dump.out")
        self.assertEqual(len(data["steps"]), 2)
        self.assertEqual(data["natoms"], [1, 2])
        self.assertEqual(len(data["cells"]), 2)
        self.assertEqual(len(data["indices"]), 2)
        self.assertEqual(len(data["forces"]), 2)
        self.assertEqual(len(data["velocities"]), 2)
        self.assertEqual(len(data["positions"]), 2)
        self.assertEqual(len(data["unwrapped_positions"]), 2)
        self.assertEqual(len(data["computes"]), 0)

    def test_parse_raw_lammps_log_multiple_thermo(self):
        df = parse_raw_lammps_log("tests/static/multiple_thermo/log.lammps")
        self.assertEqual(len(df), 2)
        self.assertIn("LogStep", df.columns)
        self.assertEqual(df["LogStep"].nunique(), 2)

    def test_parse_raw_lammps_log_no_pressure(self):
        df = parse_raw_lammps_log("tests/static/no_pressure/log.lammps")
        self.assertEqual(len(df), 1)
        self.assertNotIn("Press", df.columns)

    def test_parse_raw_dump_from_text_mean_fields(self):
        data = parse_raw_dump_from_text("tests/static/mean_dump/dump.out")
        self.assertEqual(len(data["steps"]), 1)
        self.assertEqual(len(data["mean_forces"]), 1)
        self.assertEqual(len(data["mean_velocities"]), 1)
        self.assertEqual(len(data["mean_unwrapped_positions"]), 1)
        self.assertIn("test", data["computes"])
        self.assertEqual(len(data["mean_unwrapped_positions"]), 1)
        self.assertEqual(len(data["computes"]["test"]), 1)
