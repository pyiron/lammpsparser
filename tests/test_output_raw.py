import unittest
from src.lammpsparser.output_raw import (
    to_amat,
    parse_raw_dump_from_text,
    parse_raw_lammps_log,
    _iter_raw_frames,
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


class TestIterRawFrames(unittest.TestCase):
    def test_yields_all_frames(self):
        frames = list(_iter_raw_frames("tests/static/jagged_dump/dump.out"))
        self.assertEqual(len(frames), 2)

    def test_frame_keys(self):
        frame = next(_iter_raw_frames("tests/static/jagged_dump/dump.out"))
        for key in ["steps", "natoms", "cells", "indices", "forces", "velocities",
                    "positions", "unwrapped_positions", "computes"]:
            self.assertIn(key, frame)

    def test_start_stop(self):
        frames = list(_iter_raw_frames("tests/static/jagged_dump/dump.out", start=1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["steps"], 1)

    def test_step_slice(self):
        # jagged_dump has 2 frames; step=2 should yield only frame 0
        frames = list(_iter_raw_frames("tests/static/jagged_dump/dump.out", step=2))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["steps"], 0)

    def test_truncated_file_raises(self):
        import tempfile, os
        # Write a dump with a started-but-unfinished frame
        content = (
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n-1 1\n-1 1\n-1 1\n"
            "ITEM: ATOMS id type xsu ysu zsu fx fy fz vx vy vz\n"
            "1 1 0 0 0 0 0 0 0 0 0\n"
            # second atom line missing — truncated
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                list(_iter_raw_frames(path))
        finally:
            os.unlink(path)
