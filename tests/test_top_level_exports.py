import unittest


class TestTopLevelMlipExports(unittest.TestCase):
    def test_mlip_functions_importable_from_top_level(self):
        from lammpsparser import (
            check_mlip_convergence,
            get_mlip_selected_structures,
            load_mlip_cfgs,
            write_mlip_input_file,
        )

        self.assertTrue(callable(write_mlip_input_file))
        self.assertTrue(callable(check_mlip_convergence))
        self.assertTrue(callable(load_mlip_cfgs))
        self.assertTrue(callable(get_mlip_selected_structures))


if __name__ == "__main__":
    unittest.main()
