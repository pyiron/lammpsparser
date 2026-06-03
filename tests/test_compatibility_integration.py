import os
import unittest
import numpy as np
from ase.build import bulk
from ase.constraints import FixAtoms
from lammpsparser.compatibility.file import lammps_file_interface_function

try:
    import lammps

    skip_lammps_test = False
except ImportError:
    skip_lammps_test = True


@unittest.skipIf(skip_lammps_test, "LAMMPS executable not available")
class TestLammpsIntegration(unittest.TestCase):
    def setUp(self):
        self.static_path = os.path.abspath(
            os.path.join("..", os.path.dirname(__file__), "static")
        )
        structure = bulk("Al", cubic=True)
        c = FixAtoms(
            indices=[atom.index for i, atom in enumerate(structure) if i % 2 == 0]
        )
        structure.set_constraint(c)
        self.structure = structure
        self.potential = "1999--Mishin-Y--Al--LAMMPS--ipr1"

    def test_lammps_integration(self):
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=os.path.abspath("lmp_working_directory"),
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs={
                "pressure": 0.0,
                "temperature": 500.0,
                "seed": 12345,
                "n_ionic_steps": 1000,
            },
            units="metal",
            lmp_command="lmp_mpi -in lmp.in",
        )
        self.assertFalse(job_crashed)
        self.assertTrue(np.sum(parsed_output["generic"]["forces"][:, 0]) == 0.0)
        self.assertTrue(np.sum(parsed_output["generic"]["forces"][:, 2]) == 0.0)
