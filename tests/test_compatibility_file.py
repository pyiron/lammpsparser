import unittest
import os
import shutil
from ase.build import bulk
import pandas
from lammpsparser.compatibility.file import (
    lammps_file_interface_function,
    _get_potential,
)
from lammpsparser.potential import get_potential_by_name
from lammpsparser.compatibility.data import CalcMDInput, CalcMinimizeInput


class TestCompatibilityFile(unittest.TestCase):
    def setUp(self):
        self.working_dir = os.path.abspath(os.path.join(__file__, "..", "lmp"))
        self.static_path = os.path.abspath(
            os.path.join("..", os.path.dirname(__file__), "static")
        )
        self.structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        self.potential = "1999--Mishin-Y--Al--LAMMPS--ipr1"
        self.units = "metal"
        self.keys = [
            "steps",
            "natoms",
            "cells",
            "indices",
            "forces",
            "velocities",
            "unwrapped_positions",
            "positions",
            "temperature",
            "energy_pot",
            "energy_tot",
            "volume",
            "pressures",
        ]

    def tearDown(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

    def test_calc_error(self):
        with self.assertRaises(TypeError):
            lammps_file_interface_function(
                working_directory=self.working_dir,
                structure=self.structure,
                potential=1,
                calc_mode="static",
                units=self.units,
                resource_path=os.path.join(self.static_path, "potential"),
            )
        with self.assertRaises(ValueError):
            lammps_file_interface_function(
                working_directory=self.working_dir,
                structure=self.structure,
                potential=self.potential,
                calc_mode="error",
                units=self.units,
                resource_path=os.path.join(self.static_path, "potential"),
            )
        with self.assertRaises(NotImplementedError):
            lammps_file_interface_function(
                working_directory=self.working_dir,
                structure=self.structure,
                potential=self.potential,
                units="error",
                calc_mode="md",
                resource_path=os.path.join(self.static_path, "potential"),
            )
        with self.assertRaises(NotImplementedError):
            lammps_file_interface_function(
                working_directory=self.working_dir,
                structure=self.structure,
                potential=self.potential,
                units="error",
                calc_mode="minimize",
                resource_path=os.path.join(self.static_path, "potential"),
            )
        with self.assertRaises(ValueError):
            lammps_file_interface_function(
                working_directory=self.working_dir,
                structure=self.structure,
                potential=self.potential,
                units=self.units,
                calc_kwargs={"seed": -1},
                calc_mode="md",
                resource_path=os.path.join(self.static_path, "potential"),
            )

    def test_calc_md_npt(self):
        md_input = CalcMDInput(
            temperature=500.0,
            pressure=0.0,
            n_ionic_steps=1000,
            n_print=100,
        )
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=get_potential_by_name(
                potential_name=self.potential,
                resource_path=os.path.join(self.static_path, "potential"),
            ),
            calc_dataclass=md_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
            input_control_file={"thermo_modify": "flush yes"},
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all npt temp 500.0 500.0 0.1 iso 0.0 0.0 1.0\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "velocity all create 1000.0 80996 dist gaussian\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify flush yes\n",
            "thermo ${thermotime}\n",
            "run 1000 \n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_md_npt_langevin(self):
        md_input = CalcMDInput(
            temperature=500.0,
            pressure=0.0,
            n_ionic_steps=1000,
            n_print=100,
            langevin=True,
        )
        potential = get_potential_by_name(
            potential_name=self.potential,
            resource_path=os.path.join(self.static_path, "potential"),
        )
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=pandas.DataFrame({k: [potential[k]] for k in potential.keys()}),
            calc_dataclass=md_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all nph iso 0.0 0.0 1.0\n",
            "fix langevin all langevin 500.0 500.0 0.1 80996 zero yes\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "velocity all create 1000.0 80996 dist gaussian\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 1000 \n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_md_nvt(self):
        md_input = CalcMDInput(
            temperature=500.0,
            n_print=100,
        )
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_dataclass=md_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all nvt temp 500.0 500.0 0.1\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "velocity all create 1000.0 80996 dist gaussian\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 1 \n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_md_nvt_restart(self):
        md_input = CalcMDInput(
            temperature=500.0,
            n_print=100,
        )
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_dataclass=md_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
            restart_file=os.path.join(self.static_path, "restart", "restart.out"),
            write_restart_file=True,
            read_restart_file=True,
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "read_restart restart.out\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all nvt temp 500.0 500.0 0.1\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "reset_timestep 0\n",
            "run 1 \n",
            "write_restart restart.out\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_md_nvt_langevin(self):
        md_input = CalcMDInput(temperature=500.0, n_print=100, langevin=True)
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_dataclass=md_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all nve\n",
            "fix langevin all langevin 500.0 500.0 0.1 80996 zero yes\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "velocity all create 1000.0 80996 dist gaussian\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 1 \n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_md_nve(self):
        md_input = CalcMDInput(n_print=100, langevin=True)
        potential = [
            "# Bouhadja et al., J. Chem. Phys. 138, 224510 (2013) \n",
            "units metal\n",
            "dimension 3\n",
            "atom_style charge\n",
            "\n",
            "# create groups ###\n",
            "group Al type 1\n",
            "group Ca type 2\n",
            "group O type 3\n",
            "group Si type 4\n",
            "\n### set charges ###\n",
            "set type 1 charge 1.8\n",
            "set type 2 charge 1.2\n",
            "set type 3 charge -1.2\n",
            "set type 4 charge 2.4\n",
            "\n### Bouhadja Born-Mayer-Huggins + Coulomb Potential Parameters ###\n",
            "pair_style born/coul/dsf 0.25 8.0\n",
            "pair_coeff 1 1 0.002900 0.068000 1.570400 14.049800 0.000000\n",
            "pair_coeff 1 2 0.003200 0.074000 1.957200 17.171000 0.000000\n",
            "pair_coeff 1 3 0.007500 0.164000 2.606700 34.574700 0.000000\n",
            "pair_coeff 1 4 0.002500 0.057000 1.505600 18.811600 0.000000\n",
            "pair_coeff 2 2 0.003500 0.080000 2.344000 20.985600 0.000000\n",
            "pair_coeff 2 3 0.007700 0.178000 2.993500 42.255600 0.000000\n",
            "pair_coeff 2 4 0.002700 0.063000 1.892400 22.990700 0.000000\n",
            "pair_coeff 3 3 0.012000 0.263000 3.643000 85.084000 0.000000\n",
            "pair_coeff 3 4 0.007000 0.156000 2.541900 46.293000 0.000000\n",
            "pair_coeff 4 4 0.001200 0.046000 1.440800 25.187300 0.000000\n",
            "\npair_modify shift yes\n",
        ]
        element_lst = ["Al", "Ca", "O", "Si"]
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=pandas.DataFrame(
                {"Config": [potential], "Species": [element_lst]}
            ),
            calc_dataclass=md_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style charge\n",
            "\n",
            "read_data lammps.data\n",
            "# Bouhadja et al., J. Chem. Phys. 138, 224510 (2013) \n",
            "# create groups ###\n",
            "group Al type 1\n",
            "group Ca type 2\n",
            "group O type 3\n",
            "group Si type 4\n",
            "### set charges ###\n",
            "set type 1 charge 1.8\n",
            "set type 2 charge 1.2\n",
            "set type 3 charge -1.2\n",
            "set type 4 charge 2.4\n",
            "### Bouhadja Born-Mayer-Huggins + Coulomb Potential Parameters ###\n",
            "pair_style born/coul/dsf 0.25 8.0\n",
            "pair_coeff 1 1 0.002900 0.068000 1.570400 14.049800 0.000000\n",
            "pair_coeff 1 2 0.003200 0.074000 1.957200 17.171000 0.000000\n",
            "pair_coeff 1 3 0.007500 0.164000 2.606700 34.574700 0.000000\n",
            "pair_coeff 1 4 0.002500 0.057000 1.505600 18.811600 0.000000\n",
            "pair_coeff 2 2 0.003500 0.080000 2.344000 20.985600 0.000000\n",
            "pair_coeff 2 3 0.007700 0.178000 2.993500 42.255600 0.000000\n",
            "pair_coeff 2 4 0.002700 0.063000 1.892400 22.990700 0.000000\n",
            "pair_coeff 3 3 0.012000 0.263000 3.643000 85.084000 0.000000\n",
            "pair_coeff 3 4 0.007000 0.156000 2.541900 46.293000 0.000000\n",
            "pair_coeff 4 4 0.001200 0.046000 1.440800 25.187300 0.000000\n",
            "pair_modify shift yes\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all nve\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 1 \n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_static(self):
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="static",
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 1 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "variable thermotime equal 1\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 0\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_minimize(self):
        minimize_input = CalcMinimizeInput(
            n_print=100,
            max_iter=20,
        )
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_dataclass=minimize_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "variable thermotime equal 20 \n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "min_style cg\n",
            "minimize 0.0 0.0001 20 2000\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_minimize_pressure(self):
        minimize_input = CalcMinimizeInput(
            pressure=0.0,
        )
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_dataclass=minimize_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 1 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "variable thermotime equal 1 \n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "fix ensemble all box/relax iso 0.0\n",
            "min_style cg\n",
            "minimize 0.0 0.0001 100000 10000000\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_minimize_pressure_3d(self):
        minimize_input = CalcMinimizeInput(
            pressure=[0.0, 0.0, 0.0],
        )
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_dataclass=minimize_input,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 1 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "variable thermotime equal 1 \n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "fix ensemble all box/relax x 0.0 y 0.0 z 0.0 couple none\n",
            "min_style cg\n",
            "minimize 0.0 0.0001 100000 10000000\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)


class TestGlassPotential(unittest.TestCase):
    def setUp(self):
        self.static_path = os.path.abspath(
            os.path.join("..", os.path.dirname(__file__), "static")
        )

    def test_bouhadja(self):
        potential = [
            "# Bouhadja et al., J. Chem. Phys. 138, 224510 (2013) \n",
            "units metal\n",
            "dimension 3\n",
            "atom_style charge\n",
            "\n",
            "# create groups ###\n",
            "group Al type 1\n",
            "group Ca type 2\n",
            "group O type 3\n",
            "group Si type 4\n",
            "\n### set charges ###\n",
            "set type 1 charge 1.8\n",
            "set type 2 charge 1.2\n",
            "set type 3 charge -1.2\n",
            "set type 4 charge 2.4\n",
            "\n### Bouhadja Born-Mayer-Huggins + Coulomb Potential Parameters ###\n",
            "pair_style born/coul/dsf 0.25 8.0\n",
            "pair_coeff 1 1 0.002900 0.068000 1.570400 14.049800 0.000000\n",
            "pair_coeff 1 2 0.003200 0.074000 1.957200 17.171000 0.000000\n",
            "pair_coeff 1 3 0.007500 0.164000 2.606700 34.574700 0.000000\n",
            "pair_coeff 1 4 0.002500 0.057000 1.505600 18.811600 0.000000\n",
            "pair_coeff 2 2 0.003500 0.080000 2.344000 20.985600 0.000000\n",
            "pair_coeff 2 3 0.007700 0.178000 2.993500 42.255600 0.000000\n",
            "pair_coeff 2 4 0.002700 0.063000 1.892400 22.990700 0.000000\n",
            "pair_coeff 3 3 0.012000 0.263000 3.643000 85.084000 0.000000\n",
            "pair_coeff 3 4 0.007000 0.156000 2.541900 46.293000 0.000000\n",
            "pair_coeff 4 4 0.001200 0.046000 1.440800 25.187300 0.000000\n",
            "\npair_modify shift yes\n",
        ]
        element_lst = ["Al", "Ca", "O", "Si"]
        potential_lst, potential_replace, species = _get_potential(
            potential=pandas.DataFrame(
                {"Config": [potential], "Species": [element_lst]}
            ),
            resource_path=os.path.join(self.static_path, "potential"),
        )
        for i, l in enumerate(potential):
            if i in [1, 2, 3]:
                self.assertFalse(l in potential_lst)
            else:
                self.assertTrue(l in potential_lst)

        for k, v in {
            "units": "units metal\n",
            "dimension": "dimension 3\n",
            "atom_style": "atom_style charge\n",
        }.items():
            self.assertEqual(potential_replace[k], v)

        self.assertEqual(species, element_lst)
