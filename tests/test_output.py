from ase.build import bulk
import numpy as np
import os
import tempfile
import unittest
from lammpsparser import parse_lammps_output_files
from lammpsparser.output import (
    remap_indices_ase,
    _parse_dump,
    _collect_output_log,
    iter_lammps_frames,
    LammpsFrame,
)
from lammpsparser.output_raw import to_amat, _iter_raw_frames, parse_raw_lammps_log
from lammpsparser.structure import UnfoldingPrism


try:
    import h5py

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


class TestLammpsOutput(unittest.TestCase):
    def setUp(self):
        self.static_folder = os.path.abspath(os.path.join(__file__, "..", "static"))

    def test_remap_indices(self):
        structure = bulk("Ag", cubic=True).repeat([2, 2, 2])
        structure.set_chemical_symbols(
            [
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Au",
                "Au",
                "Au",
                "Au",
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Au",
                "Au",
                "Au",
                "Au",
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
            ]
        )
        ind = remap_indices_ase(
            lammps_indices=[
                1,
                1,
                1,
                1,
                5,
                5,
                5,
                5,
                3,
                3,
                3,
                3,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                5,
                5,
                5,
                5,
                3,
                3,
                3,
                3,
                1,
                1,
                1,
                1,
            ],
            potential_elements=["Ag", "Al", "Cu", "Co", "Au"],
            structure=structure,
        )
        self.assertEqual(sum(ind), 24)
        ind = remap_indices_ase(
            lammps_indices=[2, 2, 2, 2],
            potential_elements=["Ag", "Al", "Cu", "Co", "Au"],
            structure=structure,
        )
        self.assertEqual(sum(ind), 8)
        ind = remap_indices_ase(
            lammps_indices=[2, 2, 2, 2],
            potential_elements=["Au", "Ag", "Cu"],
            structure=structure,
        )
        self.assertEqual(sum(ind), 0)

    def test_dump_chemical(self):
        test_folder = os.path.join(self.static_folder, "dump_chemical")
        structure_ni = bulk("Ni", cubic=True)
        structure_al = bulk("Ni", cubic=True)
        structure_all = bulk("Ni", cubic=True)
        structure_ni.set_chemical_symbols(["H", "Ni", "Ni", "Ni"])
        structure_al.set_chemical_symbols(["H", "Al", "Al", "Al"])
        structure_all.set_chemical_symbols(["Al", "Al", "Ni", "H"])
        for l, s, ind in zip(
            ["dump_NiH.out", "dump_AlH.out", "dump_NiAlH.out"],
            [structure_ni, structure_al, structure_all],
            [np.array([0, 1, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 2])],
        ):
            output = _parse_dump(
                dump_h5_full_file_name="",
                dump_out_full_file_name=os.path.join(test_folder, l),
                prism=UnfoldingPrism(cell=s.cell),
                structure=s,
                potential_elements=["Ni", "Al", "H"],
                remap_indices_funct=remap_indices_ase,
            )
            self.assertEqual(output["steps"], [0])
            self.assertEqual(output["natoms"], [4])
            self.assertTrue(np.all(np.equal(output["indices"][0], ind)))
            self.assertTrue(
                np.all(
                    np.isclose(
                        output["forces"],
                        [
                            np.array(
                                [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                ]
                            )
                        ],
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.isclose(
                        output["velocities"],
                        [
                            np.array(
                                [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                ]
                            )
                        ],
                    )
                )
            )

    def test_empty_job_output(self):
        structure_ni = bulk("Ni", cubic=True)
        structure_ni.set_chemical_symbols(["H", "Ni", "Ni", "Ni"])
        output_dict = parse_lammps_output_files(
            working_directory=os.path.join(self.static_folder, "dump_chemical"),
            structure=structure_ni,
            potential_elements=["Ni", "Al", "H"],
            units="metal",
            prism=None,
            dump_h5_file_name="dump.h5",
            dump_out_file_name="dump_NiH.out",
            log_lammps_file_name="log.lammps",
            remap_indices_funct=remap_indices_ase,
        )
        self.assertEqual(len(output_dict["generic"].keys()), 8)
        with self.assertRaises(FileNotFoundError):
            _parse_dump(
                dump_h5_full_file_name=os.path.join(self.static_folder, "empty"),
                dump_out_full_file_name=os.path.join(self.static_folder, "empty"),
                prism=None,
                structure=structure_ni,
                potential_elements=["Ni", "Al", "H"],
            )

    def test_full_job_output(self):
        test_folder = os.path.join(self.static_folder, "full_job")
        structure_ni = bulk("Ni", cubic=True)
        output_dict = parse_lammps_output_files(
            working_directory=test_folder,
            structure=structure_ni,
            potential_elements=["Ni", "Al", "H"],
            units="metal",
            prism=UnfoldingPrism(structure_ni.cell),
            dump_h5_file_name="dump.h5",
            dump_out_file_name="dump.out",
            log_lammps_file_name="log.lammps",
            remap_indices_funct=remap_indices_ase,
        )
        self.assertEqual(output_dict["generic"]["steps"], np.array([0]))
        self.assertEqual(output_dict["generic"]["natoms"], np.array([4.0]))
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["energy_pot"], np.array([-17.80000005])
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["energy_tot"], np.array([-17.80000005])
                )
            )
        )
        self.assertTrue(
            np.all(np.isclose(output_dict["generic"]["volume"], np.array([43.614208])))
        )
        self.assertEqual(output_dict["generic"]["temperature"], np.array([0.0]))
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["pressures"],
                    np.array(
                        [
                            [
                                [5.38768850e-05, -4.07839310e-16, -1.83528844e-15],
                                [-4.07839310e-16, 5.38768850e-05, -1.01960322e-15],
                                [-1.83528844e-15, -1.01960322e-15, 5.38768850e-05],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["positions"],
                    np.array(
                        [
                            [
                                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                                [0.00000000e00, 1.76000000e00, 1.76000000e00],
                                [1.76000000e00, 1.07768918e-16, 1.76000000e00],
                                [1.76000000e00, 1.76000000e00, 2.15537837e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["unwrapped_positions"],
                    np.array(
                        [
                            [
                                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                                [0.00000000e00, 1.76000000e00, 1.76000000e00],
                                [1.76000000e00, 1.07768918e-16, 1.76000000e00],
                                [1.76000000e00, 1.76000000e00, 2.15537837e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["velocities"],
                    np.array(
                        [
                            [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["forces"],
                    np.array(
                        [
                            [
                                [-2.22044605e-16, -1.38777878e-17, -5.55111512e-17],
                                [-5.55111512e-17, -1.66533454e-16, 6.93889390e-17],
                                [-5.55111512e-17, -3.39907768e-33, -2.08166817e-16],
                                [0.00000000e00, -6.93889390e-18, -2.42861287e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["cells"],
                    np.array(
                        [
                            [
                                [3.52000000e00, 2.15537837e-16, 2.15537837e-16],
                                [0.00000000e00, 3.52000000e00, 2.15537837e-16],
                                [0.00000000e00, 0.00000000e00, 3.52000000e00],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["indices"],
                    np.array([[0, 0, 0, 0]]),
                )
            )
        )

    @unittest.skipIf(skip_h5py_test, "h5py not available")
    def test_full_job_output_h5(self):
        test_folder = os.path.join(self.static_folder, "full_job_h5")
        structure_ni = bulk("Ni", cubic=True)
        output_dict = parse_lammps_output_files(
            working_directory=test_folder,
            structure=structure_ni,
            potential_elements=["Ni", "Al", "H"],
            units="metal",
            prism=UnfoldingPrism(structure_ni.cell),
            dump_h5_file_name="dump.h5",
            dump_out_file_name="dump.out",
            log_lammps_file_name="log.lammps",
            remap_indices_funct=remap_indices_ase,
        )
        self.assertEqual(output_dict["generic"]["steps"], np.array([0]))
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["energy_pot"], np.array([-17.80000005])
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["energy_tot"], np.array([-17.80000005])
                )
            )
        )
        self.assertTrue(
            np.all(np.isclose(output_dict["generic"]["volume"], np.array([43.614208])))
        )
        self.assertEqual(output_dict["generic"]["temperature"], np.array([0.0]))
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["pressures"],
                    np.array(
                        [
                            [
                                [5.38768850e-05, -4.07839310e-16, -1.83528844e-15],
                                [-4.07839310e-16, 5.38768850e-05, -1.01960322e-15],
                                [-1.83528844e-15, -1.01960322e-15, 5.38768850e-05],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["positions"],
                    np.array(
                        [
                            [
                                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                                [0.00000000e00, 1.76000000e00, 1.76000000e00],
                                [1.76000000e00, 1.07768918e-16, 1.76000000e00],
                                [1.76000000e00, 1.76000000e00, 2.15537837e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["forces"],
                    np.array(
                        [
                            [
                                [-2.22044605e-16, -1.38777878e-17, -5.55111512e-17],
                                [-5.55111512e-17, -1.66533454e-16, 6.93889390e-17],
                                [-5.55111512e-17, -3.39907768e-33, -2.08166817e-16],
                                [0.00000000e00, -6.93889390e-18, -2.42861287e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["cells"],
                    np.array(
                        [
                            [
                                [3.52000000e00, 2.15537837e-16, 2.15537837e-16],
                                [0.00000000e00, 3.52000000e00, 2.15537837e-16],
                                [0.00000000e00, 0.00000000e00, 3.52000000e00],
                            ]
                        ]
                    ),
                )
            )
        )

    def test_to_amat(self):
        out = to_amat([1, 2, 3, 4, 5, 6])
        self.assertTrue(
            np.all(np.equal(out, np.array([[1.0, 0, 0], [0.0, 1.0, 0], [0.0, 0.0, 1]])))
        )
        out = to_amat([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue(
            np.all(np.equal(out, np.array([[-8.0, 0, 0], [3, -8.0, 0], [6, 9, 1]])))
        )
        with self.assertRaises(ValueError):
            to_amat([])

    def test_collect_output_log(self):
        generic_keys_lst, pressure_dict, df = _collect_output_log(
            file_name=os.path.join(self.static_folder, "no_pressure", "log.lammps"),
            prism=UnfoldingPrism(cell=bulk("Al").cell),
        )
        self.assertEqual(
            generic_keys_lst,
            ["steps", "temperature", "energy_pot", "energy_tot", "volume"],
        )
        self.assertEqual(len(pressure_dict), 0)
        self.assertEqual(len(df), 1)

    def test_collect_output_log_multi(self):
        generic_keys_lst, pressure_dict, df = _collect_output_log(
            file_name=os.path.join(self.static_folder, "multiple_thermo", "log.lammps"),
            prism=UnfoldingPrism(cell=bulk("Al").cell),
        )
        self.assertEqual(
            generic_keys_lst,
            [
                "steps",
                "temperature",
                "energy_pot",
                "energy_tot",
                "volume",
                "LogStep",
            ],
        )
        self.assertEqual(len(pressure_dict), 0)
        self.assertEqual(len(df), 2)

    def test_mean_values(self):
        generic_keys_lst, pressure_dict, df = _collect_output_log(
            file_name=os.path.join(self.static_folder, "mean_values", "log.lammps"),
            prism=UnfoldingPrism(cell=bulk("Al").cell),
        )
        self.assertTrue("mean_foo" in generic_keys_lst)
        self.assertTrue("mean_bar" in generic_keys_lst)
        self.assertTrue(
            np.all(
                np.isclose(
                    pressure_dict["mean_pressures"],
                    np.array([[[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]]]),
                )
            )
        )
        self.assertEqual(len(df), 1)

    def test_mean_dump(self):
        s = bulk("Al")
        prism = UnfoldingPrism(cell=s.cell)
        output = _parse_dump(
            dump_h5_full_file_name="",
            dump_out_full_file_name=os.path.join(
                self.static_folder, "mean_dump", "dump.out"
            ),
            prism=prism,
            structure=s,
            potential_elements=["Al"],
            remap_indices_funct=remap_indices_ase,
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output["mean_forces"],
                    np.matmul(np.array([[[1, 2, 3]]]), prism.R.T),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output["mean_velocities"],
                    np.matmul(np.array([[[4, 5, 6]]]), prism.R.T),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output["mean_unwrapped_positions"],
                    np.matmul(np.array([[[7, 8, 9]]]), prism.R.T),
                )
            )
        )
        self.assertTrue(np.all(np.isclose(output["computes"]["test"], [[10]])))

    def test_jagged_array(self):
        structure_ni = bulk("Ni", cubic=True)
        structure_ni.set_chemical_symbols(["H", "Ni", "Ni", "Ni"])
        parse_lammps_output_files(
            working_directory=os.path.join(self.static_folder, "dump_chemical"),
            structure=structure_ni,
            potential_elements=["Ni", "Al", "H"],
            units="metal",
            prism=None,
            dump_h5_file_name="dump.h5",
            dump_out_file_name="dump_NiH.out",
            log_lammps_file_name="log.lammps",
            remap_indices_funct=remap_indices_ase,
        )
        s = bulk("Al")
        output = parse_lammps_output_files(
            working_directory=os.path.join(self.static_folder, "jagged_dump"),
            structure=s,
            potential_elements=["Al"],
            units="metal",
        )
        self.assertEqual(len(output["generic"]["positions"]), 2)
        self.assertEqual(len(output["generic"]["positions"][0]), 1)
        self.assertEqual(len(output["generic"]["positions"][1]), 2)
        self.assertTrue(isinstance(output["generic"]["positions"], list))

    def test_to_amat_triclinic(self):
        out = to_amat([0, 1, 0.1, 0, 1, 0.2, 0, 1, 0.3])
        self.assertTrue(
            np.all(
                np.isclose(
                    out,
                    [[0.7, 0.0, 0.0], [0.1, 0.7, 0.0], [0.2, 0.3, 1.0]],
                )
            )
        )

    def test_mean_values_non_ortho(self):
        cell = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        generic_keys_lst, pressure_dict, df = _collect_output_log(
            file_name=os.path.join(
                self.static_folder, "mean_values_non_ortho", "log.lammps"
            ),
            prism=UnfoldingPrism(cell=cell),
        )
        self.assertTrue("mean_foo" in generic_keys_lst)
        self.assertTrue("mean_bar" in generic_keys_lst)
        self.assertTrue("mean_pressures" in pressure_dict.keys())

    def test_mean_values_ortho_prism(self):
        # Use ortho prism to cover lines 311-312 (mean_pressures rotation with ortho cell)
        structure = bulk("Al", a=4.0, cubic=True)
        ortho_prism = UnfoldingPrism(cell=structure.cell)
        generic_keys_lst, pressure_dict, df = _collect_output_log(
            file_name=os.path.join(self.static_folder, "mean_values", "log.lammps"),
            prism=ortho_prism,
        )
        self.assertTrue("mean_pressures" in pressure_dict.keys())

    def test_parse_output_with_computes(self):
        # Cover line 95: computes in dump (mean_dump/dump.out has c_test column)
        import warnings

        structure = bulk("Al", a=3.52)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            output = parse_lammps_output_files(
                working_directory=os.path.join(self.static_folder, "mean_dump"),
                structure=structure,
                potential_elements=["Al"],
                units="metal",
            )
        self.assertIn("test", output["generic"])

    def test_parse_dump_h5md_non_ortho_raises(self):
        # Cover line 135: h5md file with non-ortho prism raises RuntimeError
        cell = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        non_ortho_prism = UnfoldingPrism(cell=cell)
        h5_file = os.path.join(self.static_folder, "full_job_h5", "dump.h5")
        with self.assertRaises(RuntimeError):
            _parse_dump(
                dump_h5_full_file_name=h5_file,
                dump_out_full_file_name="",
                prism=non_ortho_prism,
                structure=bulk("Ni", cubic=True),
                potential_elements=["Ni"],
            )

    def test_parse_output_extra_log_column(self):
        # Cover line 114: column in log that is not in generic_keys_lst goes to hdf_lammps
        import tempfile
        import warnings

        structure = bulk("Al", a=4.0, cubic=True)
        log_content = (
            "LAMMPS log\n"
            "   Step          Temp          KinEng         PotEng         TotEng"
            "          Pxx            Pxy            Pxz            Pyy            Pyz"
            "            Pzz           Volume\n"
            "         0   0.0   0.0   -17.8   -17.8   0.0   0.0   0.0   0.0   0.0   0.0   43.6\n"
            "Loop time of 0.0 on 1 procs for 0 steps with 4 atoms\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write log file with extra "KinEng" column
            log_path = os.path.join(tmpdir, "log.lammps")
            with open(log_path, "w") as f:
                f.write(log_content)
            # Copy dump.out from full_job into tmpdir
            import shutil

            dump_src = os.path.join(self.static_folder, "full_job", "dump.out")
            shutil.copy(dump_src, os.path.join(tmpdir, "dump.out"))

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                output = parse_lammps_output_files(
                    working_directory=tmpdir,
                    structure=structure,
                    potential_elements=["Al"],
                    units="metal",
                )
            # "KinEng" column is not a generic key, so it goes to hdf_lammps
            self.assertIn("KinEng", output["lammps"])


class TestIterLammpsFrames(unittest.TestCase):
    def setUp(self):
        self.static_folder = os.path.abspath(os.path.join(__file__, "..", "static"))
        self.structure = bulk("Ni", cubic=True)
        self.potential_elements = ["Ni", "Al", "H"]
        self.units = "metal"
        # compatibility_output has 11 frames and uses Al repeat structure
        self.multi_structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        self.multi_potential_elements = ["Al"]

    def _full_job_dir(self):
        return os.path.join(self.static_folder, "full_job")

    def _multi_frame_dir(self):
        return os.path.join(self.static_folder, "compatibility_output")

    def test_yields_lammps_frame_instances(self):
        frames = list(
            iter_lammps_frames(
                working_directory=self._full_job_dir(),
                structure=self.structure,
                potential_elements=self.potential_elements,
                units=self.units,
            )
        )
        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0], LammpsFrame)

    def test_frame_has_required_fields(self):
        frame = next(
            iter(
                iter_lammps_frames(
                    working_directory=self._full_job_dir(),
                    structure=self.structure,
                    potential_elements=self.potential_elements,
                    units=self.units,
                )
            )
        )
        self.assertIsInstance(frame.step, int)
        self.assertEqual(frame.cell.shape, (3, 3))
        self.assertEqual(frame.positions.ndim, 2)
        self.assertEqual(frame.positions.shape[1], 3)
        self.assertEqual(frame.forces.ndim, 2)
        self.assertEqual(frame.indices.ndim, 1)

    def test_equivalence_with_batch_parser(self):
        """Streaming all frames must produce the same positions and forces as parse_lammps_output_files."""
        frames = list(
            iter_lammps_frames(
                working_directory=self._full_job_dir(),
                structure=self.structure,
                potential_elements=self.potential_elements,
                units=self.units,
            )
        )
        batch = parse_lammps_output_files(
            working_directory=self._full_job_dir(),
            structure=self.structure,
            potential_elements=self.potential_elements,
            units=self.units,
        )
        streamed_positions = np.stack([f.positions for f in frames])
        np.testing.assert_allclose(
            streamed_positions, batch["generic"]["positions"], rtol=1e-10
        )
        streamed_forces = np.stack([f.forces for f in frames])
        np.testing.assert_allclose(
            streamed_forces, batch["generic"]["forces"], rtol=1e-10
        )

    def test_start_stop_slicing(self):
        all_frames = list(
            iter_lammps_frames(
                working_directory=self._multi_frame_dir(),
                structure=self.multi_structure,
                potential_elements=self.multi_potential_elements,
                units=self.units,
            )
        )
        sliced = list(
            iter_lammps_frames(
                working_directory=self._multi_frame_dir(),
                structure=self.multi_structure,
                potential_elements=self.multi_potential_elements,
                units=self.units,
                start=1,
                stop=3,
            )
        )
        self.assertEqual(len(sliced), 2)
        np.testing.assert_array_equal(sliced[0].positions, all_frames[1].positions)
        np.testing.assert_array_equal(sliced[1].positions, all_frames[2].positions)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            list(
                iter_lammps_frames(
                    working_directory="/nonexistent/path",
                    structure=self.structure,
                    potential_elements=self.potential_elements,
                    units=self.units,
                )
            )
        self.assertIn("Dump file not found", str(ctx.exception))

    def test_empty_positions_and_forces_when_no_xsu_columns(self):
        # Dump without xsu/ysu/zsu and without fx/fy/fz → positions and forces are empty arrays.
        dump_content = (
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 3.52\n0.0 3.52\n0.0 3.52\n"
            "ITEM: ATOMS id type\n"
            "1 1\n2 1\n"
        )
        structure = bulk("Ni", cubic=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as f:
            f.write(dump_content)
            fname = f.name
        try:
            frames = list(
                iter_lammps_frames(
                    working_directory=os.path.dirname(fname),
                    structure=structure,
                    potential_elements=["Ni"],
                    units="metal",
                    dump_out_file_name=os.path.basename(fname),
                )
            )
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].positions.shape, (0, 3))
            self.assertEqual(frames[0].forces.shape, (0, 3))
        finally:
            os.unlink(fname)

    def test_computes_are_yielded(self):
        # mean_dump/dump.out has a c_test column → computes dict must be populated.
        structure = bulk("Al", a=3.52)
        frames = list(
            iter_lammps_frames(
                working_directory=os.path.join(self.static_folder, "mean_dump"),
                structure=structure,
                potential_elements=["Al"],
                units="metal",
            )
        )
        self.assertGreater(len(frames), 0)
        self.assertIsNotNone(frames[0].computes)
        self.assertIn("test", frames[0].computes)

    def test_step_slicing(self):
        all_frames = list(
            iter_lammps_frames(
                working_directory=self._multi_frame_dir(),
                structure=self.multi_structure,
                potential_elements=self.multi_potential_elements,
                units=self.units,
            )
        )
        every_other = list(
            iter_lammps_frames(
                working_directory=self._multi_frame_dir(),
                structure=self.multi_structure,
                potential_elements=self.multi_potential_elements,
                units=self.units,
                step=2,
            )
        )
        expected_count = len(range(0, len(all_frames), 2))
        self.assertEqual(len(every_other), expected_count)
        np.testing.assert_array_equal(every_other[0].positions, all_frames[0].positions)
        np.testing.assert_array_equal(every_other[1].positions, all_frames[2].positions)


_MINIMAL_FRAME = (
    "ITEM: TIMESTEP\n0\n"
    "ITEM: NUMBER OF ATOMS\n1\n"
    "ITEM: BOX BOUNDS pp pp pp\n"
    "0.0 3.52\n0.0 3.52\n0.0 3.52\n"
    "ITEM: ATOMS id type xsu ysu zsu fx fy fz\n"
    "1 1 0.0 0.0 0.0 0.0 0.0 0.0\n"
)


def _write_tmp(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as f:
        f.write(content)
        return f.name


class TestIterRawFrames(unittest.TestCase):
    def test_garbage_lines_before_timestep_are_skipped(self):
        # Lines 162-163: non-TIMESTEP content before a valid frame must be silently skipped.
        content = "# comment\nsome garbage\n" + _MINIMAL_FRAME
        fname = _write_tmp(content)
        try:
            frames = list(_iter_raw_frames(fname))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["steps"], 0)
        finally:
            os.unlink(fname)

    def test_malformed_timestep_raises(self):
        # Lines 168-169: non-integer timestep must raise ValueError.
        content = "ITEM: TIMESTEP\nnot_an_int\n"
        fname = _write_tmp(content)
        try:
            with self.assertRaises(ValueError) as ctx:
                list(_iter_raw_frames(fname))
            self.assertIn("Malformed TIMESTEP", str(ctx.exception))
        finally:
            os.unlink(fname)

    def test_missing_number_of_atoms_header_raises(self):
        # Line 173: missing "NUMBER OF ATOMS" section header must raise ValueError.
        content = "ITEM: TIMESTEP\n0\nITEM: WRONG\n1\n"
        fname = _write_tmp(content)
        try:
            with self.assertRaises(ValueError) as ctx:
                list(_iter_raw_frames(fname))
            self.assertIn("NUMBER OF ATOMS", str(ctx.exception))
        finally:
            os.unlink(fname)

    def test_malformed_number_of_atoms_raises(self):
        # Lines 178-179: non-integer atom count must raise ValueError.
        content = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\nnot_a_number\n"
        fname = _write_tmp(content)
        try:
            with self.assertRaises(ValueError) as ctx:
                list(_iter_raw_frames(fname))
            self.assertIn("NUMBER OF ATOMS", str(ctx.exception))
        finally:
            os.unlink(fname)

    def test_missing_box_bounds_header_raises(self):
        # Line 185: missing "BOX BOUNDS" section header must raise ValueError.
        content = (
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: WRONG\n"
            "0.0 3.52\n0.0 3.52\n0.0 3.52\n"
        )
        fname = _write_tmp(content)
        try:
            with self.assertRaises(ValueError) as ctx:
                list(_iter_raw_frames(fname))
            self.assertIn("BOX BOUNDS", str(ctx.exception))
        finally:
            os.unlink(fname)

    def test_missing_atoms_header_raises(self):
        # Line 198: missing "ITEM: ATOMS" section header must raise ValueError.
        content = (
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 3.52\n0.0 3.52\n0.0 3.52\n"
            "ITEM: WRONG id type\n"
            "1 1\n"
        )
        fname = _write_tmp(content)
        try:
            with self.assertRaises(ValueError) as ctx:
                list(_iter_raw_frames(fname))
            self.assertIn("ITEM: ATOMS", str(ctx.exception))
        finally:
            os.unlink(fname)

    def test_no_xsu_columns_yields_empty_positions(self):
        # Lines 286-287: dump without xsu/ysu/zsu must yield empty positions array.
        content = (
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 3.52\n0.0 3.52\n0.0 3.52\n"
            "ITEM: ATOMS id type fx fy fz\n"
            "1 1 0.0 0.0 0.0\n"
        )
        fname = _write_tmp(content)
        try:
            frames = list(_iter_raw_frames(fname))
            self.assertEqual(len(frames), 1)
            self.assertEqual(len(frames[0]["positions"]), 0)
            self.assertEqual(len(frames[0]["unwrapped_positions"]), 0)
        finally:
            os.unlink(fname)

    def test_truncated_atom_data_raises(self):
        # Lines 216-219: file truncated mid-frame must raise ValueError.
        content = (
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n3\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 3.52\n0.0 3.52\n0.0 3.52\n"
            "ITEM: ATOMS id type xsu ysu zsu fx fy fz\n"
            "1 1 0.0 0.0 0.0 0.0 0.0 0.0\n"
            # only 1 of 3 atom lines present
        )
        fname = _write_tmp(content)
        try:
            with self.assertRaises(ValueError) as ctx:
                list(_iter_raw_frames(fname))
            self.assertIn("Truncated", str(ctx.exception))
        finally:
            os.unlink(fname)


class TestParseRawLammpsLog(unittest.TestCase):
    def test_warning_in_thermo_block_is_forwarded(self):
        # Line 424: WARNING: lines inside a thermo block must be issued as Python warnings.
        log_content = (
            "LAMMPS log\n"
            "Step Temp PotEng TotEng Volume\n"
            "0 0.0 -17.8 -17.8 43.6\n"
            "WARNING: some lammps warning (src/foo.cpp:42)\n"
            "1 0.0 -17.8 -17.8 43.6\n"
            "Loop time of 0.0 on 1 procs for 1 steps with 4 atoms\n"
        )
        import warnings

        fname = _write_tmp(log_content)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                df = parse_raw_lammps_log(fname)
            self.assertTrue(any("warning" in str(w.message).lower() for w in caught))
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
