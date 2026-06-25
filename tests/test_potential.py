import os
import unittest
import shutil
import tempfile
from unittest import mock

from ase.build import bulk
import pandas
from lammpsparser.potential import (
    validate_potential_dataframe,
    get_potential_dataframe,
    get_potential_by_name,
    convert_path_to_abs_posix,
    get_resource_path_from_conda,
    find_potential_file_base,
    view_potentials,
    PotentialAbstract,
    LammpsPotentialFile,
    PotentialAvailable,
)


class TestPotential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resource_path = os.path.abspath(
            os.path.join("..", os.path.dirname(__file__), "static", "potential")
        )
        cls.potential_dataframe = get_potential_dataframe(
            structure=bulk("Al"),
            resource_path=cls.resource_path,
        )

    def test_validate_potential_dataframe(self):
        with self.assertRaises(ValueError):
            _ = validate_potential_dataframe(potential_dataframe=pandas.DataFrame({}))
        with self.assertRaises(ValueError):
            _ = validate_potential_dataframe(
                potential_dataframe=pandas.DataFrame({"a": [1, 2]})
            )
        with self.assertRaises(TypeError):
            _ = validate_potential_dataframe(potential_dataframe=0)
        series = validate_potential_dataframe(
            potential_dataframe=pandas.DataFrame({"a": [1]})
        )
        self.assertTrue(isinstance(series, pandas.Series))
        series2 = validate_potential_dataframe(potential_dataframe=series)
        self.assertTrue(series2.equals(series))

    def test_get_potential_dataframe(self):
        self.assertEqual(len(self.potential_dataframe), 1)

    def test_get_potential_by_name(self):
        df = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=self.resource_path,
        )
        self.assertEqual(df.Name, "1999--Mishin-Y--Al--LAMMPS--ipr1")

    @mock.patch("lammpsparser.potential.get_resource_path_from_conda")
    def test_get_potential_dataframe_resource_path_from_conda(self, mock_conda):
        mock_conda.return_value = self.resource_path
        df = get_potential_dataframe(structure=bulk("Al"), resource_path=None)
        mock_conda.assert_called_once()
        self.assertEqual(len(df), 1)

    @mock.patch("lammpsparser.potential.get_resource_path_from_conda")
    def test_get_potential_by_name_resource_path_from_conda(self, mock_conda):
        mock_conda.return_value = self.resource_path
        df = get_potential_by_name(
            potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1",
            resource_path=None,
        )
        mock_conda.assert_called_once()
        self.assertEqual(df.Name, "1999--Mishin-Y--Al--LAMMPS--ipr1")

    def test_convert_path_to_abs_posix(self):
        self.assertEqual(
            convert_path_to_abs_posix(path="~/test"),
            os.path.abspath(os.path.expanduser("~/test")).replace("\\", "/"),
        )
        self.assertEqual(
            convert_path_to_abs_posix(path="."),
            os.path.abspath(".").replace("\\", "/"),
        )

    @mock.patch.dict(os.environ, {"CONDA_PREFIX": "/tmp/conda"}, clear=True)
    def test_get_resource_path_from_conda(self):
        with self.assertRaises(ValueError):
            get_resource_path_from_conda()

        os.makedirs("/tmp/conda/share/iprpy")
        self.assertEqual(
            get_resource_path_from_conda().replace("\\", "/"), "/tmp/conda/share/iprpy"
        )
        shutil.rmtree("/tmp/conda")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_resource_path_from_conda_no_env(self):
        with self.assertRaises(ValueError):
            get_resource_path_from_conda()

    def test_find_potential_file_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resource_path = os.path.join(tmpdir, "resources")
            os.makedirs(resource_path)
            rel_path = "potentials"
            os.makedirs(os.path.join(resource_path, rel_path))
            file_path = os.path.join(resource_path, rel_path, "test.pot")
            with open(file_path, "w") as f:
                f.write("test")

            self.assertEqual(
                find_potential_file_base(
                    path="test.pot",
                    resource_path_lst=[resource_path],
                    rel_path=rel_path,
                ),
                file_path,
            )

            # Test finding file in direct path
            file_path_direct = os.path.join(resource_path, "test_direct.pot")
            with open(file_path_direct, "w") as f:
                f.write("test")

            self.assertEqual(
                find_potential_file_base(
                    path="test_direct.pot",
                    resource_path_lst=[resource_path],
                    rel_path=rel_path,
                ),
                file_path_direct,
            )

            with self.assertRaises(ValueError):
                find_potential_file_base(
                    path="notfound.pot",
                    resource_path_lst=[resource_path],
                    rel_path=rel_path,
                )

            with self.assertRaises(ValueError):
                find_potential_file_base(
                    path=None,
                    resource_path_lst=[resource_path],
                    rel_path=rel_path,
                )

    def test_view_potentials(self):
        df = view_potentials(
            structure=bulk("Al"),
            resource_path=self.resource_path,
        )
        self.assertEqual(len(df), 1)


class TestPotentialAbstract(unittest.TestCase):
    def setUp(self):
        self.df = pandas.DataFrame(
            {
                "Name": ["pot1", "pot2"],
                "Species": [["Al", "Ni"], ["Ni", "H"]],
            }
        )
        self.potential = PotentialAbstract(potential_df=self.df)

    def test_find(self):
        df_Al = self.potential.find("Al")
        self.assertEqual(len(df_Al), 1)
        self.assertEqual(df_Al.Name.values[0], "pot1")

        df_Ni = self.potential.find("Ni")
        self.assertEqual(len(df_Ni), 2)

        df_Al_Ni = self.potential.find(["Al", "Ni"])
        self.assertEqual(len(df_Al_Ni), 1)
        self.assertEqual(df_Al_Ni.Name.values[0], "pot1")

        df_H_Ni = self.potential.find({"H", "Ni"})
        self.assertEqual(len(df_H_Ni), 1)
        self.assertEqual(df_H_Ni.Name.values[0], "pot2")

        with self.assertRaises(TypeError):
            self.potential.find(123)

    def test_find_by_name(self):
        df = self.potential.find_by_name("pot1")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.Name.values[0], "pot1")

        with self.assertRaises(ValueError):
            self.potential.find_by_name("not_a_potential")

    def test_list(self):
        self.assertTrue(self.potential.list().equals(self.df))

    def test_getitem(self):
        potential = self.potential["Al"]
        self.assertEqual(len(potential.list()), 1)
        self.assertEqual(potential.list().Name.values[0], "pot1")

    def test_str(self):
        self.assertEqual(str(self.potential), str(self.df))

    def test_get_potential_df(self):
        with self.assertRaises(ValueError):
            self.potential._get_potential_df(
                file_name_lst={"not_a_file.csv"}, resource_path="."
            )


class TestLammpsPotentialFile(unittest.TestCase):
    def setUp(self):
        self.resource_path = os.path.abspath(
            os.path.join("..", os.path.dirname(__file__), "static", "potential")
        )
        self.potential = LammpsPotentialFile(resource_path=self.resource_path)
        self.potential_df = LammpsPotentialFile(
            resource_path=self.resource_path,
            default_df=pandas.DataFrame(
                index=["Al"], data={"Name": ["1999--Mishin-Y--Al--LAMMPS--ipr1"]}
            ),
            selected_atoms=[],
        )

    def test_init(self):
        self.assertIsInstance(self.potential, LammpsPotentialFile)
        self.assertIsInstance(self.potential, PotentialAbstract)

    def test_find_default(self):
        # This test requires a default_df, which is not set up in the test data
        self.assertIsNone(self.potential.find_default("Al"))
        self.assertIsNone(self.potential.find_default(["Al"]))
        self.assertIsNone(self.potential.default())
        self.assertEqual(
            self.potential_df.find_default("Al")["Name"].values[0],
            "1999--Mishin-Y--Al--LAMMPS--ipr1",
        )
        self.assertEqual(
            self.potential_df.Al.default()["Name"].values[0],
            "1999--Mishin-Y--Al--LAMMPS--ipr1",
        )
        with self.assertRaises(TypeError):
            self.assertIsNone(self.potential.find_default(pandas.DataFrame({})))


class TestPotentialAvailable(unittest.TestCase):
    def setUp(self):
        self.potentials = {"pot1": 1, "pot2": 2, "pot3": 3}
        self.available = PotentialAvailable(list_of_potentials=self.potentials)

    def test_repr(self):
        self.assertEqual(repr(self.available), str(dir(self.available)))

    def test_get_attr(self):
        self.assertEqual(str(self.available.pot_pot1), "pot1")

    def test_get_attr_missing(self):
        with self.assertRaises(AttributeError):
            self.available.pot_does_not_exist
