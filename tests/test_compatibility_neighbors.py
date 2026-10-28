import unittest
import warnings

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk

from lammpsparser.compatibility.neighbors import (
    get_extended_positions,
    get_neighbors,
    get_vertical_length,
    select_index,
)


class TestSelectIndex(unittest.TestCase):
    def test_select_index(self):
        structure = bulk("NaCl", "rocksalt", a=5.6, cubic=True)
        np.testing.assert_array_equal(
            select_index(structure=structure, element="Na"), [0, 2, 4, 6]
        )
        np.testing.assert_array_equal(
            select_index(structure=structure, element="Cl"), [1, 3, 5, 7]
        )


class TestExtendedPositions(unittest.TestCase):
    def test_vertical_length(self):
        structure = Atoms("Fe", cell=[[1, 1, 0], [0, 1, 0], [0, 0, 1]], pbc=True)
        np.testing.assert_allclose(
            get_vertical_length(structure=structure), [1.0, np.sqrt(0.5), 1.0]
        )

    def test_invalid_width(self):
        with self.assertRaises(ValueError):
            get_extended_positions(structure=bulk("Al", cubic=True), width=-1)

    def test_zero_width(self):
        structure = bulk("Al", cubic=True)
        positions, indices = get_extended_positions(structure=structure, width=0)
        np.testing.assert_array_equal(positions, structure.positions)
        np.testing.assert_array_equal(indices, np.arange(len(structure)))

    def test_finite_width(self):
        structure = bulk("Al", a=4.0, cubic=True)
        positions, indices = get_extended_positions(structure=structure, width=1.0)
        self.assertGreater(len(positions), len(structure))
        self.assertEqual(len(positions), len(indices))
        self.assertTrue(np.all(indices < len(structure)))
        # Each extended position is a periodic image of the indexed atom
        diff = positions - structure.positions[indices]
        scaled = np.linalg.solve(structure.cell.T, diff.T).T
        np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-8)


class TestGetNeighbors(unittest.TestCase):
    def test_invalid_num_neighbors(self):
        with self.assertRaises(ValueError):
            get_neighbors(structure=bulk("Al", cubic=True), num_neighbors=0)

    def test_invalid_width_buffer(self):
        with self.assertRaises(ValueError):
            get_neighbors(structure=bulk("Al", cubic=True), width_buffer=-1)

    def test_too_many_neighbors(self):
        structure = Atoms("Al2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3) * 5)
        with self.assertRaises(ValueError):
            get_neighbors(structure=structure, num_neighbors=12)

    def test_fcc_num_neighbors(self):
        structure = bulk("Al", a=4.0, cubic=True)
        neigh = get_neighbors(structure=structure, num_neighbors=12)
        self.assertEqual(neigh.distances.shape, (4, 12))
        np.testing.assert_allclose(neigh.distances, 4.0 / np.sqrt(2))
        self.assertTrue(np.all(neigh.indices < len(structure)))

    def test_fcc_cutoff_radius(self):
        structure = bulk("Al", a=4.0, cubic=True).repeat(2)
        neigh = get_neighbors(structure=structure, num_neighbors=40, cutoff_radius=3.0)
        self.assertEqual(neigh.distances.shape, (32, 12))
        self.assertTrue(np.all(neigh.distances < 3.0))

    def test_cutoff_radius_warning(self):
        structure = bulk("Al", a=4.0, cubic=True).repeat(2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_neighbors(structure=structure, num_neighbors=4, cutoff_radius=3.0)
        self.assertEqual(len(w), 1)

    def test_non_periodic(self):
        structure = Atoms(
            "H2O",
            positions=[[0.76, 0.59, 0.0], [-0.76, 0.59, 0.0], [0.0, 0.0, 0.0]],
            cell=np.eye(3) * 10,
            pbc=False,
        )
        neigh = get_neighbors(structure=structure, num_neighbors=3, cutoff_radius=1.2)
        # O has two neighbours, each H only the O
        self.assertEqual(neigh.distances.shape, (3, 2))
        np.testing.assert_array_equal(
            np.sum(neigh.distances < np.inf, axis=1), [1, 1, 2]
        )
        self.assertEqual(neigh.indices[0, 0], 2)
        self.assertEqual(neigh.indices[0, 1], np.iinfo(np.int32).max)


class TestNeighborsGetBonds(unittest.TestCase):
    def test_get_bonds_fcc(self):
        structure = bulk("Al", a=4.0, cubic=True).repeat(2)
        bonds = get_neighbors(structure=structure, num_neighbors=20).get_bonds()
        self.assertEqual(len(bonds), len(structure))
        # 12 nearest and 6 second nearest neighbours, 2 of the third shell
        self.assertEqual([len(s) for s in bonds[0]["Al"]], [12, 6, 2])

    def test_get_bonds_max_shells(self):
        structure = bulk("Al", a=4.0, cubic=True).repeat(2)
        bonds = get_neighbors(structure=structure, num_neighbors=20).get_bonds(
            max_shells=1
        )
        self.assertEqual(len(bonds[0]["Al"]), 1)
        self.assertEqual(len(bonds[0]["Al"][0]), 12)

    def test_get_bonds_radius(self):
        structure = bulk("Al", a=4.0, cubic=True).repeat(2)
        bonds = get_neighbors(structure=structure, num_neighbors=20).get_bonds(
            radius=3.0
        )
        self.assertEqual([len(s) for s in bonds[0]["Al"]], [12])

    def test_get_bonds_species(self):
        structure = bulk("NaCl", "rocksalt", a=5.6, cubic=True).repeat(2)
        bonds = get_neighbors(structure=structure, num_neighbors=18).get_bonds(
            max_shells=1
        )
        self.assertEqual(len(bonds[0]["Cl"][0]), 6)
        self.assertEqual(len(bonds[0]["Na"][0]), 12)

    def test_get_bonds_varying_number_of_neighbors(self):
        structure = Atoms(
            "H2O",
            positions=[[0.76, 0.59, 0.0], [-0.76, 0.59, 0.0], [0.0, 0.0, 0.0]],
            cell=np.eye(3) * 10,
            pbc=False,
        )
        bonds = get_neighbors(
            structure=structure, num_neighbors=3, cutoff_radius=1.2
        ).get_bonds()
        self.assertEqual(bonds[0], {"O": [[2]]})
        self.assertEqual(bonds[2], {"H": [[0, 1]]})


if __name__ == "__main__":
    unittest.main()
