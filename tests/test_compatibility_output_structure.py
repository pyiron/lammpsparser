"""Unit tests for :func:`lammpsparser.compatibility.output.structure_from_parsed_output`.

The function builds a new :class:`ase.Atoms` instance from a parsed LAMMPS output
dictionary.  The tests cover the basic behaviour, the optional ``wrap`` flag and
the ``index`` argument which selects a specific frame from a multi‑frame output.
"""

import unittest
import numpy as np
from ase.build import bulk
from ase.atoms import Atoms

from lammpsparser.compatibility.output import structure_from_parsed_output


class TestStructureFromParsedOutput(unittest.TestCase):
    """Test suite for ``structure_from_parsed_output``.

    The tests use a simple aluminium bulk cell (4 atoms) as the template
    structure.  ``parsed_output`` mimics the structure produced by the parser
    used elsewhere in the project – a dictionary with a ``"generic"`` key that
    contains ``indices``, ``positions``, ``velocities`` and ``cells`` arrays.
    """

    def setUp(self):
        # A small cubic Al cell – 4 atoms – provides a deterministic reference.
        self.initial = bulk("Al", a=4.0, cubic=True)
        self.natoms = len(self.initial)

        # Helper to build a parsed_output dict for a given frame index.
        def make_frame(offset=0.0):
            # Shift positions by ``offset`` (in Å) to test wrapping later.
            positions = self.initial.get_positions() + offset
            velocities = np.zeros((self.natoms, 3))
            indices = np.arange(self.natoms)
            cell = self.initial.get_cell().array
            return {
                "indices": np.array([indices]),
                "positions": np.array([positions]),
                "velocities": np.array([velocities]),
                "cells": np.array([cell]),
            }

        # Single‑frame parsed output (default index = -1).
        self.single_frame = {"generic": make_frame()}

        # Multi‑frame parsed output – two distinct frames.
        self.multi_frame = {
            "generic": {
                "indices": np.stack(
                    [np.arange(self.natoms), np.arange(self.natoms) + 10]
                ),
                "positions": np.stack(
                    [
                        self.initial.get_positions(),
                        self.initial.get_positions() + 1.0,
                    ]
                ),
                "velocities": np.stack(
                    [
                        np.zeros((self.natoms, 3)),
                        np.ones((self.natoms, 3)),
                    ]
                ),
                "cells": np.stack(
                    [
                        self.initial.get_cell().array,
                        self.initial.get_cell().array,
                    ]
                ),
            }
        }

    def test_basic_copy_and_properties(self):
        """Default call (wrap=False, index=-1) returns a correctly populated copy."""
        result = structure_from_parsed_output(self.initial, self.single_frame)

        # The function should return a *new* Atoms object, not the original.
        self.assertIsNot(result, self.initial)

        # PBC flag must be set to True.
        self.assertTrue(result.get_pbc().all())

        # Verify that the data matches the parsed output.
        np.testing.assert_array_equal(
            result.get_array("indices"), self.single_frame["generic"]["indices"][0]
        )
        np.testing.assert_allclose(
            result.get_positions(), self.single_frame["generic"]["positions"][0]
        )
        np.testing.assert_allclose(
            result.get_velocities(), self.single_frame["generic"]["velocities"][0]
        )
        np.testing.assert_allclose(
            result.get_cell().array, self.single_frame["generic"]["cells"][0]
        )

    def test_wrap_option(self):
        """When ``wrap=True`` positions outside the cell are wrapped back in."""
        # Create a parsed output where positions are shifted by one full cell.
        shifted = {
            "generic": {
                "indices": np.array([[0, 1, 2, 3]]),
                "positions": np.array(
                    [self.initial.get_positions() + self.initial.get_cell().array[0]]
                ),
                "velocities": np.zeros((1, self.natoms, 3)),
                "cells": np.array([self.initial.get_cell().array]),
            }
        }
        result = structure_from_parsed_output(self.initial, shifted, wrap=True)
        # After wrapping, positions should be equivalent to the original ones.
        np.testing.assert_allclose(result.get_positions(), self.initial.get_positions())

    def test_index_argument(self):
        """Selecting a specific frame via ``index`` returns the correct data."""
        # Choose the first frame (index 0) from the multi‑frame dict.
        result = structure_from_parsed_output(self.initial, self.multi_frame, index=0)
        np.testing.assert_array_equal(
            result.get_array("indices"), self.multi_frame["generic"]["indices"][0]
        )
        np.testing.assert_allclose(
            result.get_positions(), self.multi_frame["generic"]["positions"][0]
        )
        np.testing.assert_allclose(
            result.get_velocities(), self.multi_frame["generic"]["velocities"][0]
        )


if __name__ == "__main__":
    unittest.main()
