import unittest

import numpy as np

from lammpsparser.units import UnitConverter


class TestUnits(unittest.TestCase):
    def test_convert_array_to_pyiron_units(self):
        uc = UnitConverter(units="metal")
        result = uc.convert_array_to_pyiron_units(array=np.array([1.0]), label="energy")
        self.assertTrue(np.all(np.equal(result, np.array([1.0]))))
        result = uc.convert_array_to_pyiron_units(
            array=np.array([1.0]), label="energy_tot"
        )
        self.assertTrue(np.all(np.equal(result, np.array([1.0]))))
        result = uc.convert_array_to_pyiron_units(array=np.array([1.0]), label="time")
        self.assertTrue(np.all(np.equal(result, np.array([1000.0]))))
