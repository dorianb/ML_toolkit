import unittest

from sequence_model.classCell import Cell


class CellTestCase(unittest.TestCase):

    def test_get_parameter(self):
        """
        Test the get_parameter method

        """

        cell_1 = Cell()

        parameter_1 = cell_1.get_parameter(shape=(2, 3), initializer="zeros", name="parameter_1", seed=42)

        self.assertTupleEqual(tuple(parameter_1.get_shape().as_list()), (2, 3))


if __name__ == '__main__':
    unittest.main()
