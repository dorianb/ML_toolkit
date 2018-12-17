import unittest
import tensorflow as tf

from sequence_model.classCell import Cell


class CellTestCase(unittest.TestCase):

    def test_initialize_variable(self):
        """
        Test the initialization variable method.
        """
        shape = (10, 10)
        var_1 = Cell.initialize_variable(shape=shape, initializer="zeros", dtype=tf.float32,
                                         name="v1")
        var_2 = Cell.initialize_variable(shape=shape, initializer="orthogonal", dtype=tf.float32,
                                         name="v2")
        var_3 = Cell.initialize_variable(shape=shape, initializer="random_uniform", dtype=tf.float32,
                                         name="v3")
        var_4 = Cell.initialize_variable(shape=shape, initializer="random_normal", dtype=tf.float32,
                                         name="v4")

        self.assertTupleEqual(tuple(var_1.get_shape().as_list()), shape)
        self.assertTupleEqual(tuple(var_2.get_shape().as_list()), shape)
        self.assertTupleEqual(tuple(var_3.get_shape().as_list()), shape)
        self.assertTupleEqual(tuple(var_4.get_shape().as_list()), shape)

if __name__ == '__main__':
    unittest.main()
