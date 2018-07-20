import unittest

from functionCaltech import get_image_label

class CaltechTestCase(unittest.TestCase):

    def test_get_image_label(self):
        result = get_image_label("~/workspace/ML_toolkit/src/model/computer_vision/data/256_ObjectCategories")
        print(result)

if __name__ == '__main__':
    unittest.main()
