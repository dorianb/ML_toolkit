import unittest
from rnn.classLSTM import LSTM


class LSTMTestCase(unittest.TestCase):

    def test_init(self):
        lstm_1 = LSTM()
        self.assertIsNotNone(lstm_1)

if __name__ == '__main__':
    unittest.main()
