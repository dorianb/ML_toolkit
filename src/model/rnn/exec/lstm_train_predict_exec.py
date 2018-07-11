import argparse
import logging

from rnn.classLSTM import LSTM

"""
src/model/rnn/exec/lstm_train_predict_exec.py --train 1 --debug 1 
"""

parser = argparse.ArgumentParser(description='LSTM Train and predict')
parser.add_argument('--train', type=int, help='Training mode', default=1)
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

logger = logging.Logger("lstm_exec", level=logging.DEBUG if args.debug else logging.INFO)

try:

    lstm_0 = LSTM(batch_size=1, time_step=3, n_features=3)

except Exception:

    logger.error('Program exited with error')
