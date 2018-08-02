from distutils.core import setup

setup(
    name='ML_toolkit',
    version='1.0',
    packages=['rnn', 'computer_vision',
              'gcp',
              'dataset_utils'],
    package_dir={
        'rnn': 'src/model/rnn',
        'computer_vision': 'src/model/computer_vision',
        'gcp': 'src/cloud_tools/gcp',
        'dataset_utils': 'src/processing/dataset_utils'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Machine learning models and processing'
)
