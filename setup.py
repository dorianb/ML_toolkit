from distutils.core import setup

setup(
    name='ML_toolkit',
    version='1.0',
    packages=['rnn'],
    package_dir={
        'rnn': 'src/model/rnn',
        'computer_vision': 'src/model/computer_vision',
        'pipeline': 'src/processing/pipeline'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Machine learning models and processing'
)
