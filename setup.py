from distutils.core import setup

setup(
    name='ML_toolkit',
    version='1.0',
    packages=['sequence_model', 'computer_vision',
              'gcp_example', 'gcp_image_classification',
              'dataset_utils', 'variable_selection'],
    package_dir={
        'sequence_model': 'src/model/sequence_model',
        'computer_vision': 'src/model/computer_vision',
        'gcp_example': 'src/cloud_tools/gcp/example',
        'gcp_image_classification': 'src/cloud_tools/gcp/image_classification',
        'dataset_utils': 'src/processing/dataset_utils',
        'variable_selection': 'src/processing/variable_selection'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Machine learning models and processing'
)
