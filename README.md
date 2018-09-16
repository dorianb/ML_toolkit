# Machine learning toolkit

This package is a personal project aimed for research purpose. It covers machine learning models implementation and 
processing functions. These models are implemented and deployed mainly using Google technologies as Tensorflow and
Google Cloud Platform utilities.

Python 2.7 has been chosen to ensure components compatibility. The project is structured using Distutils as a build 
automation tool. Test are implemented for each module using unittest.

## Installation procedure

Installation of third party librairies
```
$ pip install -r requirements.txt
```

Installation of modules
```
$ python setup.py install
```

## Packages

### cloud_tools

The package cloud_tools allows to deploy complex pipeline into the cloud.

#### /gcp

Utilities for Google Cloud Platform. Processing is realised using apache beam and 
model training and prediction are realised using Google ML Engine.

A couple of process need to be acomplish before deploying pipelines into GCP.
Firstly, let us define the environment variables:
```
$ cd src/cloud_tools/gcp && . env_variable.sh
```

In order to deploy a processing pipeline, execute the following command:
```
$ python image_preprocess.py --input_dict "$DICT_FILE" --input_path "gs://cloud-ml-data/img/flower_photos/eval_set.csv" --output_path "${GCS_PATH}/preproc/eval" --cloud
```
 
### model

#### /computer_vision

Computer vision models.

#### /ner

Named Entity Recognition models.

#### /rnn

Recurrent Neural Networks.

### processing

Processing components

#### /dataset_utils

The dataset utils module aims at normalizing access to the dataset.
From dataset loading in memory to train-test splitting, the module exposes all 
the needed utilities for serveral kind of machine learning models.

#### /variable_selection

Variable selection or the process of reducing the number of variable used by a model is
often a good manner to improve a model stability and performance. In this module, several
methods are implemented in order to be used by all kind of models.

## Data

### Caltech-256

The Caltech-256 dataset contains 30 thousand images for 256 object categories. Images are in jpeg file format. Each 
object category counts at least 80 images.

For further information, follow this link: www.vision.caltech.edu/Image_Datasets/Caltech256/

### Natural image

This dataset contains 6,899 images from 8 distinct classes 
compiled from various sources. The classes include airplane,
car, cat, dog, flower, fruit, motorbike and person. 

For further information, follow this link: https://www.kaggle.com/prasunroy/natural-images

## MetaData

## Documentation

## Executable

