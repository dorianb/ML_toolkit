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

## Data

## MetaData

## Documentation

## Executable

