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

The package cloud_tools allows to deploy complex machine learning pipelines into the cloud.

#### /gcp

Utilities for Google Cloud Platform machine learning. Processing is realised using Google DataFlow (apache beam) and 
model training and prediction are realised using Google ML Engine.

##### /example 

Transfer learning with flower dataset: https://cloud.google.com/ml-engine/docs/tensorflow/flowers-tutorial

The example shows how to use dataflow and ml engine to preprocess image data and then apply a classifier model. The 
purpose is to classify image data using transfer learning.

Firstly, let us define the environment variables:
```
$ cd src/cloud_tools/gcp 
$ . example/env_variable.sh
```

In order to deploy the processing pipeline on evaluation data, execute the following command:
```
$ python example/image_preprocess.py --input_dict "$DICT_FILE" --input_path "gs://cloud-ml-data/img/flower_photos/eval_set.csv" --output_path "${GCS_PATH}/preproc/eval" --cloud
```

For the training data, use the following command:
```
$ python example/image_preprocess.py --input_dict "$DICT_FILE" --input_path "gs://cloud-ml-data/img/flower_photos/train_set.csv" --output_path "${GCS_PATH}/preproc/train" --cloud
```

Now that the embeddings of training and evaluation data set are stored on Storage, we can train our model:
(use admin privilege if necessary)
```
$ gcloud ml-engine jobs submit training "$JOB_NAME" \
    --stream-logs \
    --module-name example.image_classification_task \
    --package-path example \
    --staging-bucket "$BUCKET_NAME" \
    --region "$REGION" \
    --runtime-version=1.4\
    -- \
    --output_path "${GCS_PATH}/training" \
    --eval_data_paths "${GCS_PATH}/preproc/eval*" \
    --train_data_paths "${GCS_PATH}/preproc/train*"
```

You can follow up the training steps with tensorboard:
```
$ tensorboard --logdir=$GCS_PATH/training
```

Export the model:
```
$ gcloud ml-engine models create "$MODEL_NAME" \
  --regions "$REGION"
```

Deploy the model for prediction:
```
$ gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_PATH}/training/model" \
  --runtime-version=1.4
```

Make a prediction from an image:
```
$ cd data/flowers
$ python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' daisy.jpg &> request.json
$ gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json
```
##### /image_classification

Train an image classification vgg model with natural images dataset preprocessed:
```
$ gcloud ml-engine jobs submit training "$JOB_NAME" \
 --stream-logs --module-name image_classification.image_classification_task \ 
 --package-path image_classification \
 --staging-bucket "$BUCKET_NAME" --region "$REGION" \
 --runtime-version=1.4 \   
 -- \
 --output_path "${BUCKET_NAME}/model/vgg_16/natural_images/training" \
 --eval_data_paths "${GCS_PATH}/validation*" \
 --train_data_paths "${GCS_PATH}/training*"
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

