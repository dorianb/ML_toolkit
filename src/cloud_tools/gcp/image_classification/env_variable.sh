declare BUCKET_NAME="gs://mltoolkit"
declare REGION=$(gcloud config list compute/region --format "value(compute.region)")
declare PROJECT_ID=$(gcloud config list project --format "value(core.project)")
declare JOB_NAME="image_classification_$(date +%Y%m%d_%H%M%S)"
declare GCS_PATH="${BUCKET_NAME}/preprocessed/natural_images/"
declare DICT_FILE="${BUCKET_NAME}/dataset/natural_images/classes.csv"

declare MODEL_NAME=natural
declare VERSION_NAME=v1

declare GOOGLE_APPLICATION_CREDENTIALS=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")$PROJECT_ID".json"

declare OUTPUT_PATH="${GCS_PATH}/training"

echo
echo "Using job id: " $JOB_NAME
set -v -e
