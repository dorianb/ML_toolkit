declare BUCKET_NAME="gs://"$(gcloud config list project --format "value(core.project)")
declare REGION=$(gcloud config list compute/region --format "value(compute.region)")
declare PROJECT_ID=$(gcloud config list project --format "value(core.project)")
declare JOB_NAME="flowers_${USER}_$(date +%Y%m%d_%H%M%S)"
declare GCS_PATH="${BUCKET_NAME}/${USER}/${JOB_NAME}"
declare DICT_FILE=gs://cloud-ml-data/img/flower_photos/dict.txt

declare MODEL_NAME=flowers
declare VERSION_NAME=v1

declare GOOGLE_APPLICATION_CREDENTIALS=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")$PROJECT_ID".json"


echo
echo "Using job id: " $JOB_NAME
set -v -e
