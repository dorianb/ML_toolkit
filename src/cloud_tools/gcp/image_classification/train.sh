cd ~/workspace/ML_toolkit/src/cloud_tools/gcp/
. image_classification/env_variable.sh
sudo gcloud ml-engine jobs submit training "$JOB_NAME"  --stream-logs \
    --module-name image_classification.image_classification_task \
    --package-path image_classification \
    --staging-bucket "${BUCKET_NAME}-mlengine" \
    --region "$REGION" --runtime-version=1.4 \
    -- \
    --output_path "${BUCKET_NAME}/model/vgg_16/natural_images/training" \
    --eval_data_paths "${GCS_PATH}validation*" \
    --train_data_paths "${GCS_PATH}training*"