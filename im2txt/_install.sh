# Location to save the MSCOCO data.
MSCOCO_DIR="/Volumes/raid/_deeplearning/coco2014/"

# Build the preprocessing script.
cd /Users/eisneim/www/deepLearning/_tf_models/im2txt
bazel build //im2txt:download_and_preprocess_mscoco

# Run the preprocessing script.
bazel-bin/im2txt/download_and_preprocess_mscoco "${MSCOCO_DIR}"