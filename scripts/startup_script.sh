gcloud storage ls gs://music_train


gsutil -m \
  -o "GSUtil:parallel_process_count=128" \
  -o "GSUtil:parallel_thread_count=16" \
  cp -r "gs://music_train/data/fma_large" "/root/workspace/data"

gcsfuse music_train /root/

git clone https://github.com/westc4/Training

gcloud auth login

top -o %CPU