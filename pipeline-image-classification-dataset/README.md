# Image Classification on single file using HuggingFace transformers

Use this extractor to classify images using HuggingFace's [pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines) and`transformers` libraries. 
It uses [ray.io](https://www.ray.io/) to launch multiple inference steps concurrently, one for each image in the dataset.



Test file: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg

## Build
```
docker build -t clowder/extractors-huggingface_image_classification_dataset .
```

## Run
First install and run ray.io on the same host for testing. 
```
pip install "ray[default]"
ray start --head --port=6379
ray status
```
You can view the Ray dashboard at http://127.0.0.1:8265.

Then run the extractor.

```
docker run -t -i --rm --net clowder_clowder -e "RABBITMQ_URI=amqp://guest:guest@rabbitmq:5672/%2f" --name "extractors-huggingface_image_classification_dataset" clowder/extractors-huggingface_image_classification_dataset
```

Upload several image files to a Clowder dataset and submit the datasets for extraction. The extractor will classify each 
image and store the results in the dataset as metadata on the individual files.

Docker flags:
- `--net` links the extractor to the Clowder Docker network (run `docker network ls` to identify your own.)
- `-e RABBITMQ_URI=` sets the environment variables can be used to control what RabbitMQ server and exchange it will bind itself to. Setting the `RABBITMQ_EXCHANGE` may also help.
  - You can also use `--link` to link the extractor to a RabbitMQ container.
- `--name` assigns the container a name visible in Docker Desktop.

## Troubleshooting
**If you run into _any_ trouble**, please reach us out [Slack](https://clowder-software.slack.com/archives/CEAMPH39C).

