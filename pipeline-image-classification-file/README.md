# Image Classification Extractor using HuggingFace Pipelines

Use this extractor to classify images using HuggingFace's [pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines) 
and`transformers` libraries.

Test file: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg

## Build
```
docker build -t clowder/extractors-huggingface_image_classification .
```

## Run
```
docker run -t -i --rm --net clowder_clowder -e "RABBITMQ_URI=amqp://guest:guest@rabbitmq:5672/%2f" --name "extractors-huggingface_image_classification" clowder/extractors-huggingface_image_classification
```

Docker flags:
- `--net` links the extractor to the Clowder Docker network (run `docker network ls` to identify your own.)
- `-e RABBITMQ_URI=` sets the environment variables can be used to control what RabbitMQ server and exchange it will bind itself to. Setting the `RABBITMQ_EXCHANGE` may also help.
  - You can also use `--link` to link the extractor to a RabbitMQ container.
- `--name` assigns the container a name visible in Docker Desktop.

## Troubleshooting
**If you run into _any_ trouble**, please reach us out [Slack](https://clowder-software.slack.com/archives/CEAMPH39C).

