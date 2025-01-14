#!/usr/bin/env python

import logging
import time
from typing import Dict

import pyclowder.files
from pyclowder.extractors import Extractor
from transformers import pipeline
from PIL import Image

# where to cache downloaded models
# os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'

logger = logging.getLogger(__name__)


def run_model(image: Image):
    segmenter = pipeline(task="image-segmentation", model="facebook/detr-resnet-50-panoptic", revision="fc15262")
    preds = segmenter(image)
    preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
    return preds


class ImageClassification(Extractor):
    def __init__(self):
        Extractor.__init__(self)
        self.setup()
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    def process_message(self, connector, host, secret_key, resource, parameters):
        file_path = resource["local_paths"][0]
        file_id = resource['id']
        print(f"Processing file {file_id} with path {file_path}")

        print(f"Processing file {file_id} with path {file_path}")

        # Load file
        connector.message_process(resource, "Loading contents of file")
        with Image.open(file_path) as image:
            connector.message_process(resource, "File loaded")

            # Model prediction
            start_time = time.monotonic()
            predictions: Dict = run_model(image)
            connector.message_process(resource, f"Predictions: {predictions}")

            # Format results with "contexts" in extractor_info.json
            preds = {
                "predictions": predictions,
                "runtime (seconds)": round(time.monotonic() - start_time, 2)
            }

            # Upload metadata
            metadata = self.get_metadata(preds, 'file', file_id, host)
            connector.message_process(resource, f"Uploading metadata: {metadata}")
            pyclowder.files.upload_metadata(connector, host, secret_key, file_id, metadata)


if __name__ == "__main__":
    extractor = ImageClassification()
    extractor.start()
