import json
import logging
import posixpath
import sys
import tempfile

import ray

import os

import requests
from PIL import Image
from transformers import pipeline


# @ray.remote(num_cpus=1, memory=5*9)
@ray.remote
class BatchPredictor:
    """
    BatchPredictor class for running inference on multiple files concurrently using Ray. By making this a class
    we load the model only once.
    """

    def __init__(self):
        self.model = pipeline(task="image-segmentation", model="facebook/detr-resnet-50-panoptic", revision="fc15262")
        print("BatchPredictor initialized. Model loaded.")

    def process_file(self, host, key, file_id, extension):
        print(f"Processing file {file_id} with extension {extension}")
        # local_file_path = pyclowder.files.download(None, host, key, file_id, ext=extension)

        url = posixpath.join(host, f'api/v2/files/{file_id}')
        headers = {"X-API-KEY": key}
        result = requests.get(url, stream=True, headers=headers)

        (inputfile, inputfilename) = tempfile.mkstemp(suffix=extension)

        print(f"Temp file {inputfile} / {inputfilename} ")

        try:
            with os.fdopen(inputfile, "wb") as outputfile:
                for chunk in result.iter_content(chunk_size=10 * 1024):
                    outputfile.write(chunk)
        except Exception:
            os.remove(inputfilename)
            raise

        # Load file
        with Image.open(inputfilename) as image:
            print(f"Image {inputfilename} loaded")
            # Run model
            preds = self.model(image)
            preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
            return preds


if __name__ == "__main__":
    args = sys.argv[1:]
    host = args[0]
    dataset_id = args[1]
    key = args[2]

    # ray.init()

    """Dataset extractor. We get all filenames at once."""
    logger = logging.getLogger(__name__)

    # Get list of all files in dataset
    headers = {'Content-Type': 'application/json',
               'X-API-KEY': key}
    url = posixpath.join(host, "api/v2/datasets/%s/files" % dataset_id)

    result = requests.get(url, headers=headers)
    result.raise_for_status()

    files = json.loads(result.text)["data"]
    print(files)

    localfiles = []
    clowder_version = int(os.getenv('CLOWDER_VERSION', '2'))

    # # Loop through dataset and download all file "locally"
    for file_dict in files:
        # Use the correct key depending on the Clowder version
        if clowder_version == 2:
            extension = "." + file_dict['content_type']['content_type'].split("/")[1]
        else:
            extension = "." + file_dict['contentType'].split("/")[1]
        localfiles.append((file_dict['id'], extension))

    # Initialize actor and run machine learning module concurrently
    remoteBatchPredictor = BatchPredictor.options(max_concurrency=1).remote()
    classifications = ray.get(
        [remoteBatchPredictor.process_file.remote(host, key, localfiles[i][0], localfiles[i][1]) for i in range(len(localfiles))])

    print(classifications)

    for i in range(len(classifications)):
        # Upload metadata to each original file

        metadata = {'context_url': 'https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld', 'content': {
            'Output': classifications[i]},
            'listener': {'name': 'huggingface.image.classification.dataset.kuberay', 'version': '1.0',
                                 'description': '1.0'}}

        # Normal logs will appear in the extractor log, but NOT in the Clowder UI.
        logger.debug(metadata)

        # Upload metadata to original file
        # pyclowder.files.upload_metadata(None, host, key, files[i]['id'], metadata)
        headers = {'Content-Type': 'application/json',
                   'X-API-KEY': key}
        url = posixpath.join(host, 'api/v2/files/%s/metadata' % localfiles[i][0])
        result = requests.post(url, headers=headers, data=json.dumps(metadata))
        result.raise_for_status()

    # Finish
    logging.warning("Successfully extracted!")
