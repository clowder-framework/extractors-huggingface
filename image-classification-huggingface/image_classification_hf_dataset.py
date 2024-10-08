import logging

import ray

import os
import json
from PIL import Image
import pyclowder.files
from pyclowder.extractors import Extractor
from transformers import pipeline
from typing import Dict

@ray.remote
class BatchPredictor:
    """
    BatchPredictor class for running inference on multiple files concurrently using Ray. By making this a class
    we load the model only once.
    """

    def __init__(self, task, model_name):
        self.model = None
        self.model = pipeline(task=task, model=model_name)
        print("BatchPredictor initialized. Model loaded.")



    def process_file(self, local_file_path):
        # Load file
        with Image.open(local_file_path) as image:
            # Run model
            preds = self.model(image)
            preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
            return preds


class ImgExtractor(Extractor):
    """Count the number of characters, words and lines in a text file."""

    def __init__(self):
        Extractor.__init__(self)

        # add any additional arguments to parser
        # self.parser.add_argument('--max', '-m', type=int, nargs='?', default=-1,
        #                          help='maximum number (default=-1)')

        # parse command line and load default logging configuration
        self.setup()

        # set model as None first
        self.remoteBatchPredictor = None

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    def process_message(self, connector, host, secret_key, resource, parameters):
        """Dataset extractor. We get all filenames at once."""
        logger = logging.getLogger(__name__)

        # Get list of all files in dataset
        filelist = pyclowder.datasets.get_file_list(connector, host, secret_key, parameters['datasetId'])
        localfiles = []
        clowder_version = int(os.getenv('CLOWDER_VERSION', '1'))

        # Load user-defined params
        model_name = ""
        task = ""

        print(f"Parameters: {parameters}")

        if 'parameters' in parameters:
            params = None
            try:
                params = json.loads(parameters['parameters'])
            except TypeError as e:
                print(f"Failed to load parameters, it's not compatible with json.loads().\nError:{e}")
                if type(parameters == Dict):
                    params = parameters['parameters']

        if "MODEL_NAME" in params:
            model_name = params["MODEL_NAME"]

        if "MODEL_TASK" in params:
            if params["MODEL_TASK"] == "Image Classification":
                task = "image-classification"
            elif params["MODEL_TASK"] == "Object Detection":
                task = "object-detection"

        logging.info(f"Model Name: {model_name} Task: {task}")

        # Initialize ray actor and model
        self.remoteBatchPredictor = BatchPredictor.options(max_concurrency=3).remote(task, model_name)

        # # Loop through dataset and download all file "locally"
        for file_dict in filelist:
            # Use the correct key depending on the Clowder version
            if clowder_version == 2:
                extension = "." + file_dict['content_type']['content_type'].split("/")[1]
            else:
                extension = "." + file_dict['contentType'].split("/")[1]
            localfiles.append(pyclowder.files.download(connector, host, secret_key, file_dict['id'], ext=extension))

        # These process messages will appear in the Clowder UI under Extractions.
        connector.message_process(resource, "Loading contents of file...")

        # Print resource
        logging.warning("Printing Resources:")
        logging.warning(resource)
        logging.warning("\n")

        # Print localfiles
        logging.warning("Printing local files:")
        logging.warning(localfiles)
        logging.warning("\n")

        # Initialize actor and run machine learning module concurrently
        classifications = ray.get(
            [self.remoteBatchPredictor.process_file.remote(localfiles[i]) for i in range(len(localfiles))])

        for i in range(len(classifications)):
            # Upload metadata to each original file
            my_metadata = {
                'Output': classifications[i]
            }

            # Create Clowder metadata object
            metadata = self.get_metadata(my_metadata, 'file', filelist[i]['id'], host)

            # Normal logs will appear in the extractor log, but NOT in the Clowder UI.
            logger.debug(metadata)

            # Upload metadata to original file
            pyclowder.files.upload_metadata(connector, host, secret_key, filelist[i]['id'], metadata)

        # Finish
        logging.warning("Successfully extracted!")

if __name__ == "__main__":
    # Run on local ray cluster and install required dependencies
    ray.init()
    extractor = ImgExtractor()
    extractor.start()
