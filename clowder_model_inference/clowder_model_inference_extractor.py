"""Clowder Inference Extractor - Fine-tune a pre-trained model from clowder data"""

import logging
import json
from typing import Dict

from pyclowder.utils import CheckMessage
from pyclowder.extractors import Extractor
import pyclowder.files

from clowder_inference import TorchSQModelPredictor

class ClowderSQInferenceExtractor(Extractor):
    """Clowder Finetuning Extractor - Fine-tune a pre-trained model from Clowder on data"""

    def __init__(self):
        Extractor.__init__(self)
        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the extractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    def process_message(self, connector, host, secret_key, resource, parameters):
        """Dataset extractor. We get all filenames at once."""

        logger = logging.getLogger(__name__)

        dataset_id = resource['parent']['id']

        # Load user-defined params from the GUI.
        BASE_MODEL_NAME = ""
        MODEL_FILE_ID = ""
        DATASET_FILE_ID = ""
        PREDICTIONS_FILE_NAME = ""

        print(f"Parameters: {parameters}")

        if 'parameters' in parameters:
            params = None
            try:
                params = json.loads(parameters['parameters'])
            except TypeError as e:
                print(f"Failed to load parameters, it's not compatible with json.loads().\nError:{e}")
                if type(parameters == Dict):
                    params = parameters['parameters']

        if 'BASE_MODEL_NAME' in params:
            BASE_MODEL_NAME = params['BASE_MODEL_NAME']
            print(f"BASE_MODEL_NAME: {BASE_MODEL_NAME}")
        if 'MODEL_FILE_ID' in params:
            MODEL_FILE_ID = params['MODEL_FILE_ID']
            print(f"MODEL_FILE_ID: {MODEL_FILE_ID}")
        if 'DATASET_FILE_ID' in params:
            DATASET_FILE_ID = params['DATASET_FILE_ID']
            print(f"DATASET_FILE_ID: {DATASET_FILE_ID}")
        if 'PREDICTIONS_FILE_NAME' in params:
            PREDICTIONS_FILE_NAME = params['PREDICTIONS_FILE_NAME']
            print(f"PREDICTIONS_FILE_NAME: {PREDICTIONS_FILE_NAME}")


        # Get the model file
        model_file = pyclowder.files.download(connector, host, secret_key, MODEL_FILE_ID)

        # Get the dataset file
        dataset_file = pyclowder.files.download(connector, host, secret_key, DATASET_FILE_ID)

        # Run the inference
        predictor = TorchSQModelPredictor(BASE_MODEL_NAME, model_file, dataset_file, PREDICTIONS_FILE_NAME)
        predictions = predictor.predict()

        # Upload the predictions file
        pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, PREDICTIONS_FILE_NAME + ".csv")

        #Log the results
        logger.info(f"Predictions file uploaded")


if __name__ == "__main__":
    extractor = ClowderSQInferenceExtractor()
    extractor.start()






