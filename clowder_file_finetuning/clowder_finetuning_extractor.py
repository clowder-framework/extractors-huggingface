#!/usr/bin/env python
"""Clowder Finetuning Extractor - Fine-tune a pre-trained model from clowder data"""

import logging
import json
import os
from typing import Dict

from pyclowder.utils import CheckMessage
from pyclowder.extractors import Extractor
import pyclowder.files
from torch.cuda import is_available

from clowder_finetuning import ClowderSQFineTuner


class ClowderSQFinetuningExtractor(Extractor):
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

        MODEL_NAME = ""
        NUM_LABELS = 0
        MODEL_FILE_NAME = ""

        # Local Files Parameters
        LOCAL_TRAIN_FILE_ID = ""
        LOCAL_TEST_FILE_ID = ""
        DATA_TYPE = ""

        WANDB_PROJECT_NAME = None

        print(f"Parameters: {parameters}")

        if 'parameters' in parameters:
            params = None
            try:
                params = json.loads(parameters['parameters'])
            except TypeError as e:
                print(f"Failed to load parameters, it's not compatible with json.loads().\nError:{e}")
                if type(parameters == Dict):
                    params = parameters['parameters']

        if 'MODEL_NAME' in params:
            MODEL_NAME = params['MODEL_NAME']
            print(f"MODEL_NAME: {MODEL_NAME}")
        if 'NUM_LABELS' in params:
            NUM_LABELS = params['NUM_LABELS']
            print(f"NUM_LABELS: {NUM_LABELS}")

        if 'LOCAL_TRAIN_FILE_ID' in params:
            LOCAL_TRAIN_FILE_ID = params['LOCAL_TRAIN_FILE_ID']
            print(f"LOCAL_TRAIN_FILE_ID: {LOCAL_TRAIN_FILE_ID}")
        if 'LOCAL_TEST_FILE_ID' in params:
            LOCAL_TEST_FILE_ID = params['LOCAL_TEST_FILE_ID']
            print(f"LOCAL_TEST_FILE_ID: {LOCAL_TEST_FILE_ID}")
        if 'DATA_TYPE' in params:
            DATA_TYPE = params['DATA_TYPE']
            print(f"DATA_TYPE: {DATA_TYPE}")

        if 'MODEL_FILE_NAME' in params:
            MODEL_FILE_NAME = params['MODEL_FILE_NAME']
            print(f"MODEL_FILE_NAME: {MODEL_FILE_NAME}")

        # wandb parameters
        if 'WANDB_API_KEY' in params:
            WANDB_API_KEY = params['WANDB_API_KEY']
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        if 'WANDB_PROJECT_NAME' in params:
            WANDB_PROJECT_NAME = params['WANDB_PROJECT_NAME']
            print(f"WANDB_PROJECT_NAME: {WANDB_PROJECT_NAME}")

        # Load local files from Clowder
        local_train_file = pyclowder.files.download(connector, host, secret_key,
                                                    fileid=LOCAL_TRAIN_FILE_ID)
        local_test_file = pyclowder.files.download(connector, host, secret_key,
                                                   fileid=LOCAL_TEST_FILE_ID)

        # Check if GPU is available
        if is_available():
            logger.info("GPU is available")
            finetuner = ClowderSQFineTuner(model_name=MODEL_NAME,
                                           num_labels=NUM_LABELS,
                                           model_file_name=MODEL_FILE_NAME,
                                           data_type=DATA_TYPE,
                                           local_train_file=local_train_file,
                                           local_test_file=local_test_file,
                                           # Change the path to the desired path. TODO: Use environment variable
                                           ray_storage_path=os.getenv("", "clowder_finetuning"),
                                           use_gpu=True,
                                           wandb_project=WANDB_PROJECT_NAME,
                                           num_workers=1)
        else:
            finetuner = ClowderSQFineTuner(model_name=MODEL_NAME,
                                           num_labels=NUM_LABELS,
                                           model_file_name=MODEL_FILE_NAME,
                                           data_type=DATA_TYPE,
                                           local_train_file=local_train_file,
                                           local_test_file=local_test_file,
                                           use_gpu=False,
                                           wandb_project=WANDB_PROJECT_NAME,
                                           num_workers=1)
        result, metrics = finetuner.run()

        # Log fine-tuning results
        logger.info("Fine-tuning results: ")
        logger.info(metrics)

        # Save the model to Clowder
        model_path = MODEL_FILE_NAME + ".pt"
        model_file_id = pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, model_path)
        # Add metrics to the dataset
        metrics = {"Model metrics": metrics}

        # post metadata to Clowder
        metadata = self.get_metadata(result, 'file', model_file_id, host)

        # Upload metadata to original file
        pyclowder.files.upload_metadata(connector, host, secret_key, model_file_id, metadata)


if __name__ == "__main__":
    extractor = ClowderSQFinetuningExtractor()
    extractor.start()
