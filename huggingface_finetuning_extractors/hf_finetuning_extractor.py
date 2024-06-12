#!/usr/bin/env python
"""Huggingface Finetuning Extractor - Fine-tune a pre-trained model from huggingface on data"""

import logging
import json
import subprocess
from typing import Dict

from pyclowder.utils import CheckMessage
from pyclowder.extractors import Extractor
import pyclowder.files

from hf_finetuning import HuggingFaceSequenceClassificationFineTuner


class HuggingFaceFinetuningExtractor(Extractor):
    """Huggingface Finetuning Extractor - Fine-tune a pre-trained model from huggingface on data"""

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

        # HuggingFace Parameters
        HF_DATASET_NAME = ""
        NUM_TRAIN_EXAMPLES = 0
        NUM_TEST_EXAMPLES = 0

        MODEL_FILE_NAME = ""

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

        if 'HF_DATASET_NAME' in params:
            HF_DATASET_NAME = params['HF_DATASET_NAME']
            print(f"HF_DATASET_NAME: {HF_DATASET_NAME}")
        if 'NUM_TRAIN_EXAMPLES' in params:
            NUM_TRAIN_EXAMPLES = params['NUM_TRAIN_EXAMPLES']
            print(f"NUM_TRAIN_EXAMPLES: {NUM_TRAIN_EXAMPLES}")
        if 'NUM_TEST_EXAMPLES' in params:
            NUM_TEST_EXAMPLES = params['NUM_TEST_EXAMPLES']
            print(f"NUM_TEST_EXAMPLES: {NUM_TEST_EXAMPLES}")

        if 'MODEL_FILE_NAME' in params:
            MODEL_FILE_NAME = params['MODEL_FILE_NAME']
            print(f"MODEL_FILE_NAME: {MODEL_FILE_NAME}")

        finetuner = HuggingFaceSequenceClassificationFineTuner(model_name=MODEL_NAME,
                                                               num_labels=NUM_LABELS,
                                                               model_file_name=MODEL_FILE_NAME,
                                                               hf_dataset=HF_DATASET_NAME,
                                                               num_train_examples=NUM_TRAIN_EXAMPLES,
                                                               num_test_examples=NUM_TEST_EXAMPLES,
                                                               use_gpu=False,
                                                               num_workers=1)
        result = finetuner.run()

        # Log fine-tuning results
        logger.info("Fine-tuning results: ")
        logger.info(result.metrics)

        # Save the model to Clowder
        model_path = MODEL_FILE_NAME + ".pt"
        # TODO: Change the name of the model file
        pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, model_path)


if __name__ == "__main__":
    extractor = HuggingFaceFinetuningExtractor()
    extractor.start()
