import logging
import json
from typing import Dict
import os
import numpy as np

from pyclowder.utils import CheckMessage
from pyclowder.extractors import Extractor
import pyclowder.files
from PIL import Image

from segment_anything_file import SegmentAnything


# Helper class to encode the masks as JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SegmentAnythingFileExtractor(Extractor):
    """ Cloweder Segment-Anything extractor - Uses Meta AI Segment-Anything model to create masks """

    def __init__(self):
        # parse command line and load default logging configuration
        Extractor.__init__(self)
        self.setup()

        # setup logging for the extractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    def process_message(self, connector, host, secret_key, resource, parameters):

        logger = logging.getLogger(__name__)

        file_path = resource["local_paths"][0]
        dataset_id = resource["parent"]["id"]
        logger.info("Resource: " + str(resource))

        # Load parameters
        SAVE_IMAGE = True
        BBOX = None

        if 'parameters' in parameters:
            params = None
            logging.info("Received parameters "+ str(parameters['parameters']))
            params = parameters['parameters']

            if "SAVE_IMAGE" in params:
                SAVE_IMAGE = eval(params["SAVE_IMAGE"])
            if "bbox" in params:
                BBOX = params["bbox"]
                # convert string to list
                BBOX = json.loads(BBOX)['boundingBox']

        logging.info("Parameters: " + str(parameters))

        # segment_anything = SegmentAnything()
        #
        # file_name = resource['name'].split(".")[0]
        # logger.info("File name: " + file_name)
        #
        # if BBOX is None:
        #     segmented_json_mask = segment_anything.generate_mask(file_path)
        # else:
        #     segmented_json_mask = segment_anything.generate_prompt_mask(file_path, BBOX)
        #
        # # Encode the masks as JSON and upload to dataset
        # json_file_name = file_name + "_mask.json"
        # with open(json_file_name, 'w') as f:
        #     json.dump(segmented_json_mask, f, cls=NumpyEncoder)
        #
        # #Upload file
        # pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, json_file_name)
        # os.remove(json_file_name)

        logging.info("User chose to save image- " + str(SAVE_IMAGE))
        # if SAVE_IMAGE:
        #     img_file_name = file_name + "_masked.png"
        #     if BBOX is not None:
        #         segment_anything.save_prompt_output(segmented_json_mask, file_path, img_file_name)
        #     else:
        #         segment_anything.save_output(segmented_json_mask, file_path, img_file_name)
        #     logging.info("Uploading masked image")
        #     pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, img_file_name)
        #     os.remove(img_file_name)
        #
        # logging.warning("Successfully extracted!")



if __name__ == "__main__":
    extractor = SegmentAnythingFileExtractor()
    extractor.start()
