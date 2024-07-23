import logging
import json
from typing import Dict
import os
import numpy as np

from pyclowder.utils import CheckMessage
from pyclowder.extractors import Extractor
import pyclowder.files
from torch.cuda import is_available
import ray
from PIL import Image

from segment_anything_file_ray import SegmentAnything


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
        """Get all the files from the resource and process it."""

        try:
            logger = logging.getLogger(__name__)

            file_path = resource["local_paths"][0]
            dataset_id = resource["parent"]["id"]
            logger.info("Resource: " + str(resource))

            # Load parameters
            SAVE_IMAGE = True
            BBOX = None

            if 'parameters' in parameters:
                params = None
                logging.info("Received parameters")
                try:
                    params = json.loads(parameters['parameters'])
                except TypeError as e:
                    print(f"Failed to load parameters, it's not compatible with json.loads().\nError:{e}")
                    if type(parameters == Dict):
                        params = parameters['parameters']

                if "SAVE_IMAGE" in parameters:
                    SAVE_IMAGE = parameters["SAVE_IMAGE"]
                if "bbox" in parameters:
                    BBOX = parameters["bbox"]
                    # convert string to list
                    BBOX = json.loads(BBOX)['boundingBox']

            logging.info("Parameters: " + str(parameters))


            # Check if gpu is available
            if is_available():
                logging.warning("GPU is available")
                actor = SegmentAnything.options(num_gpus=1).remote()
            else:
                logging.warning("GPU is not available")
                actor = SegmentAnything.remote()

            file_name = resource['name'].split(".")[0]
            logger.info("File name: " + file_name)

            if BBOX is None:
                segmented_json_mask = ray.get(actor.generate_mask.remote(file_path))
            else:
                segmented_json_mask = ray.get(actor.generate_prompt_mask.remote("test.jpeg", BBOX))

            # Encode the masks as JSON and upload to dataset
            json_file_name = file_name + "_mask.json"
            with open(json_file_name, 'w') as f:
                json.dump(segmented_json_mask, f, cls=NumpyEncoder)

            #Upload file
            pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, json_file_name)
            os.remove(json_file_name)


            if SAVE_IMAGE:
                img_file_name = file_name + "_masked.png"
                if BBOX is not None:
                    ref = actor.save_prompt_output.remote(segmented_json_mask, file_path, img_file_name)
                else:
                    ref = actor.save_output.remote(segmented_json_mask, file_path, img_file_name)
                ray.get(ref)
                logging.info("Uploading masked image")
                pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, img_file_name)
                os.remove(img_file_name)

            logging.warning("Successfully extracted!")
            return
        except Exception as e:
            logging.error(e)
            logging.warning("Error in running the extractor")
            return


if __name__ == "__main__":
    ray.shutdown()
    print("Starting Ray")
    ray.init(_temp_dir="/taiga/mohanar2/segment-anything/ray")
    assert ray.is_initialized()
    print("Ray initialized")
    extractor = SegmentAnythingFileExtractor()
    extractor.start()
