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

from segment_anything_ray import SegmentAnything

class SegmentAnythingExtractor(Extractor):
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

        logger = logging.getLogger(__name__)

        dataset_id = resource['id']
        # Get list of all files in dataset
        filelist = pyclowder.datasets.get_file_list(connector, host, secret_key, dataset_id)
        localfiles = []
        filenames = []
        clowder_version = int(os.getenv('CLOWDER_VERSION', '1'))

        # # Loop through dataset and download all file "locally"
        for file_dict in filelist:
            # Use the correct key depending on the Clowder version
            if clowder_version == 2:
                extension = "." + file_dict['content_type']['content_type'].split("/")[1]
            else:
                extension = "." + file_dict['contentType'].split("/")[1]

            logging.info("Extension: " + extension)
            # Check if the file is an image
            if extension not in ['.jpg', '.jpeg', '.png']:
                continue

            logging.info("Downloading file")
            logging.info(file_dict['name'])
            localfiles.append(pyclowder.files.download(connector, host, secret_key, file_dict['id'], ext=extension))
            filenames.append(file_dict['name'])

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

        SAVE_IMAGE = False
        if 'parameters' in parameters:
            params = None
            logging.info("Received parameters")
            try:
                params = json.loads(parameters['parameters'])
            except TypeError as e:
                print(f"Failed to load parameters, it's not compatible with json.loads().\nError:{e}")
                if type(parameters == Dict):
                    params = parameters['parameters']

            if 'SAVE_IMAGE' in params:
                SAVE_IMAGE = params['SAVE_IMAGE'] == "True"
                logging.info(f"Received SAVE_IMAGE: {SAVE_IMAGE}")


        # Check if gpu is available
        if is_available():
            logging.warning("GPU is available")
            actor = SegmentAnything.options(num_gpus=1).remote()
        else:
            logging.warning("GPU is not available")
            actor = SegmentAnything.remote()

        logging.info("Running inference on files")
        segmented_json_masks = ray.get([
            actor.generate_mask.remote(localfiles[i]) for i in range(len(localfiles))
        ])

        if SAVE_IMAGE:
            for i in range(len(segmented_json_masks)):
                actor.save_output.remote(segmented_json_masks[i], localfiles[i], localfiles[i].split(".")[0] + "_masked.png")
        # Encode the masks as JSON

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        for i in range(len(segmented_json_masks)):
            file_name = filelist[i]['name'].split(".")[0]
            json_file_name = file_name + "_mask.json"
            with open(json_file_name, 'w') as f:
                json.dump(segmented_json_masks[i], f, cls=NumpyEncoder)
            pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, json_file_name)
            # Delete the file after uploading
            os.remove(json_file_name)

            if SAVE_IMAGE:
                logging.info("Uploading masked image")
                img_file_name = file_name + "_masked.png"
                pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, img_file_name)
                # Delete the file after uploading
                os.remove(img_file_name)

            logging.warning("Successfully extracted!")


if __name__ == "__main__":
    ray.shutdown()
    print("Starting Ray")
    ray.init(_temp_dir="/taiga/mohanar2/segment-anything/ray")
    assert ray.is_initialized()
    print("Ray initialized")
    extractor = SegmentAnythingExtractor()
    extractor.start()






