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

from segment_anything_ray import SegmentAnything


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

        logger = logging.getLogger(__name__)

        file_path = resource["local_paths"][0]
        file_id = resource["id"]

        # Load parameters

        #Load file
        connector.message_process(resource, "Loading contents of file...")
        with Image.open(file_path) as img:
            img = np.array(img)



if __name__ == "__main__":
    ray.shutdown()
    print("Starting Ray")
    ray.init(_temp_dir="/taiga/mohanar2/segment-anything/ray")
    assert ray.is_initialized()
    print("Ray initialized")
    extractor = SegmentAnythingFileExtractor()
    extractor.start()
