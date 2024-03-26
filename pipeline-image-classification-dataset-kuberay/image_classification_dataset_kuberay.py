#!/usr/bin/env python


import logging
import time

from pyclowder.extractors import Extractor
from ray.job_submission import JobSubmissionClient, JobStatus


class ImgExtractor(Extractor):
    """Count the number of characters, words and lines in a text file."""

    def __init__(self, job_submission_client):
        Extractor.__init__(self)

        # add any additional arguments to parser
        # self.parser.add_argument('--max', '-m', type=int, nargs='?', default=-1,
        #                          help='maximum number (default=-1)')

        # parse command line and load default logging configuration
        self.setup()

        self.job_submission_client = job_submission_client

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    def process_message(self, connector, host, secret_key, resource, parameters):
        """Dataset extractor. We get all filenames at once."""

        job_id = self.job_submission_client.submit_job(
            # Entrypoint shell command to execute
            entrypoint=f"python remote_script.py {host} {parameters['datasetId']} {secret_key}",
            # Path to the local directory that contains the script.py file
            runtime_env={"working_dir": "./", "pip": ["transformers", "torch", "torchvision", "timm"],
                         "env_vars": {"CLOWDER_VERSION": "2"}}
        )
        print(job_id)

        def wait_until_status(job_id, status_to_wait_for, timeout_seconds=10000):
            start = time.time()
            while time.time() - start <= timeout_seconds:
                status = client.get_job_status(job_id)
                print(f"status: {status}")
                if status in status_to_wait_for:
                    break
                time.sleep(1)

        wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
        logs = client.get_job_logs(job_id)
        print(logs)

        # Finish
        logging.warning("Successfully extracted!")


if __name__ == "__main__":
    # If using a remote cluster, replace 127.0.0.1 with the head node's IP address.
    client = JobSubmissionClient("http://127.0.0.1:8265")
    # Run on local ray cluster and install required dependencies
    extractor = ImgExtractor(client)
    extractor.start()
