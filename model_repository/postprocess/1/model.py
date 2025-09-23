import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        
    def execute(self, requests):
        responses = []

        for request in requests:
            pass
        
        return responses