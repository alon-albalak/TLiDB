# NOTES: What do models have?
# attributes: possible tasks, single model, multiple possible output layers
# methods: transform_input, transform_output

class TLiDB_model:
    def __init__(self, config):
        self.config = config

    @property
    def model(self):
        """
        Base model underlying the encoding/decoding
        model is of type transformers.PreTrainedModel
        """
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model
    
    def __call__(self, x):
        return NotImplementedError

    def transform_inputs(self, inputs):
        return NotImplementedError

    def transform_outputs(self, outputs):
        return NotImplementedError

    def to(self, device):
        self.model.to(device)