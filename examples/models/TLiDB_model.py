import torch
from itertools import chain

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

    @property
    def layers(self):
        return

    @layers.setter
    def layers(self, layers):
        self._layers = layers

    @layers.getter
    def layers(self):
        return self._layers

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def named_parameters(self):
        named_params = chain()
        for layer in self.layers:
            named_params = chain(named_params, layer.named_parameters())
        return named_params

    def zero_grad(self, set_to_none=True):
        for layer in self.layers:
            layer.zero_grad(set_to_none=set_to_none)

    @property
    def forward(self):
        return NotImplementedError

    @forward.setter
    def forward(self, forward):
        self._forward = forward

    def __call__(self, x):
        return self._forward(x)

    def transform_inputs(self, inputs):
        return NotImplementedError

    def transform_outputs(self, outputs):
        return NotImplementedError

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
