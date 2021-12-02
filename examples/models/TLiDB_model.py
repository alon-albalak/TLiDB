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
        return self._layers

    @layers.setter
    def layers(self, layers):
        self._layers = layers

    @layers.getter
    def layers(self):
        return self._layers

    def parameters(self):
        """Convenience function to gather all model parameters
            to be passed to a torch.optim optimizer"""
        params = []
        for layer_name, layer in self.layers.items():
            params.extend(layer.parameters())

        return params

    def named_parameters(self):
        """Convenience function to gather all named model parameters
            to be passed to a torch.optim optimizer"""
        named_params = chain()
        for layer_name, layer in self.layers.items():
            named_params = chain(named_params, layer.named_parameters())
        return named_params

    def state_dict(self):
        state_dict = {}
        for layer_name, layer in self.layers.items():
            state_dict[layer_name] = layer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """Convenience function to load state dict for all layers"""
        return NotImplementedError

    def zero_grad(self, set_to_none=True):
        """Convenience function to zero gradients for all layers"""
        for layer_name, layer in self.layers.items():
            layer.zero_grad(set_to_none=set_to_none)

    def train(self, mode=True):
        """Convenience function to set all layers to train mode"""
        for layer_name, layer in self.layers.items():
            layer.train(mode)

    @property
    def forward(self):
        return NotImplementedError

    @forward.setter
    def forward(self, forward):
        self._forward = forward

    def __call__(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def transform_inputs(self, inputs):
        return NotImplementedError

    def transform_outputs(self, outputs):
        return NotImplementedError

    def to(self, device):
        """Convenience function to move all layers to a device"""
        for layer_name, layer in self.layers.items():
            layer.to(device)
