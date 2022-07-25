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
    def named_modules(self):
        return self._model.named_modules

    @property
    def half(self):
        return self._model.half

    @half.setter
    def half(self, half):
        self._model.half = half

    @property
    def training(self):
        return self._model.training

    @training.setter
    def training(self, training):
        self._model.training = training

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

    def state_dict(self, destination, prefix, keep_vars):
        state_dict = {}
        for layer_name, layer in self.layers.items():
            state_dict[layer_name] = layer.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
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

    def init_weights(self):
        for layer_name, layer in self.layers.items():
            # main model is always Transformers-based and does it's own init
            if layer_name == "model":
                pass
            elif isinstance(layer, torch.nn.Module):
                torch.nn.init.xavier_uniform_(layer.weight)
            else:
                raise ValueError("Layer {} is not a torch.nn.Module or Transformers Model".format(layer_name))

    @property
    def forward(self):
        return NotImplementedError

    @forward.setter
    def forward(self, forward):
        self._forward = forward

    def backward(self, objective):
        return self.model.backward(objective)

    def step(self):
        return self.model.step()

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

    def parallelize(self):
        if self.model.is_parallelizable:
            self.model.parallelize()
        else:
            raise TypeError("Model is not parallelizable")