import torch
from examples.losses import initialize_loss

# NOTES: What do models have?
# attributes: possible tasks, single model, multiple possible output layers
# methods: transform_input, transform_output

class TLiDB_model:
    def __init__(self, config):
        self.config = config
        self._loss = initialize_loss(config.loss_function)

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
    def loss(self):
        return self._loss

    @property
    def layers(self):
        return

    @layers.setter
    def layers(self, layers):
        self._layers = layers

    @layers.getter
    def layers(self):
        return self._layers

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


def get_loss(loss_function):
    """
    Returns a loss function based on the loss function name
    """
    if loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_function == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError("Loss function {} not supported".format(loss_function))