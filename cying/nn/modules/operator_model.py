import torch.nn as nn
from .operator_layer import OperatorLayer1d, OperatorLayer2d, OperatorLayer3d

class BaseOperatorModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.num_layers = len(params)
        self.opt_net = self._create_opt_net()

    def _create_opt_net(self):
        raise NotImplementedError

    def _get_opt_weight(self):
        return [p for name, p in self.named_parameters() if 'opt_weight' in name]

    def forward(self, input):
        return self.opt_net(input)


class OperatorModel1d(BaseOperatorModel):
    def __init__(self, params):
        super().__init__(params)

    def _create_opt_net(self):
        return nn.Sequential(
            *[OperatorLayer1d(**param) for param in self.params]
        )


class OperatorModel2d(BaseOperatorModel):
    def __init__(self, params):
        super().__init__(params)

    def _create_opt_net(self):
        return nn.Sequential(
            *[OperatorLayer2d(**param) for param in self.params]
        )


class OperatorModel3d(BaseOperatorModel):
    def __init__(self, params):
        super().__init__(params)

    def _create_opt_net(self):
        return nn.Sequential(
            *[OperatorLayer3d(**param) for param in self.params]
        )