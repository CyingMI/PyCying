import torch.nn as nn
from .opt_layer import OptLayer1d, OptLayer2d, OptLayer3d

class BaseModel(nn.Module):
    def __init__(self, size, params):
        super().__init__()
        self.size = size
        self.params = params
        self.num_layers = len(params)
        self.opt_net = self._create_opt_net()

    def _create_opt_net(self):
        raise NotImplementedError
    
    def _get_opt_weight(self):
        return [p for name, p in self.named_parameters() if 'opt_weight' in name]

    def forward(self, input):
        return self.opt_net(input)


class OptModel1d(BaseModel):
    def __init__(self, size, params):
        super().__init__(size, params)

    def _create_opt_net(self):
        return nn.Sequential(
            *[OptLayer1d(self.size,**param) for param in self.params]
        )


class OptModel2d(BaseModel):
    def __init__(self, size, params):
        super().__init__(size, params)

    def _create_opt_net(self):
        return nn.Sequential(
            *[OptLayer2d(self.size,**param) for param in self.params]
        )


class OptModel3d(BaseModel):
    def __init__(self, size, params):
        super().__init__(size, params)

    def _create_opt_net(self):
        return nn.Sequential(
            *[OptLayer3d(self.size,**param) for param in self.params]
        )