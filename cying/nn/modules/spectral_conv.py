import torch
import torch.nn as nn
from .. import functional as F
import torch
import torch.nn as nn


class BSpline1D(nn.Module):
    def __init__(
        self,
        out_dim,
        num_control_points=16,
        degree=3,
        x_min=0.0,
        x_max=1.0
    ):
        super().__init__()

        self.N = num_control_points
        self.p = degree

        self.x_min = x_min
        self.x_max = x_max


        self.control_points = nn.Parameter(
            torch.zeros(self.N, out_dim)
        )


        # 基本间距
        h = (x_max-x_min)/(self.N-self.p)


        # 扩展 knot
        knots = (
            torch.arange(
                -self.p,
                self.N+1,
                dtype=torch.float32
            )
            * h
            + x_min
        )


        self.register_buffer(
            "knots",
            knots
        )



    def basis(self,x):

        shape=x.shape

        x=x.reshape(-1,1)

        t=self.knots

        B = (
            (x >= t[:-1]) &
            (x < t[1:])
        ).to(x.dtype)

        for d in range(1,self.p+1):

            n=B.shape[1]-1

            i=torch.arange(
                n,
                device=x.device
            )

            a=t[i+d]-t[i]
            b=t[i+d+1]-t[i+1]

            left=torch.zeros_like(B[:,:-1])
            right=torch.zeros_like(B[:,:-1])

            mask=a>0
            left[:,mask]=(
                (x-t[i[mask]])
                /
                a[mask]
            )*B[:,i[mask]]

            mask=b>0
            right[:,mask]=(
                (t[i[mask]+d+1]-x)
                /
                b[mask]
            )*B[:,i[mask]+1]

            B=left+right

        return B.reshape(
            *shape,
            self.N
        )

    def forward(self,x):

        return self.basis(x) @ self.control_points

class BaseSpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, opt_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.opt_size = opt_size
        self.register_buffer('k1', torch.fft.fftfreq(opt_size[0])[:,None])
        self.register_buffer('k2', torch.fft.rfftfreq(opt_size[0])[:,None])

        self.phi1 = BSpline1D(out_dim=5, num_control_points=64, degree=5, x_range=(-0.5,0.5))
        self.phi2 = BSpline1D(out_dim=5, num_control_points=64, degree=5, x_range=(0,0.5))
        
        self.Phi = nn.Sequential(
            nn.Conv2d(5, 128, 1),
            nn.SiLU(),
            nn.Conv2d(128, 2*in_channels, 1)
        )
        
        self.mask_weight = nn.Parameter(torch.randn(in_channels, *self.opt_size) * 0.01, requires_grad=True)

    def _get_conv_function(self):
        raise NotImplementedError

    def forward(self, input, mode = 'train'):
        phi1 = self.phi1(self.k1).T[:,:,None]
        phi2 = self.phi2(self.k2).T[:,None,:]
        phi = phi1 + phi2
        opt_weight = self.Phi(phi[None,...]).view(2, self.in_channels, *self.opt_size).permute(1,2,3,0).contiguous()
        
        mask = torch.sigmoid(self.mask_weight)
        mask = mask * (1 - mask) * 4
        if mode == 'valid':
            weight = torch.view_as_complex(opt_weight) * mask
        else:
            weight = torch.view_as_complex(opt_weight) * (mask > 0.5)
        return self._get_conv_function()(input, weight)


class SpectralConv1d(BaseSpectralConv):
    def __init__(self, in_channels, opt_size):
        super().__init__(in_channels, opt_size)

    def _get_conv_function(self):
        return F.spectral_conv1d


class SpectralConv2d(BaseSpectralConv):
    def __init__(self, in_channels, opt_size):
        super().__init__(in_channels, opt_size)

    def _get_conv_function(self):
        return F.spectral_conv2d


class SpectralConv3d(BaseSpectralConv):
    def __init__(self, in_channels, opt_size):
        super().__init__(in_channels, opt_size)

    def _get_conv_function(self):
        return F.spectral_conv3d