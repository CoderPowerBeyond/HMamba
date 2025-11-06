# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
def arcosh(x):
    return Arcosh.apply(x)
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model+1, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
       

        
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        # self.eps = {torch.float32: 1e-3, torch.float64: 1e-8}
        self.min_norm = 1e-15
        # self.min_norm = 1e-3
        # self.min_norm = 1e-6
        # self.max_norm = 1e6
        self.max_norm = 1e1
        # self.max_norm = 1e1
        self.l3 = nn.Linear(65, 64)
        self.acti_sig = nn.Sigmoid()
        # self.acti_sig = nn.ReLU()
        # self.acti_sig = nn.SiLU()
        self.alpha = torch.nn.Parameter(torch.ones(1)*1.0)
        self.cp = torch.nn.Parameter(torch.ones(1)*1.0)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # self.fourier_proj_real = nn.Linear(200,200)
        # self.fourier_proj_image = nn.Linear(200,200)
        # self.fourier_proj_A = nn.Linear(2,1)
        # self.fourier_proj_B = nn.Linear(2,1)
        # self.fourier_proj_C = nn.Linear(66,200)
        self.tcn = nn.Conv1d(128, 128, 4, stride=3)
        self.tcn_linear = nn.Linear(66, 200)
        self.tcn_final = nn.Conv1d(200, 200, 2, stride=2)
        # self.normlization = nn.functional.normalize(input, p=2.0, dim=-1, eps=1e-12)

    
    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1)-1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[...,0] = 0
        vals = torch.zeros_like(x)
        vals[..., 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        # return (vals + mask * x)/(d)
        return vals + mask * x


    def sqdist(self, x, y):
        c= 1.0
        K = 1. / c
        # print("&:", x.size(), y.size())
        # prod = self.minkowski_dot(x, y)
        # import pdb
        # pdb.set_trace()
        x.narrow(-1, 0, 1).mul_(-1)
        theta = torch.clamp(-x @ y / K, min=1.0+ self.eps[x.dtype], max = 10.0)
        
        sqdist = K * arcosh(theta) ** 2
        # print("check prod:", torch.mean(prod))
        # theta = torch.clamp(-prod / K, min=1.0+ self.eps[x.dtype], max = 10.0)
        # print("theta:", torch.mean(theta))
        # sqdist = K * arcosh(theta) ** 2
        # # print("sdist:", torch.mean(sqdist))
        # sqdist = nn.functional.normalize(sqdist, p=2.0, dim=-1, eps=1e-12)
        return torch.clamp(sqdist, max=30.0)
       
    # def minkowski_dot(self, x, y, keepdim=True):
    #     # import pdb
    #     # pdb.set_trace()
    #     # print("**:", x.size(), y.size())
    #     # res = torch.sum(x * y, dim=-1) - 2 * x[..., 0].mean() * y[..., 0].mean()
    #     # print("check shape here:", x[..., 0].shape, y[..., 0].shape)
    #     # res = torch.matmul(x.unsqueeze(1), y.unsqueeze(0)).squeeze() - 2 * x[..., 0].mean() * y[..., 0].mean()

    #     # res = torch.sum(torch.matmul(x,y), dim=-1) - 2 * torch.matmul(x[..., 0],y[0, ...]) 
        
    #     if keepdim:
    #         res = res.view(res.shape + (1,))
    #     return res
    def compute_dist(self, x,y):
        # dsc = self.acti_sig(1/ self.sqdist(x,y))
        # dsc = - 1/ self.sqdist(x,y)
        # dsc = torch.exp(-self.alpha*(self.sqdist(x,y))-self.cp)
        # dsc = -2 * self.cp - 2 * self.sqdist(x,y)
        dsc = -2 * 1.0 - 2 * self.sqdist(x,y)
        # dsc = self.acti_sig(-2 * 1.0 - 2 * self.sqdist(x,y))
        # dsc = torch.exp(self.acti_sig(1/ self.sqdist(x,y)))
        return dsc
   

    
    def expmap0(self, u):
        # import pdb
        # pdb.set_trace()
        # if u.size(-1)==65:
        # u = self.l3(u)
        c = 1.0
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # x_norm = torch.clamp(x_norm, min=self.min_norm, max = self.max_norm)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[..., 0:1] = sqrtK * torch.cosh(theta.view(u.size(0), u.size(1), -1))
        res[..., 1:] = sqrtK * torch.sinh(theta.view(u.size(0), u.size(1), -1)) * x.view(u.size(0), u.size(1), -1)  / x_norm.view(u.size(0), u.size(1), -1)
        return self.proj(res, c)
    
    def logmap0(self, x):
        c = 1.0
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)

        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm, max=self.max_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, :, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:,:, 1:] = sqrtK * arcosh(theta.view(x.size(0), x.size(1), -1)) * y.view(x.size(0), x.size(1), -1) / y_norm.view(x.size(0), x.size(1), -1)
        return res
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # print("check shape:", hidden_states.size())
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            
            u = torch.zeros(x.shape) 
                   
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
           
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            assert self.activation in ["silu", "swish"]
            # x = torch.fft.ifft(x,dim=1)
            # x = torch.cosh(x)
            # x = torch.cosh(x) + torch.sinh(x)  # dimension: 1, 128, 200
            # x = torch.exp(x)
            
            # x = self.expmap(x)
            # dt = torch.cosh(dt)
            # print("change x:", torch.norm(x, p=2))
            # import pdb
            # pdb.set_trace()
            # x = self.tcn_linear(self.tcn(x))
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            
            # print("neural output:", torch)    
            # print("----check y1----:", torch.mean(y))
            # y = self.logmap(x,y)
            # print("----check y2----:", torch.mean(y))
            y = rearrange(y, "b d l -> b l d")
            # out = self.proj(y, c)
            # 
            # 
            # tmp_value = torch.sqrt(out*out+K)
            # out = torch.cat([tmp_value, out], dim=-1)
            # print("----check y3----:", torch.mean(y))
            out = self.out_proj(y)
            # print("----check y4----:", torch.mean(out))
            # out = torch.log(out) +1e-6
            
            # transfor back
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
