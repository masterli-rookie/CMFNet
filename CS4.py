import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fmconv_crack import FMConv

try:
    from timm.layers import DropPath
except Exception:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

from models.Vssmamba import VSSBlock


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def autopad(k, p=None, d=1):
    if p is not None:
        return p
    return ((k - 1) * d) // 2


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def conv_flops(c_in, c_out, k, h, w, groups=1, has_bias=False):
    """单次 conv2d 的 FLOPs (multiply-adds 计为 1)。"""
    kh, kw = (k, k) if isinstance(k, int) else k
    flops = c_in * c_out * kh * kw * h * w
    if groups != 1:
        flops = flops / groups * groups   # depthwise: c_in*1*kh*kw*h*w * c_in
        # 实际 depthwise: c_in * kh * kw * h * w
        flops = c_in * kh * kw * h * w
    if has_bias:
        flops += c_out * h * w
    return flops


def linear_flops(c_in, c_out, n_tokens):
    return c_in * c_out * n_tokens


def count_flops_g(model, input_shape=(3, 256, 256), device='cpu'):
    """
    解析式 FLOPs 估算 (GMac)：
    覆盖 Conv2d / Linear / DepthwiseConv。
    注：精确 FLOPs 需 trace forward，此处为模块级近似。
    """
    total = 0.0
    h, w = input_shape[1], input_shape[2]
    c_in = input_shape[0]

    hooks = []

    def conv_hook(module, inp, out):
        nonlocal total
        x = inp[0]
        b, c, hh, ww = x.shape
        oh, ow = out.shape[-2:]
        k = module.kernel_size
        g = module.groups
        c_out = module.out_channels
        c_in_eff = module.in_channels
        # 输出每点 cost: c_in_eff/g * k*k (每个输出通道组)
        total += c_out * (c_in_eff // g) * (k[0] * k[1]) * oh * ow * b / b  # 去掉 batch
        if module.bias is not None:
            total += c_out * oh * ow

    def linear_hook(module, inp, out):
        nonlocal total
        x = inp[0]
        n = x.numel() // x.shape[-1]  # token 数
        total += module.in_features * module.out_features * n

    def gn_hook(module, inp, out):
        nonlocal total
        x = inp[0]
        # GroupNorm: 2 * numel (mean+var+normalize)
        total += 2 * x.numel() / x.shape[0]

    def ln_hook(module, inp, out):
        nonlocal total
        x = inp[0]
        total += 2 * x.numel() / x.shape[0]

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.GroupNorm):
            handles.append(m.register_forward_hook(gn_hook))
        elif isinstance(m, nn.LayerNorm):
            handles.append(m.register_forward_hook(ln_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, *input_shape, device=device)
        try:
            model(dummy)
        except Exception as e:
            print("[FLOPs] forward 失败:", e)
    if was_training:
        model.train()

    for h in handles:
        h.remove()

    return total / 1e9


# ---------------------------------------------------------------------------
# 基础模块
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class ConvGNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, gn_groups=1):
        super().__init__()
        p = autopad(k, p=p, d=d)
        gn_groups = min(gn_groups, c2)
        if c2 % gn_groups != 0:
            gn_groups = 1
        self.conv = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=False)
        self.norm = nn.GroupNorm(gn_groups, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DSConvGNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, d=1, act=True, gn_groups=1):
        super().__init__()
        self.dw = nn.Conv2d(
            c1, c1,
            kernel_size=k, stride=s,
            padding=autopad(k, p=None, d=d),
            dilation=d, groups=c1, bias=False,
        )

        gn_groups1 = min(gn_groups, c1)
        if c1 % gn_groups1 != 0:
            gn_groups1 = 1
        gn_groups2 = min(gn_groups, c2)
        if c2 % gn_groups2 != 0:
            gn_groups2 = 1

        self.gn1 = nn.GroupNorm(gn_groups1, c1)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups2, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.gn1(self.dw(x)))
        x = self.act(self.gn2(self.pw(x)))
        return x


class MultiScaleStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32):
        super().__init__()
        mid = max(embed_dim // 2, 16)
        self.branch3 = nn.Sequential(
            ConvGNAct(in_chans, mid, k=3, s=2),
            DSConvGNAct(mid, mid, k=3, s=1),
        )
        self.branch5 = nn.Sequential(
            ConvGNAct(in_chans, mid, k=5, s=2),
            DSConvGNAct(mid, mid, k=3, s=1),
        )
        self.branch7 = nn.Sequential(
            ConvGNAct(in_chans, mid, k=7, s=2),
            DSConvGNAct(mid, mid, k=3, s=1),
        )
        self.fuse = nn.Sequential(
            ConvGNAct(mid * 3, embed_dim, k=1, s=1, p=0),
            DSConvGNAct(embed_dim, embed_dim, k=3, s=1),
        )

    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        return self.fuse(torch.cat([x3, x5, x7], dim=1))


class Downsample(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class SimpleMambaBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, d_state=16):
        super().__init__()
        self.block = VSSBlock(hidden_dim=dim, drop_path=drop_path, d_state=d_state)

    def forward(self, x):
        return self.block(x)


class MultiScaleLocalBlock(nn.Module):
    def __init__(self, dim, expansion=2.0, drop_path=0.0):
        super().__init__()
        hidden = max(int(dim * expansion), dim)
        self.norm = LayerNorm2d(dim)
        self.b3 = DSConvGNAct(dim, dim, k=3, s=1)
        self.b5 = DSConvGNAct(dim, dim, k=5, s=1)
        self.bd = DSConvGNAct(dim, dim, k=3, s=1, d=2)
        self.fuse = nn.Sequential(
            ConvGNAct(dim * 3, hidden, k=1, s=1, p=0),
            DSConvGNAct(hidden, hidden, k=3, s=1),
            ConvGNAct(hidden, dim, k=1, s=1, p=0, act=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y3 = self.b3(y)
        y5 = self.b5(y)
        yd = self.bd(y)
        y = self.fuse(torch.cat([y3, y5, yd], dim=1))
        return x + self.drop_path(self.gamma * y)


class EdgeAwareGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(1, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e = self.edge(x)
        g = self.gate(e)
        return x * g + x


class ContextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGNAct(dim, dim, k=1, s=1, p=0),
            ConvGNAct(dim, dim, k=1, s=1, p=0, act=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)


class EncoderStage(nn.Module):
    def __init__(self, dim, depth, next_dim=None, drop_path_rate=0.0,
                 use_mamba=False, d_state=16,
                 use_context=False, use_edge_gate=False):
        super().__init__()
        if depth > 1:
            dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        else:
            dpr = [drop_path_rate]

        blocks = []
        for i in range(depth):
            blocks.append(MultiScaleLocalBlock(dim, expansion=2.0, drop_path=dpr[i] * 0.5))
            if use_mamba and i == depth - 1:
                blocks.append(SimpleMambaBlock(dim, drop_path=dpr[i] * 0.5, d_state=d_state))

        self.blocks = nn.Sequential(*blocks)
        self.context = ContextBlock(dim) if use_context else nn.Identity()
        self.edge_gate = EdgeAwareGate(dim) if use_edge_gate else nn.Identity()
        self.down = Downsample(dim, next_dim) if next_dim is not None else None

    def forward(self, x):
        x = self.blocks(x)
        x = self.context(x)
        x = self.edge_gate(x)
        skip = x
        if self.down is not None:
            x = self.down(x)
        return skip, x


# ---------------------------------------------------------------------------
# 反向调制：decoder -> encoder
# ---------------------------------------------------------------------------
class ReverseEncoderModulation(nn.Module):
    """
    Decoder 反向调制 Encoder:
    用深层 decoder / bottleneck 特征去门控 (空间 + 通道) 浅层 encoder skip,
    使浅层 skip 在被 decoder 消费前就接收到深层语义的引导。

    Modulation = encoder + gamma * (encoder ⊙ spatial_gate ⊙ channel_gate)
    其中 gate 由 [encoder, deep_proj] 拼接后产生。
    """
    def __init__(self, enc_dim, deep_dim):
        super().__init__()
        self.deep_proj = ConvGNAct(deep_dim, enc_dim, k=1, s=1, p=0)
        self.enc_norm = LayerNorm2d(enc_dim)

        hidden = max(enc_dim // 4, 8)
        # 通道门
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(enc_dim * 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, enc_dim, 1, bias=False),
            nn.Sigmoid(),
        )
        # 空间门
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(enc_dim * 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, enc_feat, deep_feat):
        deep = self.deep_proj(deep_feat)
        deep = F.interpolate(deep, size=enc_feat.shape[-2:],
                             mode="bilinear", align_corners=False)
        enc = self.enc_norm(enc_feat)
        cat = torch.cat([enc, deep], dim=1)
        cg = self.channel_gate(cat)
        sg = self.spatial_gate(cat)
        modulated = enc_feat * sg * cg
        return enc_feat + self.gamma * (modulated - enc_feat)


class BiDirectionalRefine(nn.Module):
    """decoder 输出与 encoder skip 的双向融合 (post-decoder refine)。"""
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.enc_proj = ConvGNAct(enc_dim, enc_dim, k=1, s=1, p=0)
        self.dec_proj = ConvGNAct(dec_dim, enc_dim, k=1, s=1, p=0)
        self.mix = nn.Sequential(
            DSConvGNAct(enc_dim * 2, enc_dim, k=3, s=1),
            ConvGNAct(enc_dim, enc_dim, k=1, s=1, p=0, act=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, enc_skip, dec_feat):
        dec_up = F.interpolate(dec_feat, size=enc_skip.shape[-2:],
                               mode="bilinear", align_corners=False)
        enc = self.enc_proj(enc_skip)
        dec = self.dec_proj(dec_up)
        out = self.mix(torch.cat([enc, dec], dim=1))
        return enc_skip + self.gamma * out


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.x_proj = ConvGNAct(in_dim, out_dim, k=1, s=1, p=0)
        self.skip_proj = ConvGNAct(skip_dim, out_dim, k=1, s=1, p=0)
        self.fuse = nn.Sequential(
            DSConvGNAct(out_dim * 2, out_dim, k=3, s=1),
            ConvGNAct(out_dim, out_dim, k=1, s=1, p=0, act=False),
        )
        self.refine = nn.Sequential(
            DSConvGNAct(out_dim, out_dim, k=3, s=1),
            ConvGNAct(out_dim, out_dim, k=1, s=1, p=0, act=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.x_proj(x)
        skip = self.skip_proj(skip)
        y = self.fuse(torch.cat([x, skip], dim=1))
        y = y + self.gamma * self.refine(y)
        if self.use_residual:
            y = y + x
        return y


class BranchFeatureSelect(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 4, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)


class TriBranchInteraction(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim * 3, 1, bias=False),
        )

    def forward(self, feats):
        x = torch.cat(feats, dim=1)
        w = self.proj(x)
        w1, w2, w3 = torch.chunk(w, 3, dim=1)
        return [feats[0] * torch.sigmoid(w1),
                feats[1] * torch.sigmoid(w2),
                feats[2] * torch.sigmoid(w3)]


class TriBranchCrossFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch1 = BranchFeatureSelect(dim)
        self.branch2 = BranchFeatureSelect(dim)
        self.branch3 = BranchFeatureSelect(dim)
        self.interact = TriBranchInteraction(dim)

        hidden = max(dim // 2, 16)
        self.weight_mlp = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        self.global_residual = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.SiLU(inplace=True),
        )
        self.post = nn.Sequential(
            ConvGNAct(dim, dim, k=1, s=1, p=0),
            DSConvGNAct(dim, dim, k=3, s=1),
            ConvGNAct(dim, dim, k=1, s=1, p=0, act=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, p1, p2, p3):
        p1 = self.branch1(p1)
        p2 = self.branch2(p2)
        p3 = self.branch3(p3)
        p1, p2, p3 = self.interact([p1, p2, p3])
        x = torch.cat([p1, p2, p3], dim=1)
        w = torch.softmax(self.weight_mlp(x), dim=1)
        fused = p1 * w[:, 0:1] + p2 * w[:, 1:2] + p3 * w[:, 2:3]
        residual = self.global_residual(x)
        fused = fused + self.gamma * residual
        fused = self.post(fused)
        return fused


class BoundaryRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.refine = nn.Sequential(
            DSConvGNAct(dim, dim, k=3, s=1),
            ConvGNAct(dim, dim, k=1, s=1, p=0, act=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.gamma * self.refine(x)


# ---------------------------------------------------------------------------
# 模型预设
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "Lite": {
        "dims": (24, 40, 80, 128),
        "depths": (1, 1, 2, 1),
        "drop_path_rate": 0.06,
    },
    "Tiny": {
        "dims": (32, 48, 96, 160),
        "depths": (1, 2, 2, 2),
        "drop_path_rate": 0.08,
    },
    "Base": {
        "dims": (32, 64, 128, 256),
        "depths": (2, 2, 4, 2),
        "drop_path_rate": 0.10,
    },
}


def resolve_model_config(variant="Lite", dims=None, depths=None, drop_path_rate=None):
    preset = MODEL_PRESETS.get(variant, MODEL_PRESETS["Lite"])
    dims = dims if dims is not None else preset["dims"]
    depths = depths if depths is not None else preset["depths"]
    drop_path_rate = drop_path_rate if drop_path_rate is not None else preset["drop_path_rate"]
    assert len(dims) == 4
    assert len(depths) == 4
    return dims, depths, drop_path_rate


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class CrackStepNet(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1,
        variant="Lite",
        dims=None,
        depths=None,
        drop_path_rate=None,
        d_state=16,
        use_context=None,
        use_mamba=(True, True, True, True),
        use_edge_gate=(True, True, True, True),
        deep_supervision=True,
        enable_boundary_refine=True,
        decoder_residual=True,
        enable_reverse_modulation=True,
    ):
        super().__init__()

        dims, depths, drop_path_rate = resolve_model_config(
            variant=variant, dims=dims, depths=depths, drop_path_rate=drop_path_rate,
        )

        if use_context is None:
            use_context = (False, False, True, True)

        self.variant = variant
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.dims = tuple(dims)
        self.depths = tuple(depths)
        self.drop_path_rate = float(drop_path_rate)
        self.use_context = tuple(use_context)
        self.use_mamba = tuple(use_mamba)
        self.use_edge_gate = tuple(use_edge_gate)
        self.deep_supervision = bool(deep_supervision)
        self.enable_boundary_refine = bool(enable_boundary_refine)
        self.decoder_residual = bool(decoder_residual)
        self.enable_reverse_modulation = bool(enable_reverse_modulation)

        # ---- stem ----
        self.patch_embed = MultiScaleStem(in_chans=in_chans, embed_dim=dims[0])

        # ---- encoder stages ----
        self.stage1 = EncoderStage(
            dims[0], depths[0], next_dim=dims[1],
            drop_path_rate=drop_path_rate * 0.25,
            use_mamba=use_mamba[0], d_state=d_state,
            use_context=use_context[0], use_edge_gate=use_edge_gate[0],
        )
        self.stage2 = EncoderStage(
            dims[1], depths[1], next_dim=dims[2],
            drop_path_rate=drop_path_rate * 0.50,
            use_mamba=use_mamba[1], d_state=d_state,
            use_context=use_context[1], use_edge_gate=use_edge_gate[1],
        )
        self.stage3 = EncoderStage(
            dims[2], depths[2], next_dim=dims[3],
            drop_path_rate=drop_path_rate * 0.75,
            use_mamba=use_mamba[2], d_state=d_state,
            use_context=use_context[2], use_edge_gate=use_edge_gate[2],
        )
        self.stage4 = EncoderStage(
            dims[3], depths[3], next_dim=None,
            drop_path_rate=drop_path_rate,
            use_mamba=use_mamba[3], d_state=d_state,
            use_context=use_context[3], use_edge_gate=use_edge_gate[3],
        )

        # ---- 反向调制 (decoder -> encoder) ----
        # 用 bottleneck s4 逐级调制 s3 -> s2 -> s1
        if self.enable_reverse_modulation:
            self.rev_mod3 = ReverseEncoderModulation(enc_dim=dims[2], deep_dim=dims[3])
            self.rev_mod2 = ReverseEncoderModulation(enc_dim=dims[1], deep_dim=dims[2])
            self.rev_mod1 = ReverseEncoderModulation(enc_dim=dims[0], deep_dim=dims[1])
        else:
            self.rev_mod3 = nn.Identity()
            self.rev_mod2 = nn.Identity()
            self.rev_mod1 = nn.Identity()

        # ---- decoder ----
        self.decoder3 = DecoderBlock(dims[3], dims[2], dims[2], use_residual=decoder_residual)
        self.decoder2 = DecoderBlock(dims[2], dims[1], dims[1], use_residual=decoder_residual)
        self.decoder1 = DecoderBlock(dims[1], dims[0], dims[0], use_residual=decoder_residual)

        # ---- 双向 refine (post decoder) ----
        self.bi3 = BiDirectionalRefine(dims[2], dims[2])
        self.bi2 = BiDirectionalRefine(dims[1], dims[1])
        self.bi1 = BiDirectionalRefine(dims[0], dims[0])

        # ---- 浅层 refine ----
        self.shallow_refine = nn.Sequential(
            DSConvGNAct(dims[0], dims[0], k=3, s=1),
            ConvGNAct(dims[0], dims[0], k=1, s=1, p=0, act=False),
        )

        # ---- 多分支融合 ----
        self.proj_d1 = ConvGNAct(dims[0], dims[0], k=1, s=1, p=0)
        self.proj_d2 = ConvGNAct(dims[1], dims[0], k=1, s=1, p=0)
        self.proj_d3 = ConvGNAct(dims[2], dims[0], k=1, s=1, p=0)

        self.tri_fusion = TriBranchCrossFusion(dims[0])
        self.boundary_refine = BoundaryRefine(dims[0]) if enable_boundary_refine else nn.Identity()
        self.out_refine = nn.Sequential(
            DSConvGNAct(dims[0], dims[0], k=3, s=1),
            ConvGNAct(dims[0], dims[0], k=1, s=1, p=0, act=False),
        )
        self.head = nn.Conv2d(dims[0], num_classes, 1)

        if self.deep_supervision:
            self.aux_head1 = nn.Conv2d(dims[0], num_classes, 1)
            self.aux_head2 = nn.Conv2d(dims[1], num_classes, 1)
            self.aux_head3 = nn.Conv2d(dims[2], num_classes, 1)
            self.boundary_head = nn.Conv2d(dims[0], num_classes, 1)
        else:
            self.aux_head1 = None
            self.aux_head2 = None
            self.aux_head3 = None
            self.boundary_head = None

    def forward(self, x):
        input_size = x.shape[-2:]
        x0 = self.patch_embed(x)

        s1, x = self.stage1(x0)
        s2, x = self.stage2(x)
        s3, x = self.stage3(x)
        s4, _ = self.stage4(x)

        # ===== 反向调制: decoder/bottleneck -> encoder skips =====
        # 自顶向下: s4 -> s3 -> s2 -> s1
        if self.enable_reverse_modulation:
            s3 = self.rev_mod3(s3, s4)
            s2 = self.rev_mod2(s2, s3)
            s1 = self.rev_mod1(s1, s2)

        # ===== decoder 主流程 (消费被调制的 skip) =====
        d3 = self.decoder3(s4, s3)
        s3 = self.bi3(s3, d3)

        d2 = self.decoder2(d3, s2)
        s2 = self.bi2(s2, d2)

        d1 = self.decoder1(d2, s1)
        s1 = self.bi1(s1, d1)

        d1 = d1 + x0
        d1 = self.shallow_refine(d1)

        p1 = F.interpolate(self.proj_d1(d1), size=input_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.proj_d2(d2), size=input_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj_d3(d3), size=input_size, mode="bilinear", align_corners=False)

        fused = self.tri_fusion(p1, p2, p3)
        fused = self.boundary_refine(fused)
        fused = self.out_refine(fused)
        main_out = self.head(fused)

        if not self.deep_supervision:
            return main_out

        aux = {
            "aux_d1": F.interpolate(self.aux_head1(d1), size=input_size, mode="bilinear", align_corners=False),
            "aux_d2": F.interpolate(self.aux_head2(d2), size=input_size, mode="bilinear", align_corners=False),
            "aux_d3": F.interpolate(self.aux_head3(d3), size=input_size, mode="bilinear", align_corners=False),
            "boundary": F.interpolate(self.boundary_head(fused), size=input_size, mode="bilinear", align_corners=False),
        }
        return main_out, aux

    def extra_repr(self):
        return (
            f"variant={self.variant}, in_chans={self.in_chans}, num_classes={self.num_classes}, "
            f"dims={self.dims}, depths={self.depths}, drop_path_rate={self.drop_path_rate}, "
            f"use_context={self.use_context}, use_mamba={self.use_mamba}, use_edge_gate={self.use_edge_gate}, "
            f"deep_supervision={self.deep_supervision}, enable_boundary_refine={self.enable_boundary_refine}, "
            f"decoder_residual={self.decoder_residual}, "
            f"enable_reverse_modulation={self.enable_reverse_modulation}"
        )


def build_model(
    variant="Lite",
    in_chans=3,
    num_classes=1,
    dims=None,
    depths=None,
    drop_path_rate=None,
    d_state=16,
    use_context=None,
    use_mamba=(True, True, True, True),
    use_edge_gate=(True, True, True, True),
    deep_supervision=True,
    enable_boundary_refine=True,
    decoder_residual=True,
    enable_reverse_modulation=True,
):
    return CrackStepNet(
        in_chans=in_chans, num_classes=num_classes, variant=variant,
        dims=dims, depths=depths, drop_path_rate=drop_path_rate,
        d_state=d_state, use_context=use_context,
        use_mamba=use_mamba, use_edge_gate=use_edge_gate,
        deep_supervision=deep_supervision,
        enable_boundary_refine=enable_boundary_refine,
        decoder_residual=decoder_residual,
        enable_reverse_modulation=enable_reverse_modulation,
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for variant in ["Lite", "Tiny", "Base"]:
        print("=" * 70)
        print(f"Variant: {variant}")
        model = build_model(
            variant=variant,
            in_chans=3,
            num_classes=1,
            deep_supervision=True,
            enable_boundary_refine=True,
            decoder_residual=True,
            enable_reverse_modulation=True,
        ).to(device)

        x = torch.randn(2, 3, 256, 256, device=device)
        y = model(x)

        if isinstance(y, tuple):
            main_out, aux = y
            print("main_out:", tuple(main_out.shape))
            for k, v in aux.items():
                print(f"  {k}: {tuple(v.shape)}")
        else:
            print("out:", tuple(y.shape))

        params_m = count_params_m(model)
        train_m = count_trainable_params_m(model)
        flops_g = count_flops_g(model, input_shape=(3, 256, 256), device=device)

        print(f"Params (M)        : {params_m:.4f}")
        print(f"Trainable ()     : {train_m:.4f}")
        print(f"FLOPs (GMac) @256 : {flops_g:.4f}")
        # 参数效率: GMac / M_params (越低表示每参数算力开销越省, 反映参数利用效率)
        if params_m > 0:
            print(f"FLOPs / Param     : {flops_g / params_m:.4f}  (GMac per M-param)")
        print()
