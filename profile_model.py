import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.CS import build_model


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def model_size_mb(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / 1024 / 1024


def _strip_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_weights(model, weights_path, device="cpu", strict=True):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return model, missing, unexpected


def parse_int_tuple(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def _numel(shape):
    n = 1
    for v in shape:
        n *= int(v)
    return int(n)


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _call_flops_method(module, inp_shape):
    if not hasattr(module, "flops"):
        return None

    H = inp_shape[-2] if len(inp_shape) >= 2 else None
    W = inp_shape[-1] if len(inp_shape) >= 2 else None

    trials = [()]
    if H is not None and W is not None:
        trials.append((H, W))
    trials.append((inp_shape,))
    try:
        trials.append((torch.zeros(inp_shape),))
    except Exception:
        pass

    for args in trials:
        try:
            if len(args) == 0:
                val = module.flops()
            else:
                val = module.flops(*args)
            if isinstance(val, (int, float)):
                return float(val)
        except Exception:
            pass
    return None


def estimate_standard_module_flops(module, inp, out):
    x = _first_tensor(inp)
    y = _first_tensor(out)

    if x is None and y is None:
        return 0.0

    if isinstance(module, nn.Conv2d):
        if y is None:
            return 0.0
        b, cout, hout, wout = y.shape
        cin = module.in_channels
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        flops = 2.0 * b * cout * hout * wout * (cin / groups * kh * kw)
        return float(flops)

    if isinstance(module, nn.ConvTranspose2d):
        if y is None:
            return 0.0
        b, cout, hout, wout = y.shape
        cin = module.in_channels
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        flops = 2.0 * b * cout * hout * wout * (cin / groups * kh * kw)
        return float(flops)

    if isinstance(module, nn.Linear):
        if x is None:
            return 0.0
        in_features = module.in_features
        out_features = module.out_features
        vecs = _numel(x.shape[:-1])
        flops = 2.0 * vecs * in_features * out_features
        return float(flops)

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                           nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                           nn.LayerNorm)):
        if y is None:
            return 0.0
        return float(5.0 * _numel(y.shape))

    if module.__class__.__name__ == "LayerNorm2d":
        if y is None:
            return 0.0
        return float(5.0 * _numel(y.shape))

    if isinstance(module, nn.AdaptiveAvgPool2d):
        if x is None:
            return 0.0
        return float(_numel(x.shape))

    if isinstance(module, (nn.ReLU, nn.SiLU, nn.GELU, nn.Sigmoid, nn.Dropout, nn.Identity, nn.Dropout2d, nn.Dropout3d)):
        return 0.0

    if isinstance(module, nn.Softmax):
        return 0.0

    return 0.0


def estimate_vssblock_flops(module, inp, out):
    x = _first_tensor(inp)
    if x is None or x.ndim != 4:
        return 0.0

    shape = tuple(x.shape)
    val = _call_flops_method(module, shape)
    if val is not None:
        return float(val)

    b, c, h, w = shape
    attn = getattr(module, "self_attention", None)

    d_model = getattr(attn, "d_model", c)
    d_inner = getattr(attn, "d_inner", d_model * 2)
    d_state = getattr(attn, "d_state", 16)
    d_conv = getattr(attn, "d_conv", 3)

    tokens = b * h * w

    # Rough estimate for SS2D / VSSBlock:
    # in_proj + depthwise conv + scan + out_proj + norm
    flops = 0.0
    flops += 2.0 * tokens * d_model * (2 * d_inner)
    flops += 2.0 * b * d_inner * h * w * (d_conv * d_conv)
    flops += 6.0 * tokens * d_inner * d_state
    flops += 2.0 * tokens * d_inner * d_model
    flops += 5.0 * tokens * d_inner
    return float(flops)


def estimate_interpolate_extra(inp, target_hw):
    x = _first_tensor(inp)
    if x is None or x.ndim != 4:
        return 0.0
    b, c, _, _ = x.shape
    th, tw = target_hw
    return float(8.0 * b * c * th * tw)


def profile_gflops(model, img_size=512, in_chans=3, device="cuda"):
    model = model.to(device).eval()
    x = torch.randn(1, in_chans, img_size, img_size, device=device)

    total_flops = 0.0
    handles = []

    vss_prefixes = []
    for name, m in model.named_modules():
        if m.__class__.__name__ == "VSSBlock":
            vss_prefixes.append(name)

    def _inside_vss(name):
        for p in vss_prefixes:
            if name == p or name.startswith(p + "."):
                return True
        return False

    def make_hook(name):
        def hook(mod, inp, out):
            nonlocal total_flops

            if _inside_vss(name) and mod.__class__.__name__ != "VSSBlock":
                return

            cls = mod.__class__.__name__

            if cls == "VSSBlock":
                total_flops += estimate_vssblock_flops(mod, inp, out)
                return

            total_flops += estimate_standard_module_flops(mod, inp, out)

            if cls == "UpFuse":
                x_in = inp[0] if isinstance(inp, (list, tuple)) and len(inp) > 0 else None
                skip = inp[1] if isinstance(inp, (list, tuple)) and len(inp) > 1 else None
                x_t = _first_tensor(x_in)
                skip_t = _first_tensor(skip)
                if x_t is not None and skip_t is not None and x_t.ndim == 4 and skip_t.ndim == 4:
                    total_flops += estimate_interpolate_extra(x_in, skip_t.shape[-2:])
            elif name == "head":
                out_t = _first_tensor(out)
                if out_t is not None and out_t.ndim == 4:
                    total_flops += estimate_interpolate_extra(out, (img_size, img_size))

        return hook

    for name, m in model.named_modules():
        if name == "":
            continue
        handles.append(m.register_forward_hook(make_hook(name)))

    with torch.inference_mode():
        _ = model(x)

    for h in handles:
        h.remove()

    return total_flops / 1e9


@torch.inference_mode()
def profile_fps(
    model,
    img_size=512,
    in_chans=3,
    device="cuda",
    batch_size=1,
    warmup=30,
    iters=100,
    amp=False,
):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = model.to(device).eval()
    x = torch.randn(batch_size, in_chans, img_size, img_size, device=device)

    def _forward():
        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                return model(x)
        return model(x)

    for _ in range(warmup):
        _ = _forward()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _forward()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = batch_size * iters / elapsed
    latency_ms_img = elapsed * 1000.0 / (iters * batch_size)
    return fps, fps, latency_ms_img


def get_model_meta(model, fallback_dims=None, fallback_depths=None, fallback_drop_path=None):
    dims = getattr(model, "dims", None)
    depths = getattr(model, "depths", None)
    drop_path_rate = getattr(model, "drop_path_rate", None)

    if dims is None:
        dims = fallback_dims
    if depths is None:
        depths = fallback_depths
    if drop_path_rate is None:
        drop_path_rate = fallback_drop_path

    return dims, depths, drop_path_rate


def build_and_profile(
    model=None,
    variant="Lite",
    in_chans=3,
    num_classes=1,
    img_size=512,
    device="cuda",
    batch_size=1,
    weights=None,
    strict_load=True,
    warmup=30,
    iters=100,
    dims=None,
    depths=None,
    drop_path_rate=None,
    amp=False,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is not available, fallback to CPU.")
        device = "cpu"

    if model is None:
        build_kwargs = {}
        if dims is not None:
            build_kwargs["dims"] = dims
        if depths is not None:
            build_kwargs["depths"] = depths
        if drop_path_rate is not None:
            build_kwargs["drop_path_rate"] = drop_path_rate

        print("[INFO] Building model with:")
        print(f"       variant        = {variant}")
        print(f"       in_chans       = {in_chans}")
        print(f"       num_classes    = {num_classes}")
        print(f"       dims           = {dims}")
        print(f"       depths         = {depths}")
        print(f"       drop_path_rate  = {drop_path_rate}")

        model = build_model(
            variant=variant,
            in_chans=in_chans,
            num_classes=num_classes,
            **build_kwargs
        )

    real_dims, real_depths, real_drop_path = get_model_meta(
        model,
        fallback_dims=dims,
        fallback_depths=depths,
        fallback_drop_path=drop_path_rate
    )

    missing_keys, unexpected_keys = [], []
    if weights is not None and str(weights).strip() != "":
        weights = Path(weights)
        if weights.exists():
            model, missing_keys, unexpected_keys = load_weights(
                model=model,
                weights_path=weights,
                device="cpu",
                strict=strict_load
            )
            print(f"[INFO] Loaded weights from: {weights}")
            if not strict_load:
                if missing_keys:
                    print(f"[Warning] Missing keys (first 20): {missing_keys[:20]}")
                if unexpected_keys:
                    print(f"[Warning] Unexpected keys (first 20): {unexpected_keys[:20]}")
        else:
            print(f"[Warning] weights not found: {weights}")

    total_params, trainable_params = count_params(model)
    size_mb = model_size_mb(model)

    gflops = profile_gflops(
        model=model,
        img_size=img_size,
        in_chans=in_chans,
        device=device
    )

    try:
        fps, throughput, latency_ms_img = profile_fps(
            model=model,
            img_size=img_size,
            in_chans=in_chans,
            device=device,
            batch_size=batch_size,
            warmup=warmup,
            iters=iters,
            amp=amp,
        )
    except Exception as e:
        print(f"[Warning] FPS profiling failed: {e}")
        fps, throughput, latency_ms_img = None, None, None

    result = {
        "variant": variant,
        "img_size": img_size,
        "batch_size": batch_size,
        "device": device,
        "in_chans": in_chans,
        "num_classes": num_classes,
        "dims": None if real_dims is None else list(real_dims),
        "depths": None if real_depths is None else list(real_depths),
        "drop_path_rate": real_drop_path,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "params_m": round(total_params / 1e6, 4),
        "trainable_params_m": round(trainable_params / 1e6, 4),
        "model_size_mb": round(size_mb, 4),
        "gflops": None if gflops is None else round(float(gflops), 4),
        "fps": None if fps is None else round(float(fps), 4),
        "throughput": None if throughput is None else round(float(throughput), 4),
        "latency_ms_img": None if latency_ms_img is None else round(float(latency_ms_img), 4),
        "weights": None if weights is None else str(weights),
        "strict_load": strict_load,
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "model_class": model.__class__.__name__,
    }
    return result


def pretty_print_result(result):
    print("\n" + "=" * 80)
    print(f"Model Class   : {result['model_class']}")
    print(f"Variant       : {result['variant']}")
    print(f"Img Size      : {result['img_size']}")
    print(f"Batch Size    : {result['batch_size']}")
    print(f"Device        : {result['device']}")
    print(f"In Chans      : {result['in_chans']}")
    print(f"Num Classes   : {result['num_classes']}")
    print(f"Dims          : {result['dims']}")
    print(f"Depths        : {result['depths']}")
    print(f"Drop Path     : {result['drop_path_rate']}")
    print(f"Params        : {result['params_m']} M")
    print(f"Trainable     : {result['trainable_params_m']} M")
    print(f"Model Size    : {result['model_size_mb']} MB")
    print(f"GFLOPs        : {result['gflops']}")
    print(f"FPS           : {result['fps']} img/s")
    print(f"Throughput    : {result['throughput']} img/s")
    print(f"Latency       : {result['latency_ms_img']} ms/img")
    print(f"Weights       : {result['weights']}")
    print(f"Strict Load   : {result['strict_load']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CS2 segmentation model.")

    parser.add_argument(
        "--variant",
        type=str,
        default="Lite",
        choices=["Lite", "Tiny", "Base"],
        help="模型尺度"
    )
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--strict-load", action="store_true", help="严格加载权重")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--amp", action="store_true", help="启用 AMP 半精度测速")
    parser.add_argument("--out", type=str, default="model_profile.json")

    parser.add_argument(
        "--dims",
        type=str,
        default="",
        help='自定义通道，例如 "24,48,96,192"；不填则使用 preset'
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="",
        help='自定义深度，例如 "2,2,2,"；不填则使用 preset'
    )
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=None,
        help="自定义 drop path rate，不填则使用 preset"
    )

    args = parser.parse_args()

    dims = parse_int_tuple(args.dims)
    depths = parse_int_tuple(args.depths)

    result = build_and_profile(
        model=None,
        variant=args.variant,
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        img_size=args.img_size,
        device=args.device,
        batch_size=args.batch_size,
        weights=args.weights,
        strict_load=args.strict_load,
        warmup=args.warmup,
        iters=args.iters,
        dims=dims,
        depths=depths,
        drop_path_rate=args.drop_path_rate,
        amp=args.amp,
    )

    pretty_print_result(result)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Profile saved to: {args.out}")
