"""The VOLT point-cloud model with a SceneFun3D affordance head.

VOLT's source is loaded from a local checkout of the official repository at
``VOLT_COMMIT``. Only the files the model needs are imported, so Pointcept's
full dataset stack is not required.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


VOLT_COMMIT = "df41b456406844f5f9b8679d7dafa7e0dd9b33ce"


class _Registry:
    def __init__(self, _name: str):
        self.modules = {}

    def register_module(self, name=None):
        if isinstance(name, type):
            self.modules[name.__name__] = name
            return name

        def decorator(cls):
            self.modules[name or cls.__name__] = cls
            return cls

        return decorator


def _drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return _drop_path(
            x,
            self.drop_prob,
            self.training,
            self.scale_by_keep,
        )


class _Mlp(nn.Module):
    """Parameter-compatible subset of timm.layers.Mlp used by Volt."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        if use_conv:
            raise ValueError("Volt does not request convolutional MLPs")
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias) if isinstance(bias, bool) else bias
        drop = (drop, drop) if isinstance(drop, (float, int)) else drop
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class _LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def install_timm_stubs():
    timm = types.ModuleType("timm")
    timm_layers = types.ModuleType("timm.layers")
    timm_layers.DropPath = _DropPath
    timm_layers.Mlp = _Mlp
    timm_models = types.ModuleType("timm.models")
    vision_transformer = types.ModuleType("timm.models.vision_transformer")
    vision_transformer.LayerScale = _LayerScale
    sys.modules["timm"] = timm
    sys.modules["timm.layers"] = timm_layers
    sys.modules["timm.models"] = timm_models
    sys.modules["timm.models.vision_transformer"] = vision_transformer


def load_python_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_volt_class(volt_root: Path):
    import timm

    if timm.__version__ != "1.0.26":
        raise RuntimeError(
            f"Expected Volt lockfile timm 1.0.26, found {timm.__version__}"
        )

    pointcept = types.ModuleType("pointcept")
    pointcept.__path__ = []
    models = types.ModuleType("pointcept.models")
    models.__path__ = []
    builder = types.ModuleType("pointcept.models.builder")
    builder.MODELS = _Registry("models")
    volt_package = types.ModuleType("pointcept.models.volt")
    volt_package.__path__ = []
    sys.modules["pointcept"] = pointcept
    sys.modules["pointcept.models"] = models
    sys.modules["pointcept.models.builder"] = builder
    sys.modules["pointcept.models.volt"] = volt_package

    load_python_module(
        "pointcept.models.volt.decoder",
        volt_root / "pointcept" / "models" / "volt" / "decoder.py",
    )
    module = load_python_module(
        "pointcept.models.volt.volt_base",
        volt_root / "pointcept" / "models" / "volt" / "volt_base.py",
    )
    return module.Volt


class VoltSceneFun3D(nn.Module):
    def __init__(self, volt_class):
        super().__init__()
        # DefaultSegmentorV2 constructs the fresh target head before its backbone.
        self.seg_heads = nn.ModuleDict({"Default": nn.Linear(256, 10)})
        self.backbone = volt_class(
            in_channels=6,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            init_values=None,
            qk_norm=True,
            drop_path=0.3,
            stride=5,
            kernel_size=5,
            increase_drop_path=True,
            up_mlp_dim=256,
        )

    def forward(self, data):
        return self.seg_heads["Default"](self.backbone(data))


def load_transform_module(volt_root: Path):
    """Load transform.py without importing every Pointcept dataset."""
    pointcept = types.ModuleType("pointcept")
    pointcept.__path__ = []
    utils = types.ModuleType("pointcept.utils")
    utils.__path__ = []
    registry = types.ModuleType("pointcept.utils.registry")
    registry.Registry = _Registry
    sys.modules["pointcept"] = pointcept
    sys.modules["pointcept.utils"] = utils
    sys.modules["pointcept.utils.registry"] = registry

    path = volt_root / "pointcept" / "datasets" / "transform.py"
    spec = importlib.util.spec_from_file_location("volt_dataset_transform", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
