"""
Pytest configuration and shared fixtures.

Heavy dependencies (OpenCASCADE, wandb, diffusers) are not installed in the
lightweight CI environment, so we stub them out before any import happens.
"""

import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    """Return a simple stub module that silently accepts any attribute access."""

    class _StubModule(types.ModuleType):
        def __getattr__(self, item):
            return _StubModule(f"{self.__name__}.{item}")

        def __call__(self, *args, **kwargs):
            return None

    mod = _StubModule(name)
    return mod


_STUBS = [
    "OCC",
    "OCC.Core",
    "OCC.Core.gp",
    "OCC.Core.TColgp",
    "OCC.Core.GeomAPI",
    "OCC.Core.GeomAbs",
    "OCC.Core.BRepBuilderAPI",
    "OCC.Extend",
    "OCC.Extend.TopologyUtils",
    "OCC.Core.ShapeFix",
    "OCC.Core.ShapeAnalysis",
    "occwl",
    "occwl.io",
    "wandb",
    "diffusers",
    "diffusers.configuration_utils",
    "diffusers.utils",
    "diffusers.utils.accelerate_utils",
    "diffusers.models",
    "diffusers.models.attention_processor",
    "diffusers.models.modeling_utils",
    "diffusers.models.autoencoders",
    "diffusers.models.autoencoders.vae",
    "diffusers.models.unets",
    "diffusers.models.unets.unet_1d_blocks",
    "transformers",
    "trimesh",
    "plyfile",
    "sklearn",
    "mpl_toolkits",
    "mpl_toolkits.mplot3d",
    "mpl_toolkits.mplot3d.art3d",
]

for _name in _STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = _make_stub(_name)
