"""Offline binaural renderer for a virtual dummy head.

Places an audio file at a chosen position around a virtual head and re-records
it through a head-related transfer function, giving binaural stereo.

Two head models are available: an analytic rigid-sphere model that needs no
data, and a measured Neumann KU100 HRTF from the SADIE II database.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "HeadModel",
    "RenderConfig",
    "render",
    "render_to_file",
    "static_position",
    "keyframes",
    "orbit",
]

# Lazy imports so that ``import asmr_spatial`` does not pull in scipy or
# soundfile unless they are actually used.
_LAZY = {
    "HeadModel": ("hrtf", "HeadModel"),
    "RenderConfig": ("render", "RenderConfig"),
    "render": ("render", "render"),
    "render_to_file": ("render", "render_to_file"),
    "static_position": ("render", "static_position"),
    "keyframes": ("render", "keyframes"),
    "orbit": ("render", "orbit"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{target[0]}", __name__)
    return getattr(module, target[1])
