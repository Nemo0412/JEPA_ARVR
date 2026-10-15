from fvcore.common.registry import Registry

from scalant.config import Config


ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoders.

The registered object will be called with `obj(**kwargs)`.
The call should return a `torch.nn.Module` object.
"""


def build_encoder(cfg: Config):
    """
    Build a model, defined by `cfg.MODEL.MODEL_CLASS`.
    """
    model = ENCODER_REGISTRY.get(cfg.MODEL.ENCODER_CLASS)(cfg)
    return model
