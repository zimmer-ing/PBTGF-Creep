#from .FractionalDamper import FractionalDamperModel as FractionalDamper
from .FractionalDamper_Multiprocessing import FractionalDamperModel as FractionalDamper
from .prony_maxwell import PronyModel as PronyMaxwell

MODEL_REGISTRY = {
    "FractionalDamper": FractionalDamper,
    "PronyMaxwell": PronyMaxwell,
}

__all__ = [MODEL_REGISTRY.keys()]