"""
SWORD Lint Checks

Auto-imports all check modules to register checks with the registry.
"""

# Import all check modules to register their checks
from . import topology
from . import attributes
from . import geometry
from . import classification
from . import v17c

__all__ = [
    "topology",
    "attributes",
    "geometry",
    "classification",
    "v17c",
]
