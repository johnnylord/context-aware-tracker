import sys

from .sort import SORT
from .deepsort import DeepSORT
from .cat import CAT

def get_tracker_cls(name):
    return getattr(sys.modules[__name__], name)
