import numpy as np

__all__ = [
    'xyah_to_tlbr', 'xyah_to_tlwh',
    'tlbr_to_xyah', 'tlbr_to_tlwh',
    'tlwh_to_xyah', 'tlwh_to_tlbr'
    ]

def xyah_to_tlbr(xyah):
    # Type checking
    if isinstance(xyah, np.ndarray):
        xyah = xyah.tolist()
    elif isinstance(xyah, list) or isinstance(xyah, tuple):
        xyah = xyah
    else:
        raise Exception("Cannot handle data of type {}".format(type(xyah)))

    # Conversion
    cx, cy, a, h = tuple(xyah)
    tl_x, tl_y = cx-(a*h/2), cy-(h/2)
    br_x, br_y = cx+(a*h/2), cy+(h/2)

    return tl_x, tl_y, br_x, br_y

def xyah_to_tlwh(xyah):
    # Type checking
    if isinstance(xyah, np.ndarray):
        xyah = xyah.tolist()
    elif isinstance(xyah, list) or isinstance(xyah, tuple):
        xyah = xyah
    else:
        raise Exception("Cannot handle data of type {}".format(type(xyah)))

    # Conversion
    cx, cy, a, h = tuple(xyah)
    tl_x, tl_y = cx-(a*h/2), cy-(h/2)
    w, h = a*h, h

    return tl_x, tl_y, w, h

def tlbr_to_xyah(tlbr):
    # Type checking
    if isinstance(tlbr, np.ndarray):
        tlbr = tlbr.tolist()
    elif isinstance(tlbr, list) or isinstance(tlbr, tuple):
        tlbr = tlbr
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlbr)))

    # Conversion
    tl_x, tl_y, br_x, br_y = tuple(tlbr)
    cx, cy = (tl_x+br_x)/2, (tl_y+br_y)/2
    a, h = (br_x-tl_x)/(br_y-tl_y), (br_y-tl_y)

    return cx, cy, a, h

def tlbr_to_tlwh(tlbr):
    # Type checking
    if isinstance(tlbr, np.ndarray):
        tlbr = tlbr.tolist()
    elif isinstance(tlbr, list) or isinstance(tlbr, tuple):
        tlbr = tlbr
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlbr)))

    # Conversion
    tl_x, tl_y, br_x, br_y = tuple(tlbr)
    w, h = (br_x-tl_x), (br_y-tl_y)

    return tl_x, tl_y, w, h

def tlwh_to_tlbr(tlwh):
    # Type checking
    if isinstance(tlwh, np.ndarray):
        tlwh = tlwh.tolist()
    elif isinstance(tlwh, list) or isinstance(tlwh, tuple):
        tlwh = tlwh
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlwh)))

    # Conversion
    tl_x, tl_y, w, h = tuple(tlwh)
    return tl_x, tl_y, tl_x+w, tl_y+h

def tlwh_to_xyah(tlwh):
    # Type checking
    if isinstance(tlwh, np.ndarray):
        tlwh = tlwh.tolist()
    elif isinstance(tlwh, list) or isinstance(tlwh, tuple):
        tlwh = tlwh
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlwh)))

    # Conversion
    tl_x, tl_y, w, h = tuple(tlwh)
    cx, cy = tl_x+(w/2), tl_y+(h/2)
    a = w/h
    return cx, cy, a, h
