"""    
tshock:private
author:@elvinlife
time:since 2018/2/18
"""
import collections

def _make_tuple(obj):
    if isinstance(obj, collections.Iterable):
        return tuple(obj)
    elif obj == None:
        return ()
    else:
        return obj,