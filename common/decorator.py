import functools


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func) 
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper
