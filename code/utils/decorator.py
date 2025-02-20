import functools
import tensorflow as tf


def doublewrap(function):
    """
    A decorator's decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

"""
lazy_property is not only a decorator, but also a decorated function. The time 
when lazy_property is invoked, its decorator is invoked first to replace 
lazy_property
"""
@doublewrap
def lazy_property(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope().
    If this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attr = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attr):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attr, function(self))
        return getattr(self, attr)
    

    return decorator




@doublewrap
def lazy_method(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope().
    If this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attr = '_cache_' + function.__name__
    name = scope or function.__name__


    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        if not hasattr(self, attr):
            #import pdb; pdb.set_trace()
            with tf.variable_scope(name):
                setattr(self, attr, function(self, *args, **kwargs))
        return getattr(self, attr)
    

    return decorator


@doublewrap
def lazy_method_no_scope(function, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    """
    attr = '_cache_' + function.__name__


    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        if not hasattr(self, attr):
            setattr(self, attr, function(self, *args, **kwargs))
        return getattr(self, attr)
    

    return decorator