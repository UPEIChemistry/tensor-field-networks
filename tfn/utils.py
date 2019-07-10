"""
Module containing basic utility functions for the TFN layers
"""
from tensorflow.python.keras import backend as K


# TODO: If add support for more L values, remember to support unpacking here!
def inputs_to_dict(call):

    def get_input_dict(*args, **kwargs):
        instance, inputs = args[0], args[1]
        if not isinstance(inputs, list):
            inputs = [inputs]
        input_dict = {0: [], 1: []}
        for tensor in inputs:
            if int(tensor.shape[-1]) == 1:
                input_dict[0].append(tensor)
            else:
                input_dict[1].append(tensor)

        # Pull out keys with empty lists
        input_dict = {k: v for k, v in input_dict.items() if len(v) != 0}
        return call(instance, input_dict, **kwargs)

    return get_input_dict


def wrap_shape_dict(build):

    def get_shape_dict(*args, **kwargs):
        instance, shapes = args[0], args[1]
        if not isinstance(shapes, list):
            shapes = [shapes]
        shape_dict = {0: [], 1: []}
        for shape in shapes:
            if shape[-1] == 1:
                shape_dict[0].append(shape)
            else:
                shape_dict[1].append(shape)

        # Pull out keys with empty lists
        shape_dict = {k: v for k, v in shape_dict.items() if len(v) != 0}
        build(instance, shape_dict, **kwargs)

    return get_shape_dict


def get_l_shape(i):

    return int((i - 1) / 2)


def shifted_softplus(x):
    """
    From the SchNet code, converted to Keras backend.

    Softplus nonlinearity shifted by -log(2) such that shifted_softplus(0.) = 0.
    y = log(0.5e^x + 0.5)
    """

    y = K.switch(
        x < 14.,
        K.softplus(K.switch(x < 14., x, K.zeros_like(x))),
        x
    )

    return y - K.log(2.)
