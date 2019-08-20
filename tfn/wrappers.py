

def inputs_to_dict(call):
    """
    For internal use only. Converts a list of input tensors to a dictionary based on their representation index.
    """
    def get_input_dict(*args, **kwargs):
        if len(args) > 1:
            instance, inputs = args[0], args[1]
        else:
            instance = None
            inputs = args[0]
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
        if instance is None:
            return call(input_dict, **kwargs)
        else:
            return call(instance, input_dict, **kwargs)

    return get_input_dict


def shapes_to_dict(build):
    """
    For internal use only. Converts a list of input shapes to a dictionary based on their representation index.
    """
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
