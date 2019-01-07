import pprint
import mxnet as mx


def attach_prefix(arg_params, aux_params, prefix):
    _arg_params = {}
    _aux_params = {}
    for k in arg_params:
        _arg_params[prefix + k] = arg_params[k]
    for k in aux_params:
        _aux_params[prefix + k] = aux_params[k]
    return _arg_params, _aux_params


def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        elif tp == 'aux':
            aux_params[name] = v
        else:
            raise RuntimeError('Invalid parameter: {}'.format(k))
    return arg_params, aux_params


def summary(sym, **kwargs):
    # arg_shape, output_shape, aux_shape = sym.infer_shape(**kwargs)
    interals = sym.get_internals()
    _, out_shapes, _ = interals.infer_shape(**kwargs)
    shape_info = zip(interals.list_outputs(), out_shapes)
    return shape_info