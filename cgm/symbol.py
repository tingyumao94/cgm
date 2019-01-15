import mxnet as mx


def train_cgm_lstm(rnn_num_hidden, time_steps):
    in_data = mx.symbol.Variable('in_features')
    initial_states = mx.symbol.Variable('initial_state')
    gt_data = mx.symbol.Variable('ground_truth')

    lstm_cell = mx.rnn.LSTMCell(num_hidden=rnn_num_hidden)

    # latent state encoder
    h_begin_states = mx.symbol.FullyConnected(data=initial_states, num_hidden=rnn_num_hidden, name='initial_h_encoder')
    c_begin_states = mx.symbol.FullyConnected(data=initial_states, num_hidden=rnn_num_hidden, name='initial_c_encoder')

    # rnn
    outputs, states = lstm_cell.unroll(length=time_steps, inputs=in_data,
                                       begin_state=(h_begin_states, c_begin_states), merge_outputs=True, layout='NTC')

    # output decoder
    prediction = mx.symbol.FullyConnected(data=outputs, num_hidden=1, flatten=False, name='cgm_decoder')

    # loss
    l2_loss = mx.symbol.MakeLoss(0.5 * (prediction - gt_data) ** 2, name='l2_loss')

    return l2_loss


def test_cgm_lstm(rnn_num_hidden, time_steps):
    in_data = mx.symbol.Variable('in_features')
    initial_states = mx.symbol.Variable('initial_state')
    gt_data = mx.symbol.Variable('ground_truth')

    lstm_cell = mx.rnn.LSTMCell(num_hidden=rnn_num_hidden)

    # latent state encoder
    h_begin_states = mx.symbol.FullyConnected(data=initial_states, num_hidden=rnn_num_hidden, name='initial_h_encoder')
    c_begin_states = mx.symbol.FullyConnected(data=initial_states, num_hidden=rnn_num_hidden, name='initial_c_encoder')

    # rnn
    outputs, states = lstm_cell.unroll(length=time_steps, inputs=in_data,
                                       begin_state=(h_begin_states, c_begin_states), merge_outputs=True, layout='NTC')

    # output decoder
    prediction = mx.symbol.FullyConnected(data=outputs, num_hidden=1, flatten=False, name='cgm_decoder')

    # loss
    l2_loss = mx.symbol.MakeLoss(0.5 * (prediction - gt_data) ** 2, name='l2_loss')

    return mx.symbol.Group([prediction, l2_loss])