"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM



onehot = {
    0 : np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    1 : np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    2 : np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    3 : np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    4 : np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    5 : np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    6 : np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    7 : np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    8 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    9 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
}

bcd_excess_3_str = {
    '0' : np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    '1' : np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    '2' : np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    '3' : np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    '4' : np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    '5' : np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    '6' : np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    '7' : np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    '8' : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    '9' : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    '1000000000' : '0',
    '0100000000' : '1',
    '0010000000' : '2',
    '0001000000' : '3',
    '0000100000' : '4',
    '0000010000' : '5',
    '0000001000' : '6',
    '0000000100' : '7',
    '0000000010' : '8',
    '0000000001' : '9',
}


def dataloader(num_batches,
               batch_size,
               seq_min_len,
               seq_max_len):
    """Generator of random sequences for the repeat copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is between `min_len` to `max_len`

    :param num_batches: Total number of batches to generate.
    :param batch_size: Batch size.
    :param seq_width: The width of each item in the sequence.
    :param seq_min_len: Sequence minimum length.
    :param seq_max_len: Sequence maximum length.
    :param repeat_min: Minimum repeatitions.
    :param repeat_max: Maximum repeatitions.

    NOTE: The input width is `seq_width + 2`. One additional input
    is used for the delimiter, and one for the number of repetitions.
    The output width is `seq_width` + 1, the additional input is used
    by the network to generate an end-marker, so we can be sure the
    network counted correctly.
    """
    # Some normalization constants
    if batch_size == 1:

        for batch_num in range(num_batches):

            # All batches have the same sequence length and number of reps
            seq_len = random.randint(seq_min_len, seq_max_len)
            seq_width = 10

            # Generate the sequence
            # seq = np.zeros((seq_len, batch_size, 5))
            # seq = torch.from_numpy(seq)

            inp_numbers = []
            outp_numbers = []

            for i in range(batch_size):
                inp_number = np.random.randint(10 ** seq_len)
                outp_number = inp_number + 1
                inp_number = str(inp_number).zfill(seq_len)
                outp_number = str(outp_number).zfill(seq_len)
                inp_numbers.append(inp_number)
                outp_numbers.append(outp_number)


            inp = np.zeros((seq_len + 1, batch_size, seq_width + 1))
            outp = np.zeros((len(outp_numbers[0]) + 1 , batch_size, seq_width + 1))

            for i in range(seq_len):
                for j in range(batch_size):
                    inp_digit = onehot[int(inp_numbers[j][- i - 1])]
                    inp[i, j, :seq_width] = inp_digit

            for i in range(len(outp_numbers[0])):
                for j in range(batch_size):
                    outp_digit = onehot[int(outp_numbers[j][- i - 1])]
                    outp[i, j, :seq_width] = outp_digit


            inp[seq_len, :, seq_width] = 1.0
            outp[len(outp_numbers[0]), :, seq_width] = 1.0

            inp = torch.from_numpy(inp)
            outp = torch.from_numpy(outp)

            yield batch_num+1, inp.float(), outp.float()








@attrs
class AddOneTaskParams(object):
    name = attrib(default="add-one-onehot-task-reverse")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_heads = attrib(default=1, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1, converter=int)
    sequence_max_len = attrib(default=10, converter=int)
    repeat_min = attrib(default=1, converter=int)
    repeat_max = attrib(default=10, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=250000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)


@attrs
class AddOneTaskModelTraining(object):
    params = attrib(default=Factory(AddOneTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # See dataloader documentation
        net = EncapsulatedNTM(11, 11,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_min_len, self.params.sequence_max_len)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)


    def onehot_to_int(self, torch_array):
        return int(torch.argmax(torch_array).item())


    def input_human(self, net, number_str):
        seq_width = 10
        seq_len = len(number_str)

        inp = np.zeros((seq_len + 1, 1, seq_width + 1))
        net.init_sequence(1)
        # net = net.float()
        
        for i in range(seq_len):
            inp[i, 0, :seq_width] = bcd_excess_3_str[number_str[i]]

        inp[seq_len, :, seq_width] = 1.0
        inp = torch.from_numpy(inp).float()


        for i in range(seq_len + 1):
            o, state = net(inp[i])


        outp_human = ''
        outp_tmp, state = net()

        while(self.onehot_to_int(outp_tmp) != 10 and len(outp_human) < 30):
            outp_human = outp_human + str(self.onehot_to_int(outp_tmp))
            outp_tmp, state = net()


        # outp = np.zeros((seq_len + 1, 1, seq_width + 1))
        # outp = torch.from_numpy(outp).float()
        # for i in range(seq_len):
        #     outp[i], state = net()
        # outp_bin = outp.clone().data
        # outp_bin.apply_(lambda x: 0 if x < 0.5 else 1)

        # outp_human = ''
        # for i in range(seq_len):
        #     digit_bcd = ''
        #     for j in range(seq_width):
        #         digit_bcd = digit_bcd + str(int(outp_bin[i, 0, j].item()))
        #     # print(digit_bcd)
        #     # input()
        #     outp_human = outp_human + bcd_excess_3_str[digit_bcd]

        return outp_human





def evaluate(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }