"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM


# Generator of randomized test sequences
def dataloader_repeat_copy(num_batches,
               batch_size,
               seq_width,
               seq_min_len,
               seq_max_len,
               repeat_min,
               repeat_max):
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
    reps_mean = (repeat_max + repeat_min) / 2
    reps_var = (((repeat_max - repeat_min + 1) ** 2) - 1) / 12
    reps_std = np.sqrt(reps_var)
    def rpt_normalize(reps):
        return (reps - reps_mean) / reps_std

    for batch_num in range(num_batches):

        # All batches have the same sequence length and number of reps
        seq_len = random.randint(seq_min_len, seq_max_len)
        reps = random.randint(repeat_min, repeat_max)

        # Generate the sequence
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes 2 additional channels, for end-of-sequence and num-reps
        inp = torch.zeros(seq_len + 2, batch_size, seq_width + 2)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0
        inp[seq_len+1, :, seq_width+1] = rpt_normalize(reps)

        # The output contain the repeated sequence + end marker
        outp = torch.zeros(seq_len * reps + 1, batch_size, seq_width + 1)
        outp[:seq_len * reps, :, :seq_width] = seq.clone().repeat(reps, 1, 1)
        outp[seq_len * reps, :, seq_width] = 1.0 # End marker

        yield batch_num+1, inp.float(), outp.float()

bcd_excess_3 = {
    0 : np.array([0, 0, 1, 1]),
    1 : np.array([0, 1, 0, 0]),
    2 : np.array([0, 1, 0, 1]),
    3 : np.array([0, 1, 1, 0]),
    4 : np.array([0, 1, 1, 1]),
    5 : np.array([1, 0, 0, 0]),
    6 : np.array([1, 0, 0, 1]),
    7 : np.array([1, 0, 1, 0]),
    8 : np.array([1, 0, 1, 1]),
    9 : np.array([1, 1, 0, 0]),
}

bcd_excess_3_str = {
    '0' : np.array([0, 0, 1, 1]),
    '1' : np.array([0, 1, 0, 0]),
    '2' : np.array([0, 1, 0, 1]),
    '3' : np.array([0, 1, 1, 0]),
    '4' : np.array([0, 1, 1, 1]),
    '5' : np.array([1, 0, 0, 0]),
    '6' : np.array([1, 0, 0, 1]),
    '7' : np.array([1, 0, 1, 0]),
    '8' : np.array([1, 0, 1, 1]),
    '9' : np.array([1, 1, 0, 0]),
    '0011' : '0',
    '0100' : '1',
    '0101' : '2',
    '0110' : '3',
    '0111' : '4',
    '1000' : '5',
    '1001' : '6',
    '1010' : '7',
    '1011' : '8',
    '1100' : '9',
}


def dataloader0(num_batches,
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

    for batch_num in range(num_batches):

        # All batches have the same sequence length and number of reps
        seq_len = random.randint(seq_min_len, seq_max_len)
        seq_width = 4

        # Generate the sequence
        # seq = np.zeros((seq_len, batch_size, 5))
        # seq = torch.from_numpy(seq)

        inp = np.zeros((seq_len + 1, batch_size, seq_width + 1))
        outp = np.zeros((seq_len + 1 , batch_size, seq_width + 1))
        inp_numbers = []
        outp_numbers = []

        for i in range(batch_size):
            inp_number = np.random.randint(10 ** seq_len)
            outp_number = inp_number + 1
            inp_number = str(inp_number).zfill(seq_len)
            outp_number = str(outp_number).zfill(seq_len)
            inp_numbers.append(inp_number)
            outp_numbers.append(outp_number)

        for i in range(seq_len):
            for j in range(batch_size):
                inp_digit = bcd_excess_3[int(inp_numbers[j][i])]
                outp_digit = bcd_excess_3[int(outp_numbers[j][i])]
                inp[i, j, :seq_width] = inp_digit
                outp[i, j, :seq_width] = outp_digit


        inp[seq_len, :, seq_width] = 1.0
        outp[seq_len, :, seq_width] = 1.0

        inp = torch.from_numpy(inp)
        outp = torch.from_numpy(outp)

        yield batch_num+1, inp.float(), outp.float()



def inc(x):
    return x + 1

def dec(x):
    return x - 1

def mul(x):
    return x * 2

def mod(x):
    return x % 2

funcs = [inc, dec, mul, mod]



def dataloader(num_batches,
               batch_size,
               seq_min_len,
               seq_max_len, test=False):
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

    for batch_num in range(num_batches):
        seq_len = random.randint(seq_min_len, seq_max_len)
        seq_width = 4

        task1 = 0
        task2 = 2

        if test == False:
            while task1 == 0 and task2 == 2:
                task1 = random.randint(0, 2)
                task2 = random.randint(0, 2)

        inp = np.zeros((seq_len + 1, batch_size, seq_width + 1))
        outp = np.zeros((seq_len + 1 , batch_size, seq_width + 1))
        inp_numbers = []
        outp_numbers = []

        for i in range(batch_size):
            inp_number = np.random.randint(2, 10 ** seq_len - 1)
            outp_number = funcs[task2](funcs[task1](inp_number))
            inp_number = str(inp_number).zfill(seq_len)
            outp_number = str(outp_number).zfill(seq_len)
            inp_numbers.append(inp_number)
            outp_numbers.append(outp_number)

        for i in range(seq_len):
            for j in range(batch_size):
                inp_digit = bcd_excess_3[int(inp_numbers[j][i])]
                outp_digit = bcd_excess_3[int(outp_numbers[j][i])]
                inp[i, j, :seq_width] = inp_digit
                outp[i, j, :seq_width] = outp_digit


        inp[seq_len, :, seq_width] = 1.0
        outp[seq_len, :, seq_width] = 1.0

        inp = torch.from_numpy(inp)
        outp = torch.from_numpy(outp)
        ctrl = np.array([[bcd_excess_3[task1], bcd_excess_3[task2]]])
        ctrl = torch.from_numpy(ctrl)


        yield batch_num+1, inp.float(), ctrl.float(), outp.float()








@attrs
class Seq2BCDTaskParams(object):
    name = attrib(default="seq2-bcd")
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
    num_batches = attrib(default=500000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)


@attrs
class Seq2BCDModelTraining(object):
    params = attrib(default=Factory(Seq2BCDTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # See dataloader documentation
        net = EncapsulatedNTM(5, 5,
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


    def input_human(self, net, number_str):
        seq_width = 4
        number_str = number_str.split(r' ')
        # print(number_str)
        x_str, ctrl_str = number_str[0], number_str[1]
        seq_len = len(x_str)
        ctrl_len = len(ctrl_str)
        inp = np.zeros((seq_len + 1, 1, seq_width + 1))
        ctrl = np.zeros((ctrl_len, 1, seq_width))
        outp = np.zeros((seq_len + 1, 1, seq_width + 1))
        net.init_sequence(1)
        # net = net.float()
        
        for i in range(seq_len):
            inp[i, 0, :seq_width] = bcd_excess_3_str[x_str[i]]
        for i in range(ctrl_len):
            ctrl[i, 0, :seq_width] = bcd_excess_3_str[ctrl_str[i]]

        inp[seq_len, :, seq_width] = 1.0
        inp = torch.from_numpy(inp).float()
        ctrl = torch.from_numpy(ctrl).float()
        outp = torch.from_numpy(outp).float()

        net.seeAndListen(inp, ctrl)
        # for i in range(seq_len + 1):
        #     o, state = net(inp[i])
        for i in range(seq_len):
            outp[i], state = net()
        outp_bin = outp.clone().data
        outp_bin.apply_(lambda x: 0 if x < 0.5 else 1)

        outp_human = ''
        for i in range(seq_len + 1):
            digit_bcd = ''
            for j in range(seq_width):
                digit_bcd = digit_bcd + str(int(outp_bin[i, 0, j].item()))
            # print(digit_bcd)
            # input()
            if digit_bcd not in bcd_excess_3_str:
                break
            outp_human = outp_human + bcd_excess_3_str[digit_bcd]

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