#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

import argparse
import json
import logging
import time
import random
import re
import sys

import attr
import argcomplete
import torch
import numpy as np
from Levenshtein import *


LOGGER = logging.getLogger(__name__)


from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.addonetask import AddOneTaskModelTraining, AddOneTaskParams
import tasks.addonetask_onehot as aoo
import tasks.addonetask_onehot_reverse as aoo_r
from tasks.tellparitytask import TellParityTaskModelTraining, TellParityTaskParams
from tasks.reversetask import ReverseTaskModelTraining, ReverseTaskParams
from tasks.decreasetask import DecreaseTaskModelTraining, DecreaseTaskParams
from tasks.mult2task import MultTwoTaskModelTraining, MultTwoTaskParams
from tasks.leftshifttask import LeftShiftTaskModelTraining, LeftShiftTaskParams
from tasks.rightshifttask import RightShiftTaskModelTraining, RightShiftTaskParams
from tasks.all_bcd import AllBCDModelTraining, AllBCDTaskParams

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams),
    'add-one': (AddOneTaskModelTraining, AddOneTaskParams),
    'add-one-onehot': (aoo.AddOneTaskModelTraining, aoo.AddOneTaskParams),
    'add-one-onehot-reverse': (aoo_r.AddOneTaskModelTraining, aoo_r.AddOneTaskParams),
    'tell-parity': (TellParityTaskModelTraining, TellParityTaskParams),
    'reverse': (ReverseTaskModelTraining, ReverseTaskParams),
    'decrease': (DecreaseTaskModelTraining, DecreaseTaskParams),
    'multi-two': (MultTwoTaskModelTraining, MultTwoTaskParams),
    'left-shift': (LeftShiftTaskModelTraining, LeftShiftTaskParams),
    'right-shift': (RightShiftTaskModelTraining, RightShiftTaskParams),
    'all-bcd': (AllBCDModelTraining, AllBCDTaskParams),
}


# Default values for program arguments
RANDOM_SEED = 1000
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def save_checkpoint(net, name, args, batch_num, losses, costs, seq_lengths):
    progress_clean()

    basename = "{}/{}-{}-batch-{}".format(args.checkpoint_path, name, args.seed, batch_num)
    model_fname = basename + ".model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def train_batch(net, criterion, optimizer, X, Y):
    """Trains a single batch."""
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        net(X[i])

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], _ = net()

    loss = criterion(y_out, Y)
    loss.backward()
    clip_grads(net)
    optimizer.step()

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    return loss.item(), cost.item() / batch_size


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

    return result


def train_model(model, args):
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size

    LOGGER.info("Training model for %d batches (batch_size=%d)...",
                num_batches, batch_size)

    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()

    for batch_num, x, y in model.dataloader:
        loss, cost = train_batch(model.net, model.criterion, model.optimizer, x, y)
        losses += [loss]
        costs += [cost]
        seq_lengths += [y.size(0)]

        # Update the progress bar
        progress_bar(batch_num, args.report_interval, loss)

        # Report
        if batch_num % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            mean_cost = np.array(costs[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
            progress_clean()
            LOGGER.info("Batch %d Loss: %.6f Cost: %.2f Time: %d ms/sequence",
                        batch_num, mean_loss, mean_cost, mean_time)
            start_ms = get_ms()

        # Checkpoint
        if (args.checkpoint_interval != 0) and (batch_num % args.checkpoint_interval == 0):
            save_checkpoint(model.net, model.params.name, args,
                            batch_num, losses, costs, seq_lengths)

    LOGGER.info("Done training.")


def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()), default='all-bcd',
                        help="Choose the task to train (default: copy)")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL,
                        help="Checkpoint interval (default: {}). "
                             "Use 0 to disable checkpointing".format(CHECKPOINT_INTERVAL))
    parser.add_argument('--checkpoint-path', action='store', default='./',
                        help="Path for saving checkpoint data (default: './')")
    parser.add_argument('--report-interval', type=int, default=REPORT_INTERVAL,
                        help="Reporting interval")
    parser.add_argument('--checkpoint_file', type=str, default='',
                        help="Checkpoint file to load")

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path.rstrip('/')

    return args


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params

def init_model(args):
    LOGGER.info("Training for the **%s** task", args.task)

    model_cls, params_cls = TASKS[args.task]
    params = params_cls()
    params = update_model_params(params, args.param)

    LOGGER.info(params)

    model = model_cls(params=params)
    return model


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def main():
    init_logging()

    # Initialize arguments
    args = init_arguments()

    # Initialize random
    init_seed(args.seed)

    # Initialize the model
    model = init_model(args)
    LOGGER.info("Total number of parameters: %d", model.net.calculate_num_params())
    net = model.net
    net.load_state_dict(torch.load(args.checkpoint_file))

    # while True:
    #     inp_str = input()
    #     output_str_pred = model.input_human(net, inp_str)
    #     print(output_str_pred)

    # inp_num = '1'
    # while inp_num:
    #     print('please input a number: ', end='')
    #     inp_str = input()
    #     output_human = model.input_human(net, inp_str)
    #     print('output:', output_human)

    # same seq_len as training dataset
    error = 0
    for ctrl in range(4, 5):
        error = 0
        for i in range(1000):
            seq_len = random.randint(1, 9)
            if ctrl == 0:
                inp_number = np.random.randint(10 ** seq_len - 1)
                outp_number = inp_number + 1
            elif ctrl == 5:
                inp_number = np.random.randint(10 ** seq_len)
                outp_number = int(str(inp_number)[::-1])
            elif ctrl == 2:
                inp_number = np.random.randint(10 ** seq_len - 1)
                inp_number += 1
                outp_number = inp_number - 1
            elif ctrl == 6:
                inp_number = np.random.randint(10 ** seq_len - 1)
                outp_number = inp_number * 2
            elif ctrl == 3:
                inp_number = np.random.randint(10 ** seq_len - 1)
                outp_number = inp_number * 10
            elif ctrl == 4:
                inp_number = np.random.randint(10 ** seq_len - 1)
                outp_number = (inp_number + 5) // 10
            elif ctrl == 1:
                inp_number = np.random.randint(10 ** seq_len - 1)
                outp_number = inp_number % 2

            inp_str = str(inp_number).zfill(seq_len) + ' ' + str(ctrl)
            outp_str_gt = str(outp_number).zfill(seq_len)

            # print(inp_str, outp_str_gt)
            output_str_pred = model.input_human(net, inp_str)
            output_str_pred = output_str_pred.lstrip(r'0')
            outp_str_gt = outp_str_gt.lstrip(r'0')
            lev = distance(output_str_pred, outp_str_gt)
            print(lev, outp_str_gt, output_str_pred, i)
            # if ctrl == 1:
            #     print(outp_str_gt, output_str_pred, lev)
            #     input()
            # print()
            error += lev

        print(ctrl, "Test dataset1 distance ", error / 1000)

    # longer seq_len as training dataset
    error = 0
    for i in range(1000):
        seq_len = random.randint(1, 15)
        if args.task == "add-one":
            inp_number = np.random.randint(10 ** seq_len - 1, dtype=np.int64)
            outp_number = inp_number + 1
        elif args.task == "reverse":
            inp_number = np.random.randint(10 ** seq_len, dtype=np.int64)
            outp_number = int(str(inp_number)[::-1])
        elif args.task == "decrease":
            inp_number = np.random.randint(10 ** seq_len - 1, dtype=np.int64)
            inp_number += 1
            outp_number = inp_number - 1
        elif args.task == "multi-two":
            inp_number = np.random.randint(10 ** seq_len - 1, dtype=np.int64)
            outp_number = inp_number * 2
        elif args.task == "left-shift":
            inp_number = np.random.randint(10 ** seq_len - 1, dtype=np.int64)
            outp_number = inp_number * 10
        elif args.task == "right-shift":
            inp_number = np.random.randint(10 ** seq_len - 1, dtype=np.int64)
            outp_number = inp_number // 10
            if inp_number % 10 >= 5:
                outp_number += 1

        inp_str = str(inp_number).zfill(seq_len)
        outp_str_gt = str(outp_number).zfill(seq_len)

        output_str_pred = model.input_human(net, inp_str)
        # print(outp_str_gt, output_str_pred)
        lev = distance(output_str_pred, outp_str_gt)
        error += lev
        # print(i, error / (i + 1))

    print("Test dataset2 distance ", error / 1000)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
