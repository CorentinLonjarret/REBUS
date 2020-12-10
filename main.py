# Main driver file
import dataset
import commons
import os
import REBUS
import REBUS_simple
import REBUS_ST
import REBUS_ST_simple
import REBUS_LT
import sys
import argparse
import random
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    args = commons.parse_args()

    d = dataset.Dataset(args)

    if args.model == 'REBUS_simple':
        model = REBUS_simple.REBUS_simple(d, args)
    elif args.model == 'REBUS':
        model = REBUS.REBUS(d, args)
    elif args.model == 'REBUS_reg':
        model = REBUS_reg.REBUS_reg(d, args)
    elif args.model == 'REBUS_ST':
        model = REBUS_ST.REBUS_ST(d, args)
    elif args.model == 'REBUS_ST_simple':
        model = REBUS_ST_simple.REBUS_ST_simple(d, args)
    elif args.model == 'REBUS_LT':
        model = REBUS_LT.REBUS_LT(d, args)

    results_model = model.train()

    commons.save_results(model, results_model)

    print('')
    print(args)
