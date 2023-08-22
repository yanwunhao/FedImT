import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')
    parser.add_argument('--num_users', type=int, default=50, help='number of federated participants')
    parser.add_argument('--frac', type=float, default=0.3, help="the fraction of clients: C")
    parser.add_argument('--rounds', type=int, default=50, help="total rounds")
    parser.add_argument('--local_ep', type=int, default=5, help="local epoch for each round")

    # model arguments
    parser.add_argument('--model', type=str, default="cnn", help="model name")
    parser.add_argument('--bs', type=int, default=16, help="batch size")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--loss', type=str, default="balanced-cross-entropy", help="loss function for local update")


    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')

    args = parser.parse_args()
    return args