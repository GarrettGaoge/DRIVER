import argparse

# Some arguments

def parse_args():
    parser = argparse.ArgumentParser(description='DRIVER')
    parser.add_argument('--train_file', nargs='?',default='./data/interactions.csv',
                        help='The file path of training data.')
    parser.add_argument('--embd_size', type=int, default=128,
                        help='The embedding size.')
    parser.add_argument('--gnn_layers', type=int, default=1,
                        help='Number of GNN layers')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs')
    parser.add_argument('--time_unit', type=float, default=1,
                        help='Interval(days) to do GNN')
    parser.add_argument('--test_epoch', type=int, default=59,
                        help='Which epoch do you want to test')

    return parser.parse_args()