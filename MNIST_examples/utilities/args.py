import argparse
# TODO: from leaf, need to modified to our code.
DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser()
    # load config file from configs
    parser.add_argument("-c", "--config", type=str,
                        default="attack_test", help="Config of hyper-parameters, prefixed with config_mnist_")
    parser.add_argument('-s', "--save", type=bool,
                        default=True, help="save the model params in json in models")
    parser.add_argument('-l', '--load', type=bool,
                        default=False, help="load previous result.")
    parser.add_argument("-ln", "--load_name", type=str,
                        default=None, help="name of previous model")
    parser.add_argument('-r', "--rand_seed", type=int,
                        default=1024, help="random seed for np.random.seed")

    return parser.parse_args()
