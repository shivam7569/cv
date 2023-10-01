import argparse

from global_params import Global


def get_parser():
    parser = argparse.ArgumentParser(
        description="LeNet model processed"
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="model to use"
    )

    return parser
