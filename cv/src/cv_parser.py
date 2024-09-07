import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="LeNet model processed"
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="model to use"
    )

    parser.add_argument(
        "--gpu-devices",
        required=False,
        help="gpu devices to use"
    )

    return parser
