import argparse

from global_params import Global


def get_parser():
    parser = argparse.ArgumentParser(
        description="LeNet model processed"
    )

    parser.add_argument(
        "--config-file",
        default="/media/drive6/hqh2kor/projects/Computer_Vision/configs/proces_configs/example_config.yml",
        metavar="FILE",
        help="path to LeNet process config file"
    )

    return parser
