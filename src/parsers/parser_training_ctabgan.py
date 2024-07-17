import argparse


def ctabgan_parser():
    parser = argparse.ArgumentParser(
        description="CTABGAN", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument(
        "--mlflow-uri",
        default="../mlruns",
        help="Path to store mlruns",
    )

    parser.add_argument(
        "--exp-name", required=True, help="Experiment name for mlflow", default="Test"
    )

    parser.add_argument(
        "--epochs",
        default=[1],
        nargs="+",
        type=int,
        help="Epochs of training",
    )

    parser.add_argument(
        "--batch-size",
        default=[500],
        type=int,
        nargs="+",
        help="Batch size",
    )

    parser.add_argument(
        "--lr",
        default=[2e-4],
        type=float,
        nargs="+",
        help="Learning rate of Discriminator and Generator",
    )

    parser.add_argument(
        "--l2scale",
        default=[1e-5],
        type=float,
        nargs="+",
        help="Decay (L2 penalization) of Discriminator and Generator",
    )

    parser.add_argument(
        "--save-synth-data",
        default=False,
        type=bool,
        help="If true then synthetic data will be saved. ",
    )

    return parser
