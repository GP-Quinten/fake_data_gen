import argparse


def preproc_parser():
    parser = argparse.ArgumentParser(
        description="SDG PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument(
        "--diag-preproc",
        default="binary",
        choices=["binary", "count"],
        help="Diag preprocessing",
    )

    return parser


def training_parser():
    parser = argparse.ArgumentParser(
        description="SDG PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument("--dataset-path", required=True, help="Path to training dataset")

    parser.add_argument("--dataset-filename", required=True, help="Name of the training dataset")

    parser.add_argument(
        "--n-to-sample",
        type=int,
        help="Size of the synthetic dataset to create",
    )

    parser.add_argument(
        "--n-to-eval", type=int, help="Number of rows to sample for evaluation"
    )

    parser.add_argument(
        "--discriminator-decay",
        default=1e-6,
        type=int,
        help="Discriminator weight decay for the Adam Optimizer",
    )

    parser.add_argument(
        "--generator-decay",
        default=1e-6,
        type=int,
        help="Generator weight decay for the Adam Optimizer",
    )

    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs",
    )

    parser.add_argument(
        "--batch-size", default=500, type=int, help="Batch size for training"
    )

    parser.add_argument(
        "--learning-rate",
        default=[2e-4],
        nargs="+",
        type=float,
        help="List of discriminator and generator learning rate to test",
    )

    parser.add_argument(
        "--discrete-columns",
        nargs="+",
        default=[],
        help="Name of the discrete columns of the dataset to use in the conditional vector",
    )

    parser.add_argument(
        "--test-dataset",
        default=False,
        help="Use test dataset to run model",
    )

    parser.add_argument(
        "--mlflow-uri",
        required=True,
        help="Path to store mlruns",
    )

    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name for mlflow",
    )

    parser.add_argument(
        "--col-to-plot", default="ALL", nargs="+", help="Names of the columns to plot after training"
    )

    return parser
