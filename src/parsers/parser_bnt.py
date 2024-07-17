import argparse


def simple_parser():
    parser = argparse.ArgumentParser(
        description="MY AWESOME PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    return parser


def sampling_parser():
    parser = argparse.ArgumentParser(
        description="MY AWESOME PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument(
        "--n_sample",
        default=1000,
        help="Number of samples",
    )
    return parser