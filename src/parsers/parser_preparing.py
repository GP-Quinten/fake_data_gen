import argparse


def parser_preparing():
    parser = argparse.ArgumentParser(
        description="SDG PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    # to remove later
    parser.add_argument(
        "--diag-preproc",
        default="binary",
        choices=["binary", "count"],
        help="Diag preprocessing",
    )

    parser.add_argument(
        "--remove-nan", default=False, help="Handle variables with missing values"
    )

    return parser
