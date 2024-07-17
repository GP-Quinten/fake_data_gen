import argparse


def fidelity_parser():
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
        "--synth-dataset", required=True, help="Name of fake dataset to evaluate"
    )

    parser.add_argument(
        "--real-dataset", required=True, help="Name of real dataset to evaluate"
    )

    parser.add_argument(
        "--category-threshold",
        type=int,
        default=3,
        help="Maximum number of possible values for a discrete feature",
    )

    parser.add_argument(
        "--exp-name",
        help="Suffix added to the saved file during evaluation",
    )

    return parser


def privacy_parser():
    parser = argparse.ArgumentParser(
        description="SDG PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument("--fake-dataset", help="Name of fake dataset to evaluate")

    parser.add_argument("--real-dataset", help="Name of real dataset to evaluate")

    parser.add_argument(
        "--category-threshold",
        type=int,
        default=3,
        help="Maximum number of possible values for a discrete feature",
    )

    parser.add_argument(
        "--key-fields",
        nargs="+",
        help="A list of strings representing the column names that the attacker already knows.",
    )

    parser.add_argument(
        "--sensitive-fields",
        nargs="+",
        help="A list of string representing the column names that the attacker wants to guess.",
    )

    return parser


def utility_parser():
    parser = argparse.ArgumentParser(
        description="SDG PROJECT", epilog="Developped by Quinten"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument("--fake-dataset", help="Name of fake dataset to evaluate")

    parser.add_argument("--real-dataset", help="Name of real dataset to evaluate")

    parser.add_argument(
        "--category-threshold",
        type=int,
        default=3,
        help="Maximum number of possible values for a discrete feature",
    )

    return parser
