# -*- coding: utf-8 -*-
"""Fonctions utilitaires pour les autres scripts."""
import datetime as dt
from pathlib import Path
from argparse import ArgumentParser

try:
    from dask.distributed import Client as DaskClient
except ImportError:
    DaskClient = None


class CommandOutputStreams(object):
    """
    A context manager for running command with subprocess.

    Appends the stdout and stderr of the process to given files.
    Adds a line before stating the command and, if it exits with an error, a line at the end stating the error.
    """

    def __init__(self, command, stdout="stdout.log", stderr="stderr.log"):
        """
        Initialize the streams with the command to be launched and the file names.

        'command' can be a string or a list of arguments in the style of 'subprocess.Popen'.
        """
        self.command = command if isinstance(command, str) else " ".join(command)
        self.stderr_file = stderr
        self.stdout_file = stdout

    def __enter__(self):
        """Enter the context. Open the files, write headers and return the handles."""
        self.start = dt.datetime.now()
        self.stdout = open(self.stdout_file, "a")
        self.stdout.write(
            "***[{}] Output stream of command: {}\n".format(self.start, self.command)
        )
        self.stdout.flush()
        self.stderr = open(self.stderr_file, "a")
        self.stderr.write(
            "***[{}] Error stream of command: {}\n".format(self.start, self.command)
        )
        self.stderr.flush()
        return self.stdout, self.stderr

    def __exit__(self, exc_type, *args):
        """Exit the context. Write footers and close files."""
        duration = dt.datetime.now() - self.start
        self.stdout.write("*** Command exited afer {}".format(duration))
        if exc_type is not None:
            self.stderr.write(
                "*** Command exited with error of type {}".format(exc_type)
            )
        self.stdout.close()
        self.stderr.close()


def _outFileParser(argstr):
    """Parse the out file argument."""
    if argstr.startswith("_"):
        return Path(".").resolve().name + argstr
    return argstr


def _sizeParser(argstr):
    """Parse the size argument."""
    size = [int(float(siz)) for siz in argstr.split("x")]
    if len(size) == 1:  # If one value was given, assume cube
        return size * 3
    return size


def _listParser(caster=str):
    """Parse a comma separated list argument."""

    def _typedlistParser(argstr):
        return [caster(arg) for arg in argstr.split(",")]

    return _typedlistParser


baseArgParser = ArgumentParser(add_help=False)
baseArgParser.add_argument(
    "-o",
    "--out",
    type=_outFileParser,
    default=Path(".").resolve().name,
    help=(
        "Filename to use in figures/data files (defaults to current directory's name + suffix)."
        " If starts with _, appends this to the directory's name."
    ),
)
baseArgParser.add_argument(
    "--show-fig",
    help="Shows the figure before exiting (do not use in scripts!).",
    action="store_true",
)
baseArgParser.add_argument(
    "--no-plot",
    help="Doesn't plot the calculation, only saves the data files.",
    action="store_true",
)

baseScriptArgParser = ArgumentParser(parents=[baseArgParser], add_help=False)
baseScriptArgParser.add_argument(
    "-d",
    "--size",
    help="Cell size (m) as 2E3x2E3x1E2 (xyz). If only one number, a cube is supposed.",
    type=_sizeParser,
    default=[1, 1, 1],
)
baseScriptArgParser.add_argument(
    "-t", "--dt", help="Timestep duration (s)", type=float, default=1
)


class COLORS:
    WHITE = "\033[97m"
    LIGHTBLUE = "\033[96m"
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    DARK = "\033[90m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    STRIKE = "\033[9m"
    ENDC = "\033[0m"


def parse_kwarg(val):
    if "," in val:
        return [parse_kwarg(va) for va in val.split(",")]
    else:
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                if val.casefold() == "none":
                    return None
                elif val.casefold() == "true":
                    return True
                elif val.casefold() == "false":
                    return False
                return val


class CaseInsensitiveDict(dict):
    def __init__(self, other=None):
        super().__init__()
        if other:
            for k, v in other.items():
                super().__setitem__(k.casefold(), v)

    def __getitem__(self, key):
        return super().__getitem__(key.casefold())

    def __setitem__(self, key, val):
        super().__setitem__(key.casefold(), val)

    def pop(self, key):
        return super().pop(key.casefold())

    def get(self, key, default):
        return super().get(key.casefold(), default)
