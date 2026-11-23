"""Editable, deterministic examples shared by the Python and browser lessons."""
from importlib.resources import files
import json


def _lessons():
    return json.loads(files("weightspaceprogramming").joinpath("lessons.json").read_text(encoding="utf-8"))


def lesson_names():
    """Return the available lesson names in teaching order."""
    return tuple(_lessons())


def lesson_source(name):
    """Return a lesson's complete runnable Python source."""
    try:
        return _lessons()[name]["code"]
    except KeyError as error:
        raise ValueError(f"Unknown lesson {name!r}; choose from {', '.join(lesson_names())}") from error
