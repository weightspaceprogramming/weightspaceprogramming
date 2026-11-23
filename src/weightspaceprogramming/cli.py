"""Run a lesson or inspect an NPZ input-layer kernel."""
import argparse
import json
from pathlib import Path

from . import __version__, channel_norms, lesson_names, lesson_source, load_arrays, plot_channel_norms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    lesson = sub.add_parser("lesson", help="Run an offline Python lesson")
    lesson.add_argument("name", choices=("list", *lesson_names()))
    lesson.add_argument("--show-code", action="store_true")
    inspect = sub.add_parser("inspect", help="Compute channel norms from a local NPZ kernel")
    inspect.add_argument("checkpoint", type=Path)
    inspect.add_argument("--key", default="actor/dense1/kernel")
    inspect.add_argument("--shape", type=int, nargs=3, required=True, metavar=("H", "W", "C"))
    inspect.add_argument("--output", type=Path, help="Save numeric JSON (otherwise print it)")
    inspect.add_argument("--plot", type=Path, help="Save a PNG or SVG; requires the plot extra")
    args = parser.parse_args(argv)
    try:
        if args.command == "lesson":
            if args.name == "list":
                print("\n".join(lesson_names()))
            elif args.show_code:
                print(lesson_source(args.name), end="")
            else:
                exec(compile(lesson_source(args.name), f"<lesson:{args.name}>", "exec"), {"__name__": "__main__"})
            return 0
        arrays = load_arrays(args.checkpoint)
        if args.key not in arrays:
            raise ValueError(f"Kernel {args.key!r} is absent; keys: {', '.join(arrays)}")
        norms = channel_norms(arrays[args.key], tuple(args.shape))
        data = {"kernel": args.key, "grid_shape_hwc": args.shape,
                "computation": "sqrt(sum_j W[(row*width+column)*channels+channel,j]^2)",
                "norm_maps_chw": norms.tolist()}
        encoded = json.dumps(data, indent=2) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        if args.plot:
            plot_channel_norms(norms, output=args.plot)
        return 0
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
