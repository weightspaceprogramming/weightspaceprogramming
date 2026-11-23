"""Check the exact file allowlists for the wheel and source distribution."""
import json
import hashlib
from pathlib import Path
import sys
import tarfile
import tomllib
import zipfile


def check(directory):
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    name = f"weightspaceprogramming-{version}"
    module_files = {"__init__.py", "__main__.py", "core.py", "inspection.py", "cli.py", "lessons.py", "lessons.json"}
    wheel_files = {f"weightspaceprogramming/{file}" for file in module_files}
    wheel_files |= {f"{name}.dist-info/{file}" for file in ["METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD", "licenses/LICENSE", "licenses/SOURCES.md"]}
    sdist_files = {f"{name}/src/weightspaceprogramming/{file}" for file in module_files}
    sdist_files |= {f"{name}/{file}" for file in ["README.md", "SOURCES.md", "LICENSE", "pyproject.toml", "MANIFEST.in", "PKG-INFO", "setup.cfg", "tests/test_toolkit.py", "src/weightspaceprogramming.egg-info/SOURCES.txt"]}
    archives = [(Path(directory) / f"{name}-py3-none-any.whl", wheel_files),
                (Path(directory) / f"{name}.tar.gz", sdist_files)]
    result = {}
    for path, expected in archives:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                members = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
                hashes = {member: hashlib.sha256(archive.read(member)).hexdigest() for member in members}
        else:
            with tarfile.open(path) as archive:
                if any(entry.issym() or entry.islnk() for entry in archive.getmembers()):
                    raise ValueError(f"Links are not permitted in {path.name}")
                members = [entry.name for entry in archive.getmembers() if entry.isfile()]
                hashes = {member: hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in members}
        actual = set(members)
        if len(members) != len(actual) or actual != expected:
            raise ValueError(f"{path.name}: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}")
        result[path.name] = dict(sorted(hashes.items()))
    return {"version": version, "archives": result}


if __name__ == "__main__":
    actual = check(sys.argv[1] if len(sys.argv) > 1 else "dist")
    if "--record" not in sys.argv:
        expected = json.loads((Path(__file__).resolve().parents[1] / "release-artifacts.json").read_text())
        if actual != expected:
            raise ValueError("Archive file contents differ from release-artifacts.json")
    print(json.dumps(actual, indent=2))
