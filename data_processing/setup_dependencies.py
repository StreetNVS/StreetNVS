"""Install pinned official Waymo readers or fetch the native point rasterizer."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

RUNTIME = Path(__file__).resolve().parents[1] / ".runtime"
WAYMO_REVISION = "99a4cb3ff07e2fe06c2ce73da001f850f628e45a"
RASTERIZER_REVISION = "a67579e0bc96a6b3a2655637bd6f8637540b2947"


def checkout(name, url, revision):
    path = RUNTIME / "sources" / name
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "remote", "add", "origin", url], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", "origin", revision],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", revision], check=True
        )
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != revision:
        raise ValueError(f"Unexpected dependency revision in {path}: {head}")
    return path


def install_waymo(source):
    import grpc_tools

    destination = RUNTIME / "waymo"
    for path in (source / "src/waymo_open_dataset").rglob("*.py"):
        if path.name.endswith("_test.py") or "testdata" in path.parts:
            continue
        output = destination / path.relative_to(source / "src")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
    protos = sorted((source / "src/waymo_open_dataset").rglob("*.proto"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{source / 'src'}",
            "-I" + str(Path(grpc_tools.__file__).parent / "_proto"),
            f"--python_out={destination}",
            *map(str, protos),
        ],
        check=True,
    )
    (Path(sysconfig.get_paths()["purelib"]) / "streetnvs-waymo.pth").write_text(
        str(destination) + "\n"
    )
    print("Installed official Python readers; no native Waymo wheel is required.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dependency", choices=["waymo", "rasterizer"])
    args = p.parse_args()
    if sys.prefix == sys.base_prefix:
        p.error("Run setup inside the appropriate venv")
    if args.dependency == "waymo":
        source = checkout(
            "waymo-open-dataset",
            "https://github.com/waymo-research/waymo-open-dataset.git",
            WAYMO_REVISION,
        )
        install_waymo(source)
    else:
        source = checkout(
            "diff-point-rasterization",
            "https://github.com/yunzhiy/diff-point-rasterization.git",
            RASTERIZER_REVISION,
        )
        subprocess.run(
            ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
        # GCC 13 needs the explicit uint32_t declaration when compiling CUDA 12.8.
        header = source / "cuda_rasterizer/rasterizer_impl.h"
        text = header.read_text()
        if "#include <cstdint>" not in text:
            header.write_text("#include <cstdint>\n" + text)


if __name__ == "__main__":
    main()
