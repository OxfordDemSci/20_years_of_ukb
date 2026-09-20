#!/usr/bin/env bash
# Run with the analysis environment active, or PYTHON=/path/to/env/bin/python.
set -eu
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" - "$repo_root" "$@" <<'PY'
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import time

root = Path(sys.argv[1])
parser = argparse.ArgumentParser(
    description="Run active analysis notebooks sequentially in filename order; continue after errors.",
    epilog="Original notebooks are preserved. Notebook code still writes its usual figures/data, "
           "installs packages, or calls APIs where specified in the notebook.",
)
parser.add_argument("--list", action="store_true", help="show execution order without running")
parser.add_argument("--from", dest="start", metavar="FILENAME", help="start at this exact notebook filename")
parser.add_argument("--timeout", type=int, default=-1, metavar="SECONDS",
                    help="per-cell timeout; default -1 means unlimited")
parser.add_argument("--notebook-dir", type=Path, default=root / "src/data_analysis",
                    help="directory containing notebooks (no recursion into archives)")
args = parser.parse_args(sys.argv[2:])
notebooks = sorted(args.notebook_dir.resolve().glob("*.ipynb"))
if not notebooks:
    parser.error(f"no notebooks found in {args.notebook_dir}")
if args.start:
    names = [p.name for p in notebooks]
    if args.start not in names:
        parser.error(f"unknown notebook: {args.start}")
    notebooks = notebooks[names.index(args.start):]
if args.timeout != -1 and args.timeout <= 0:
    parser.error("--timeout must be -1 or a positive number")
if args.list:
    for number, path in enumerate(notebooks, 1):
        print(f"{number:02d}. {path.name}")
    sys.exit(0)

try:
    import nbformat
    from nbclient import NotebookClient
    import ipykernel  # Fail early if this interpreter cannot launch notebook kernels.
except ImportError as exc:
    print(f"Runner dependency missing: {exc}\nPython: {sys.executable}\n"
          "Activate your analysis environment, then install nbclient nbformat ipykernel.",
          file=sys.stderr)
    sys.exit(2)

runs = root / "output/notebook_runs"
runs.mkdir(parents=True, exist_ok=True)
run_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=runs))
summary = run_dir / "summary.csv"
rows = []
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
use_colour = sys.stdout.isatty() and "NO_COLOR" not in os.environ

def report(status, message):
    colour = "32" if status == "PASS" else "31"
    label = f"\033[{colour}m{status}\033[0m" if use_colour else status
    print(f"[{label}] {message}", flush=True)

def save_summary():
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "status", "notebook", "seconds", "error", "log", "executed_notebook",
        ])
        writer.writeheader()
        writer.writerows(rows)

# A temporary kernelspec ensures notebook kernels use THIS interpreter, even if
# a globally registered 'python3' kernel points to a different environment.
with tempfile.TemporaryDirectory(prefix="ukb-notebook-kernel-") as kernel_root:
    kernel_dir = Path(kernel_root) / "kernels/ukb-analysis-runner"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "UKB analysis runner", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = os.pathsep.join(filter(None, [
        kernel_root, os.environ.get("JUPYTER_PATH"),
    ]))
    for number, path in enumerate(notebooks, 1):
        log_path = run_dir / f"{path.stem}.log"
        executed_path = run_dir / path.name
        started = time.monotonic()
        status, error, notebook = "PASS", "", None
        current_cell = None
        print(f"[{number}/{len(notebooks)}] {path.stem} ", end="", flush=True)
        with log_path.open("w") as log:
            print(f"Notebook: {path.name}\nPython: {sys.executable}", file=log, flush=True)
            try:
                notebook = nbformat.read(path, as_version=4)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        cell.outputs = []
                        cell.execution_count = None
                        cell.metadata.pop("execution", None)

                def cell_started(cell, cell_index, **kwargs):
                    global current_cell
                    if cell.cell_type == "code" and cell.source.strip():
                        current_cell = cell_index + 1

                client = NotebookClient(
                    notebook, kernel_name="ukb-analysis-runner", timeout=args.timeout,
                    allow_errors=False, force_raise_errors=True, record_timing=True,
                    resources={"metadata": {"path": str(root)}},
                    on_cell_start=cell_started,
                )
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    client.execute()
                # Even explicitly error-tolerant notebook cells must be reported.
                errors = [o for c in notebook.cells for o in c.get("outputs", [])
                          if o.output_type == "error"]
                if errors:
                    raise RuntimeError(f"{errors[0].ename}: {errors[0].evalue}")
            except KeyboardInterrupt:
                status, error = "INTERRUPTED", "Stopped by user"
            except Exception as exc:
                status = "FAIL"
                message = getattr(exc, "evalue", str(exc))
                message = next((line.strip() for line in message.splitlines() if line.strip()), "")
                error = f"{getattr(exc, 'ename', type(exc).__name__)}: {message}"
                if notebook is not None:
                    errors = [o for c in notebook.cells for o in c.get("outputs", [])
                              if o.output_type == "error"]
                    if errors:
                        error = f"{errors[-1].ename}: {errors[-1].evalue}"
            finally:
                if notebook is not None:
                    for cell in notebook.cells:
                        for output in cell.get("outputs", []):
                            if output.output_type == "error":
                                output.traceback = []
                    nbformat.write(notebook, executed_path)
                error = " ".join(ansi.sub("", error).split())
                location = f"Cell {current_cell}: " if current_cell is not None else ""
                print(f"{status}: {location + error if error else 'Completed'}", file=log)
        elapsed = round(time.monotonic() - started, 1)
        rows.append(dict(status=status, notebook=path.name, seconds=elapsed, error=error,
                         log=str(log_path),
                         executed_notebook=str(executed_path) if notebook is not None else ""))
        save_summary()
        brief_error = textwrap.shorten(
            error.replace(str(root) + os.sep, ""), width=180, placeholder="…",
        )
        report(status, f"{elapsed}s" + (f" — {brief_error}" if brief_error else ""))
        if status == "INTERRUPTED":
            break

counts = {status: sum(row["status"] == status for row in rows)
          for status in ("PASS", "FAIL", "INTERRUPTED")}
totals = f"{counts['PASS']} passed, {counts['FAIL']} failed"
if counts["INTERRUPTED"]:
    totals += f", {counts['INTERRUPTED']} interrupted, {len(notebooks) - len(rows)} not run"
print(f"\n{totals}.\nLogs and summary.csv: {run_dir.relative_to(root)}", flush=True)
if any(row["status"] == "INTERRUPTED" for row in rows):
    sys.exit(130)
sys.exit(1 if any(row["status"] == "FAIL" for row in rows) else 0)
PY
