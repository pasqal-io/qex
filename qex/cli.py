"""Command-line interface for QEX.

Exposes the ``qex`` console script (see ``pyproject.toml``), structured as a set
of subcommands::

    qex train --config examples/h2.yaml
    qex train --config examples/h2.yaml --training.learning_rate 5e-4 --platform gpu

A run is fully described by a YAML config; any scalar leaf can be overridden on
the command line with its dot-path (``--section.key value``). The override
parser is built from the *loaded* config (not a fixed default), so the available
flags always match the file you pass in.

For ergonomics, a bare ``qex --config ...`` (no subcommand) defaults to
``train``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from qex.config import Config

# Registry of subcommands: name -> (handler, one-line help). Adding a verb
# (e.g. "tune", "eval") is a one-line change here plus its handler function.
_COMMANDS: dict[str, tuple[Callable[[list[str]], int], str]] = {}


def _register(name: str, help_text: str) -> Callable:
    def deco(fn: Callable[[list[str]], int]) -> Callable[[list[str]], int]:
        _COMMANDS[name] = (fn, help_text)
        return fn

    return deco


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into ``{"a.b.c": leaf}`` dot-path form."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _coerce_scalar(raw: str) -> Any:
    """Best-effort scalar coercion: int -> float -> leave as string."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _coerce(raw: str, template: Any) -> Any:
    """Coerce a CLI string to the type of the existing config value.

    Lists are passed as a single comma/space-separated string, e.g.
    ``--data.test_bond_lengths "0.8 1.3 2.0"``; each element is coerced to the
    element type of the existing list (falling back to scalar inference).
    """
    if isinstance(template, (list, tuple)):
        items = [tok for tok in raw.replace(",", " ").split() if tok]
        elem = template[0] if template else None
        if isinstance(elem, bool):
            return [tok.lower() in ("1", "true", "yes", "on") for tok in items]
        if isinstance(elem, int) and not isinstance(elem, bool):
            return [int(tok) for tok in items]
        if isinstance(elem, float):
            return [float(tok) for tok in items]
        return [_coerce_scalar(tok) for tok in items]
    if template is None:
        return _coerce_scalar(raw)
    if isinstance(template, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(template, int):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    return raw


def _build_train_parser(config: Config) -> argparse.ArgumentParser:
    """Create the ``train`` parser, exposing every config leaf as an override."""
    parser = argparse.ArgumentParser(
        prog="qex train",
        description="Train a (quantum-)neural XC functional from a YAML config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML experiment config (e.g. examples/h2.yaml).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip rendering/saving the dissociation profile plot.",
    )
    for path in _flatten(config.config):
        # Override flags accept a raw string; coercion happens after parsing
        # against the config's existing leaf type.
        parser.add_argument(f"--{path}", dest=path, default=None, metavar="VALUE")
    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> None:
    """Write any provided ``--section.key`` overrides back into the config."""
    # CLI-only flags that are not config leaves and must not be written back.
    reserved = {"config", "no_plot", "output", "force"}
    for dest, raw in vars(args).items():
        if dest in reserved or raw is None:
            continue
        config.set(dest, _coerce(raw, config.get(dest)))


@_register("train", "Train an XC functional and evaluate the dissociation curve.")
def _train_command(argv: list[str]) -> int:
    """Handler for ``qex train``."""
    # Two-pass parse: first read --config so we know which leaves exist, then
    # build the full parser (with per-leaf overrides) from the loaded file.
    # `--config` is left optional in the pre-parser so `--help` works without a
    # config; the full parser below still enforces it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)

    if known.config is None:
        # No config supplied (e.g. `--help` or a forgotten flag). Build a bare
        # parser so argparse prints help or the "required: --config" error.
        parser = argparse.ArgumentParser(
            prog="qex train",
            description=(
                "Train a (quantum-)neural XC functional from a YAML config."
            ),
        )
        parser.add_argument(
            "--config", required=True, help="Path to the YAML config."
        )
        parser.add_argument("--no-plot", action="store_true")
        parser.parse_args(argv)  # exits: prints --help or the required error

    config = Config(config_path=known.config)
    parser = _build_train_parser(config)
    args = parser.parse_args(argv)
    _apply_overrides(config, args)

    # Imported lazily so `--help` doesn't pay the JAX/PySCF import cost.
    from qex.training import run_experiment

    result = run_experiment(config, make_plot=not args.no_plot)

    print("\n" + "-" * 70)
    print("Training and evaluation complete!")
    print(f"  MAE: {result.mae:.3e} Ha   NPE: {result.npe:.3e} Ha")
    print(f"  Results written to: {result.output_dir}")
    print("-" * 70)
    return 0


def _read_hash(path: Path) -> str | None:
    """Read the ``config_hash`` attr of an .h5, returning None if absent/unreadable."""
    from qex.data_io import read_config_hash

    try:
        return read_config_hash(path)
    except Exception:  # noqa: BLE001 - a corrupt/old file just counts as a miss
        return None


def _gen_data_from_systems(systems_path: str, out_path: str | None, force: bool) -> int:
    """Build a dataset straight from a declarative systems file (no training config).

    This is the simple path: a YAML of molecules in, one ``.h5`` out, ready to
    train against. Resumable -- re-running reuses systems already on disk; a
    stale file (built for *different* systems, by content hash) or ``--force``
    triggers a rebuild.
    """
    from qex.data_io import (
        DataGenerator,
        build_dataset,
        load_systems_file,
        systems_file_hash,
    )

    split_configs = load_systems_file(systems_path)
    out = Path(out_path or "dataset.h5")
    cfg_hash = systems_file_hash(systems_path)

    if out.exists() and (force or _read_hash(out) != cfg_hash):
        print(f"Rebuilding {out} ({'forced' if force else 'systems changed'}).")
        out.unlink()

    out.parent.mkdir(parents=True, exist_ok=True)
    data_generator = DataGenerator(out.parent if str(out.parent) else ".")
    print(
        "Generating dataset (resumable) -> "
        + " | ".join(f"{s}: {len(c)}" for s, c in split_configs.items())
        + f"  [{out}]"
    )
    # Always store grid/atom-coordinate context: it is cheap (just coordinates)
    # and makes the dataset usable by *any* model -- descriptor networks need it,
    # others ignore the extra arrays. So one dataset serves every training config.
    dataset = build_dataset(
        data_generator,
        split_configs,
        path=out,
        with_descriptor_ctx=True,
        config_hash=cfg_hash,
    )
    print(
        f"Dataset ready -> train: {len(dataset.train)} | "
        f"val: {len(dataset.val)} | test: {len(dataset.test)}  [{out}]"
    )
    return 0


@_register("gen-data", "Generate a single-file (HDF5) train/val/test dataset.")
def _gen_data_command(argv: list[str]) -> int:
    """Handler for ``qex gen-data``: build the dataset file without training.

    Two ways to say what to build:
      --systems FILE   declarative molecules YAML (no training config needed)
      --config FILE    derive systems from an experiment config (legacy path)
    """
    # Simple path -- `qex gen-data --systems systems.yaml -o data.h5` -- handled
    # first so it needs neither a training config nor a built network.
    sys_pre = argparse.ArgumentParser(add_help=False)
    sys_pre.add_argument("--systems")
    sys_pre.add_argument("-o", "--output")
    sys_pre.add_argument("--force", action="store_true")
    sys_known, _ = sys_pre.parse_known_args(argv)
    if sys_known.systems is not None:
        return _gen_data_from_systems(sys_known.systems, sys_known.output, sys_known.force)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)

    if known.config is None:
        parser = argparse.ArgumentParser(
            prog="qex gen-data",
            description="Generate a single-file HDF5 train/val/test dataset.",
        )
        parser.add_argument(
            "--systems",
            help="Declarative molecules YAML (alternative to --config; no training config needed).",
        )
        parser.add_argument("--config", required=True, help="Path to the YAML config.")
        parser.add_argument("-o", "--output", help="Output .h5 path.")
        parser.add_argument("--force", action="store_true")
        parser.parse_args(argv)  # exits with --help or the required error

    config = Config(config_path=known.config)
    parser = _build_train_parser(config)  # reuse: same --config + dot-path overrides
    parser.prog = "qex gen-data"
    parser.add_argument("-o", "--output", default=None, help="Output .h5 path.")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if a cache exists."
    )
    args = parser.parse_args(argv)
    _apply_overrides(config, args)

    # Imported lazily so `--help` doesn't pay the JAX/PySCF import cost.
    from qex.training import build_network, dataset_for_config
    from qex.utils.logging import configure_logging

    # gen-data doesn't go through run_experiment, so honor the log config here.
    configure_logging(
        debug=bool(config.get("debug", False)),
        level=config.get("logging.level", None),
    )

    _, _, _, is_descriptor = build_network(config)
    dataset = dataset_for_config(
        config,
        is_descriptor=is_descriptor,
        cache_path=args.output or config.get("data.dataset_file", None),
        use_cache=not args.force,
    )
    print(
        f"Dataset ready -> train: {len(dataset.train)} | "
        f"val: {len(dataset.val)} | test: {len(dataset.test)}"
    )
    return 0


def _print_top_level_help() -> None:
    """Print the top-level ``qex`` usage and the list of subcommands."""
    width = max(len(name) for name in _COMMANDS)
    lines = ["usage: qex <command> [options]", "", "commands:"]
    for name, (_, help_text) in _COMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {help_text}")
    lines += [
        "",
        "Run 'qex <command> --help' for command-specific options.",
        "A bare 'qex --config ...' (no command) defaults to 'train'.",
    ]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``qex`` console script (subcommand dispatcher)."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)

    # Top-level help with no command.
    if not argv or argv[0] in ("-h", "--help"):
        _print_top_level_help()
        return 0

    first = argv[0]
    if first in _COMMANDS:
        handler, _ = _COMMANDS[first]
        return handler(argv[1:])

    # Ergonomic fallback: `qex --config ...` (a flag, not a command) -> train.
    if first.startswith("-"):
        return _train_command(argv)

    print(f"qex: unknown command {first!r}\n")
    _print_top_level_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
