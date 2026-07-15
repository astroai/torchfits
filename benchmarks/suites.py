"""Named benchmark suites for modular / release orchestration.

A suite maps to ``bench_all.py`` flags (scope, filter, operation, gpu, mmap, profile).
Presets for deficit focus and release exhaustives live here so shell wrappers stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MmapMode = Literal["matrix", "on", "off"]
Scope = Literal["all", "fits", "fitstable"]
Profile = Literal["user", "lab"]


@dataclass(frozen=True)
class Suite:
    name: str
    scope: Scope
    case_filter: str = ""
    operation: str = ""
    no_gpu: bool = False
    # When True, skip CPU image/table ops and keep GPU transport rows only.
    gpu_only: bool = False
    # Modular suites default to a single mmap mode; release/deficit use matrix.
    mmap: MmapMode = "on"
    profile: Profile = "lab"
    aliases: tuple[str, ...] = ()


SUITES: dict[str, Suite] = {}


def _reg(suite: Suite) -> Suite:
    SUITES[suite.name] = suite
    for alias in suite.aliases:
        SUITES[alias] = suite
    return suite


_reg(
    Suite(
        name="compressed_hcompress",
        scope="fits",
        case_filter="^(compressed_hcompress_)",
        mmap="matrix",
        aliases=("hcompress",),
    )
)
_reg(
    Suite(
        name="compressed_rice",
        scope="fits",
        case_filter="^(compressed_rice_)",
        aliases=("rice",),
    )
)
_reg(
    Suite(
        name="tiny_int8",
        scope="fits",
        case_filter="^(tiny_int8_)",
        mmap="matrix",
    )
)
_reg(
    Suite(
        name="cutouts",
        scope="fits",
        operation="cutout|repeated_cutouts",
        aliases=("cutout",),
    )
)
_reg(
    Suite(
        name="images_core",
        scope="fits",
        case_filter="^(medium_|large_)",
        operation="^read_full$",
        no_gpu=True,
    )
)
_reg(
    Suite(
        name="fitstable_predicate",
        scope="fitstable",
        case_filter="^(narrow_1000|narrow_10000)$",
        operation="predicate",
        no_gpu=True,
        mmap="matrix",
        aliases=("predicate",),
    )
)
_reg(
    Suite(
        name="fitstable_core",
        scope="fitstable",
        no_gpu=True,
    )
)
_reg(
    Suite(
        name="gpu_transports",
        scope="fits",
        gpu_only=True,
    )
)
_reg(
    Suite(
        name="release_fits",
        scope="fits",
        mmap="matrix",
    )
)
_reg(
    Suite(
        name="release_fitstable",
        scope="fitstable",
        no_gpu=True,
        mmap="matrix",
    )
)
_reg(
    Suite(
        name="release",
        scope="all",
        mmap="matrix",
        profile="lab",
    )
)

# Suites run by ``bench-deficit-focus`` / ``bench_deficit_focus.sh``.
DEFICIT_FOCUS_SUITES: tuple[str, ...] = (
    "compressed_hcompress",
    "tiny_int8",
    "fitstable_predicate",
)


def resolve_suite(name: str) -> Suite:
    key = name.strip().lower().replace("-", "_")
    if key not in SUITES:
        known = ", ".join(sorted({s.name for s in SUITES.values()}))
        raise KeyError(f"unknown suite {name!r}; known: {known}")
    return SUITES[key]


def list_suite_names() -> list[str]:
    return sorted({s.name for s in SUITES.values()})
