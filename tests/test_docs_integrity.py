from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _collect_nav_paths(nav: list[Any]) -> list[str]:
    paths: list[str] = []
    for item in nav:
        if isinstance(item, str):
            paths.append(item)
            continue
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    if not value.startswith("http"):
                        paths.append(value)
                else:
                    paths.extend(_collect_nav_paths(value))
    return paths


def test_docs_reference_existing_local_files() -> None:
    expected_paths = [
        "docs/index.md",
        "zensical.toml",
        "docs/api.md",
        "docs/benchmarks.md",
        "docs/changelog.md",
        "docs/examples.md",
        "docs/install.md",
        "docs/parity.md",
        "docs/roadmap.md",
        "docs/migration_datasets.md",
        "docs/migration_fitsio.md",
        "docs/migration_astropy.md",
        "examples/example_image.py",
        "examples/example_image_cube.py",
        "examples/example_image_cutouts.py",
        "examples/example_image_dataset.py",
        "examples/example_data_catalogs.py",
        "examples/example_transforms.py",
        "examples/example_hyperspectral.py",
        "examples/example_image_mef.py",
        "examples/example_polars.py",
        "examples/example_table.py",
        "examples/example_table_interop.py",
        "examples/example_table_recipes.py",
        "examples/example_time_series.py",
        "benchmarks/bench_all.py",
        "benchmarks/bench_arrow_tables.py",
        "benchmarks/bench_cpp_backend.py",
        "benchmarks/bench_fast.py",
        "benchmarks/bench_fits_io.py",
        "benchmarks/bench_fitstable_io.py",
        "benchmarks/bench_gpu_transports.py",
        "benchmarks/bench_table.py",
        "scripts/launch_canfar_gpu_bench.sh",
        "scripts/canfar_gpu_bench_incontainer.sh",
        "scripts/canfar_gpu_bench_remote.sh",
        "scripts/selfcheck_canfar_launcher.sh",
        "scripts/gpu-bootstrap.sh",
        "scripts/ci_local.sh",
        "scripts/patch_canfar_exhaustive_docs.sh",
        "scripts/publish_canfar_bench_vos.sh",
        "scripts/fetch_canfar_bench_vos.sh",
        "scripts/import_canfar_bench_artifacts.py",
        "scripts/run_exhaustive_bench_and_patch_docs.sh",
    ]

    missing = [path for path in expected_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing doc-referenced files: {missing}"


def test_public_docs_do_not_claim_torchfits_owns_sky_domain_features() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "MAINTENANCE.md",
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "benchmarks.md",
        ROOT / "docs" / "changelog.md",
        ROOT / "docs" / "contributing.md",
        ROOT / "docs" / "examples.md",
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "parity.md",
        ROOT / "docs" / "release.md",
        ROOT / "docs" / "roadmap.md",
    ]
    forbidden_claims = [
        "covers the same ground",
        "torchfits.get_wcs",
        "torchfits.sphere",
        "healpy-compatible",
        "spherical harmonics",
        "spherical polygons",
        "Sparse HEALPix",
        "ML Integration",
        "torchsky",
    ]

    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for claim in forbidden_claims:
            if claim in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {claim!r}")

    assert not offenders, "\n".join(offenders)


def test_public_docs_do_not_reference_missing_root_cache_aliases() -> None:
    """Cache tuning lives on torchfits.cache; root exposes I/O cache helpers only."""
    docs = [
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "install.md",
    ]
    forbidden_root_calls = [
        "torchfits.configure_for_environment(",
        "torchfits.get_cache_stats(",
        "torchfits.clear_cache(",
    ]

    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for call in forbidden_root_calls:
            if call in text:
                offenders.append(f"{path.relative_to(ROOT)} references {call!r}")

    assert not offenders, "\n".join(offenders)


def test_docs_do_not_advertise_unimplemented_worker_handle_env() -> None:
    """TORCHFITS_WORKER_HANDLE is roadmap-only; docs must not present it as settable."""
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "api.md",
        ROOT / "docs" / "install.md",
        ROOT / "docs" / "examples.md",
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "migration_astropy.md",
        ROOT / "docs" / "migration_fitsio.md",
        ROOT / "docs" / "migration_datasets.md",
    ]
    # Allowed: honesty notes that the feature does not exist.
    # Forbidden: documentation that tells users to set / rely on it.
    forbidden = [
        "TORCHFITS_WORKER_HANDLE=1",
        "setting the environment variable `TORCHFITS_WORKER_HANDLE",
        "set the environment variable `TORCHFITS_WORKER_HANDLE",
        "When the environment variable `TORCHFITS_WORKER_HANDLE",
    ]
    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for claim in forbidden:
            if claim in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {claim!r}")
    assert not offenders, "\n".join(offenders)


def test_api_md_env_var_table_matches_source() -> None:
    """Only document TORCHFITS_* env vars that exist in the tree (table rows)."""
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    # Rows in the Environment variables Markdown table: | `TORCHFITS_...` | ...
    documented = set(re.findall(r"\|\s*`?(TORCHFITS_[A-Z0-9_]+)`?\s*\|", api))
    source_envs: set[str] = set()
    for path in (ROOT / "src").rglob("*"):
        if path.suffix not in {".py", ".cpp", ".h", ".hpp", ".cc", ".cu"}:
            continue
        if not path.is_file():
            continue
        source_envs.update(
            re.findall(
                r"TORCHFITS_[A-Z0-9_]+",
                path.read_text(encoding="utf-8", errors="ignore"),
            )
        )
    missing = sorted(documented - source_envs)
    assert not missing, f"api.md env table documents missing vars: {missing}"


def test_api_md_core_io_signatures_match_live() -> None:
    """Guard against invented parameters on the most-copied Core I/O signatures."""
    import torchfits

    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    def _section(heading: str) -> str:
        pattern = rf"### `{re.escape(heading)}`\n(.*?)(?=\n### |\n## |\Z)"
        match = re.search(pattern, api, flags=re.S)
        assert match, f"missing section for {heading}"
        return match.group(1)

    def _first_signature_block(section: str) -> str:
        match = re.search(r"```python\n(def .*?)\n```", section, flags=re.S)
        assert match, "missing python signature fence"
        return match.group(1)

    read_tensor_sig = _first_signature_block(_section("torchfits.read_tensor"))
    assert "scale_on_device" not in read_tensor_sig, (
        "read_tensor must not document scale_on_device (that belongs to read_fast / kwargs)"
    )
    assert "handle_cache" in read_tensor_sig

    subset_section = _section("torchfits.read_subset")
    subset_sig = _first_signature_block(subset_section)
    assert "device" not in subset_sig
    assert "handle_cache_capacity" in subset_sig
    assert (
        "exclusive" in subset_section.lower() or "half-open" in subset_section.lower()
    )

    stream_section = _section("torchfits.stream_table")
    stream_sig = _first_signature_block(stream_section)
    assert "file_path" in stream_sig
    assert (
        "dict[str, torch.Tensor]" in stream_section
        or "dict[str, Tensor]" in stream_section
    )

    read_section = _section("torchfits.read")
    assert (
        "dict[str, torch.Tensor]" in read_section or "dict[str, Tensor]" in read_section
    )

    # Live sanity: key helpers remain importable
    assert callable(torchfits.read_table)
    assert callable(torchfits.read_subset)


def test_docs_examples_reference_existing_scripts() -> None:
    text = (ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    refs = re.findall(r"\]\(\.\./examples/([^)]+)\)", text)
    missing_scripts = [name for name in refs if not (ROOT / "examples" / name).exists()]
    assert not missing_scripts, (
        f"docs/examples.md references missing scripts: {missing_scripts}"
    )


def test_zensical_config_targets_existing_docs() -> None:
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    project = config["project"]
    site_url = project["site_url"]
    assert site_url == "https://astroai.github.io/torchfits/"
    assert project["repo_url"] == "https://github.com/astroai/torchfits"

    nav_paths = _collect_nav_paths(project["nav"])
    missing = [path for path in nav_paths if not (ROOT / "docs" / path).exists()]
    assert not missing, f"zensical.toml nav references missing docs: {missing}"

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'Documentation = "{site_url.rstrip("/")}/"' in pyproject, (
        "pyproject.toml Documentation URL must match zensical site_url"
    )
