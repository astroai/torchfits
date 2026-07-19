"""VOSpace / vault URI helpers (no Dataset imports)."""

from __future__ import annotations

_VAULT_ROOT = "vos://cadc.nrc.ca~vault/"


def is_vos_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.startswith("vos://")
        or lowered.startswith("vos:")
        or lowered.startswith("vault:")
    )


def normalize_vos_uri(path: str) -> str:
    """Map short ``vos:<user>/...`` / ``vault:...`` to a full vos URI.

    Full ``vos://...`` URIs are returned unchanged.
    """
    if path.lower().startswith("vault:"):
        rest = path.split(":", 1)[1].lstrip("/")
        return f"{_VAULT_ROOT}{rest}"
    if path.lower().startswith("vos://"):
        return path
    if path.lower().startswith("vos:"):
        rest = path.split(":", 1)[1].lstrip("/")
        if rest.startswith("//"):
            return f"vos:{rest}"
        return f"{_VAULT_ROOT}{rest}"
    raise ValueError(f"not a vos/vault path: {path}")
