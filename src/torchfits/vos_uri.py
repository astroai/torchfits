"""VOSpace / vault URI helpers (no Dataset imports)."""

from __future__ import annotations

_VAULT_ROOT = "vos://cadc.nrc.ca~vault/"


def is_vos_path(path: str) -> bool:
    """True for a VOSpace/vault URI: ``vos://``, ``vos:`` or ``vault:`` followed
    by a non-empty path, with no embedded whitespace."""
    if (
        not isinstance(path, str)
        or path != path.strip()
        or any(c.isspace() for c in path)
    ):
        return False
    lowered = path.lower()
    for prefix in ("vos://", "vos:", "vault:"):
        if lowered.startswith(prefix):
            return len(path) > len(prefix)
    return False


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
