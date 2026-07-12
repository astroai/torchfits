"""
Cloud/HPC optimized caching for torchfits.

This module provides intelligent caching strategies for different environments
including local development, HPC clusters, and cloud platforms.
"""

import os
from typing import Any, Dict, Optional


# Single source of truth for env vars that trigger environment classification in
# CacheConfig._is_hpc_environment / _is_cloud_environment. Tests, docs, and any
# downstream tooling should import these tuples rather than hard-code the lists, so a new
# sentinel added here propagates everywhere automatically.
HPC_ENV_SENTINELS: tuple[str, ...] = (
    "SLURM_JOB_ID",
    "PBS_JOBID",
    "LSB_JOBID",
    "SGE_JOB_ID",
)
CLOUD_ENV_SENTINELS: tuple[str, ...] = (
    # AWS / GCP / Azure
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_NAME",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "AZURE_FUNCTIONS_ENVIRONMENT",
    # Web / PaaS / k8s
    "WEBSITE_SITE_NAME",
    "KUBERNETES_SERVICE_HOST",
    "K_SERVICE",
)
# Convenience union used by tests to clear every classification variable at once.
CACHE_ENV_SENTINELS: tuple[str, ...] = HPC_ENV_SENTINELS + CLOUD_ENV_SENTINELS


class CacheConfig:
    """Cache configuration for different environments."""

    # NOTE: torch.cuda.is_available() is already cached internally by PyTorch
    # after the first call, and os.sysconf() is a cheap syscall.  Env-var
    # checks are cheap dict lookups.  Therefore we do NOT add a separate
    # cache layer here — it would only break test mocking of these methods.
    # The structural improvement is extracting detection into named methods
    # for clarity and testability.

    def __init__(
        self,
        max_files: int = 100,
        max_memory_mb: int = 1024,
        disk_cache_gb: int = 10,
        prefetch_enabled: bool = True,
    ):
        self.max_files = max_files
        self.max_memory_mb = max_memory_mb
        self.disk_cache_gb = disk_cache_gb
        self.prefetch_enabled = prefetch_enabled

    @staticmethod
    def _detect_gpu() -> bool:
        """Detect GPU availability (delegates to torch, which caches internally)."""
        try:
            import torch

            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False

    @staticmethod
    def _detect_memory_gb() -> float:
        """Detect system memory in GB via POSIX sysconf."""
        try:
            pagesize = os.sysconf("SC_PAGE_SIZE")
            physpages = os.sysconf("SC_PHYS_PAGES")
            if pagesize > 0 and physpages > 0:
                return (pagesize * physpages) / (1024**3)
        except Exception:
            pass
        return 10.0

    @classmethod
    def for_environment(cls) -> "CacheConfig":
        """Auto-detect optimal cache configuration."""
        memory_gb = cls._detect_memory_gb()

        # Detect environment using the static methods (test-patchable)
        if cls._is_hpc_environment():
            # HPC: Large memory, shared filesystem
            return cls(
                max_files=1000,
                max_memory_mb=int(memory_gb * 1024 * 0.3),  # 30% of memory
                disk_cache_gb=50,
                prefetch_enabled=True,
            )
        elif cls._is_cloud_environment():
            # Cloud: Variable memory, network storage
            return cls(
                max_files=500,
                max_memory_mb=int(memory_gb * 1024 * 0.2),  # 20% of memory
                disk_cache_gb=20,
                prefetch_enabled=True,
            )
        elif cls._is_gpu_environment():
            # GPU workstation: High memory, fast local storage
            return cls(
                max_files=200,
                max_memory_mb=int(memory_gb * 1024 * 0.4),  # 40% of memory
                disk_cache_gb=30,
                prefetch_enabled=True,
            )
        else:
            # Default: Conservative settings
            return cls(
                max_files=100,
                max_memory_mb=min(2048, int(memory_gb * 1024 * 0.1)),
                disk_cache_gb=5,
                prefetch_enabled=False,
            )

    @staticmethod
    def _is_hpc_environment() -> bool:
        """Detect HPC batch system environment."""
        return any(var in os.environ for var in HPC_ENV_SENTINELS)

    @staticmethod
    def _is_cloud_environment() -> bool:
        """Detect cloud platform environment."""
        return any(var in os.environ for var in CLOUD_ENV_SENTINELS)

    @staticmethod
    def _is_gpu_environment() -> bool:
        """Detect GPU environment."""
        return CacheConfig._detect_gpu()


class CacheManager:
    """Advanced cache manager with multiple strategies."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig.for_environment()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_usage_mb": 0,
            "disk_usage_gb": 0,
        }

    def configure_cpp_cache(self):
        """Configure the C++ cache backend."""
        try:
            import torchfits._C as cpp

            if hasattr(cpp, "configure_cache"):
                cpp.configure_cache(self.config.max_files, self.config.max_memory_mb)
        except (ImportError, AttributeError):
            # Fallback when C++ module not available or function missing
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics, including I/O engine stats."""
        try:
            import torchfits._C as cpp

            cpp_size = cpp.get_cache_size() if hasattr(cpp, "get_cache_size") else 0
        except (ImportError, AttributeError):
            cpp_size = 0

        # Merge I/O engine cache statistics when available
        io_stats: Dict[str, Any] = {}
        try:
            from ._io_engine.caches import get_cache_performance

            io_stats = get_cache_performance()
        except Exception:
            pass

        merged_hits = self._stats["hits"] + io_stats.get("hits", 0)
        merged_misses = self._stats["misses"] + io_stats.get("misses", 0)
        merged_total = merged_hits + merged_misses

        return {
            **self._stats,
            "io_hits": io_stats.get("hits", 0),
            "io_misses": io_stats.get("misses", 0),
            "io_total_requests": io_stats.get("total_requests", 0),
            "cpp_cache_size": cpp_size,
            "config": {
                "max_files": self.config.max_files,
                "max_memory_mb": self.config.max_memory_mb,
                "disk_cache_gb": self.config.disk_cache_gb,
                "prefetch_enabled": self.config.prefetch_enabled,
            },
            "hit_rate": merged_hits / max(1, merged_total),
        }

    def clear(self):
        """Clear all caches."""
        try:
            import torchfits._C as cpp

            if hasattr(cpp, "clear_file_cache"):
                cpp.clear_file_cache()
        except (ImportError, AttributeError):
            pass

        self._stats = {key: 0 for key in self._stats}

    def optimize_for_dataset(self, file_paths: list, avg_file_size_mb: float):
        """Optimize cache settings for a specific dataset."""
        total_size_gb = len(file_paths) * avg_file_size_mb / 1024

        # Adjust cache size based on dataset
        if total_size_gb < self.config.disk_cache_gb:
            # Dataset fits in cache - enable aggressive caching
            self.config.max_files = len(file_paths)
            self.config.prefetch_enabled = True
        else:
            # Large dataset - use LRU strategy
            optimal_files = int(self.config.disk_cache_gb * 1024 / avg_file_size_mb)
            self.config.max_files = min(optimal_files, 1000)

        self.configure_cpp_cache()


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        _cache_manager.configure_cpp_cache()
    return _cache_manager


def configure_for_environment():
    """Auto-configure cache for different environments."""
    manager = get_cache_manager()
    manager.configure_cpp_cache()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache_manager().get_stats()


def clear_cache():
    """Clear Python, C++ I/O, and table-handle caches."""
    get_cache_manager().clear()
    try:
        from ._io_engine.caches import clear_file_cache

        clear_file_cache()
    except Exception:
        pass
    try:
        from ._table.cache import _close_all_cached_handles

        _close_all_cached_handles()
    except Exception:
        pass


def clear() -> None:
    """Clear all torchfits-managed caches."""
    clear_cache()


def stats() -> Dict[str, Any]:
    """Return cache statistics for the public cache namespace."""
    result = get_cache_stats()
    try:
        from ._io_engine.caches import get_cache_performance

        result = {**result, "io": get_cache_performance()}
    except Exception:
        pass
    return result


def configure_cache(max_files: int, max_memory_mb: int, disk_cache_gb: int = 10):
    """Manually configure cache settings."""
    global _cache_manager
    config = CacheConfig(max_files, max_memory_mb, disk_cache_gb)
    _cache_manager = CacheManager(config)
    _cache_manager.configure_cpp_cache()


def optimize_for_dataset(file_paths: list, avg_file_size_mb: float = 10.0):
    """Optimize cache for a specific dataset."""
    get_cache_manager().optimize_for_dataset(file_paths, avg_file_size_mb)


def get_optimal_cache_config() -> Dict[str, Any]:
    """Get optimal cache configuration for current environment."""
    config = CacheConfig.for_environment()
    return {
        "max_files": config.max_files,
        "max_memory_mb": config.max_memory_mb,
        "disk_cache_gb": config.disk_cache_gb,
        "prefetch_enabled": config.prefetch_enabled,
        "environment": _detect_environment_type(),
    }


def _detect_environment_type() -> str:
    """Detect the type of environment we're running in."""
    if CacheConfig._is_hpc_environment():
        return "hpc"
    elif CacheConfig._is_cloud_environment():
        return "cloud"
    elif CacheConfig._is_gpu_environment():
        return "gpu_workstation"
    else:
        return "local"


__all__ = [
    "CacheConfig",
    "CacheManager",
    "clear",
    "clear_cache",
    "configure_cache",
    "configure_for_environment",
    "get_cache_manager",
    "get_cache_stats",
    "get_optimal_cache_config",
    "optimize_for_dataset",
    "stats",
]
