import os

from torchfits.cache import (
    CacheConfig,
    get_optimal_cache_config,
)


def _mock_sysconf_with_memory(monkeypatch, memory_gb: float):
    total_bytes = int(memory_gb * (1024**3))
    page_size = 4096
    phys_pages = total_bytes // page_size

    def sysconf_mock(name):
        if name == "SC_PAGE_SIZE":
            return page_size
        if name == "SC_PHYS_PAGES":
            return phys_pages
        raise ValueError(f"Unknown sysconf: {name}")

    monkeypatch.setattr(os, "sysconf", sysconf_mock)


def test_optimal_cache_config_no_psutil(monkeypatch):
    """Test fallback when sysconf raises an error."""

    def sysconf_fail(name):
        raise ValueError("Simulated sysconf error")

    monkeypatch.setattr(os, "sysconf", sysconf_fail)

    monkeypatch.setattr(CacheConfig, "_is_gpu_environment", lambda: False)

    config = get_optimal_cache_config()

    assert config["max_files"] == 100
    assert config["max_memory_mb"] == 1024
    assert config["disk_cache_gb"] == 5
    assert config["prefetch_enabled"] is False
    assert config["environment"] == "local"


def test_optimal_cache_config_local(monkeypatch):
    """Test default local environment."""
    _mock_sysconf_with_memory(monkeypatch, 16.0)

    monkeypatch.setattr(CacheConfig, "_is_gpu_environment", lambda: False)

    config = get_optimal_cache_config()

    assert config["environment"] == "local"
    # 10% of 16GB = 1.6GB = 1638.4 MB -> int(1638.4) = 1638
    assert config["max_memory_mb"] == 1638
    assert config["disk_cache_gb"] == 5


def test_optimal_cache_config_hpc(monkeypatch):
    """Test HPC environment detection."""
    _mock_sysconf_with_memory(monkeypatch, 64.0)

    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    config = get_optimal_cache_config()

    assert config["environment"] == "hpc"
    # 30% of 64GB = 19.2GB = 19660.8 MB -> int() = 19660
    assert config["max_memory_mb"] == 19660
    assert config["max_files"] == 1000
    assert config["disk_cache_gb"] == 50
    assert config["prefetch_enabled"] is True


def test_optimal_cache_config_cloud(monkeypatch):
    """Test cloud environment detection."""
    _mock_sysconf_with_memory(monkeypatch, 16.0)

    monkeypatch.setenv("AWS_EXECUTION_ENV", "lambda")

    config = get_optimal_cache_config()

    assert config["environment"] == "cloud"
    # 20% of 16GB = 3.2GB = 3276.8 MB -> int() = 3276
    assert config["max_memory_mb"] == 3276
    assert config["max_files"] == 500
    assert config["disk_cache_gb"] == 20
    assert config["prefetch_enabled"] is True


def test_optimal_cache_config_gpu(monkeypatch):
    """Test GPU workstation detection."""
    _mock_sysconf_with_memory(monkeypatch, 32.0)

    monkeypatch.setattr(CacheConfig, "_is_gpu_environment", lambda: True)

    config = get_optimal_cache_config()

    assert config["environment"] == "gpu_workstation"
    # 40% of 32GB = 12.8GB = 13107.2 MB -> int() = 13107
    assert config["max_memory_mb"] == 13107
    assert config["max_files"] == 200
    assert config["disk_cache_gb"] == 30
    assert config["prefetch_enabled"] is True
