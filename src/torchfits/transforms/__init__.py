"""Machine-learning friendly transformations for FITS images and high-DR data.

All transforms implement the :class:`FITSTransform` callable protocol
(``forward`` / ``inverse`` / ``__call__``). They are **not**
``torch.nn.Module`` subclasses.

Inverse state is **instance-local**. Construct one pipeline per DataLoader
worker when ``num_workers > 0``.

Import from this package only::

    from torchfits.transforms import ArcsinhStretch, Compose, ZScaleNormalize
"""

from __future__ import annotations

from .base import Compose, FITSTransform
from .clip import AsymmetricSigmaClip, SigmaClip
from .continuum import (
    AlphaShapeContinuum,
    AsymmetricLeastSquares,
    RunningPercentile,
    SavitzkyGolayFilter,
    UpperEnvelopeContinuum,
    WaveletDecompose,
)
from .fits_meta import (
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSScaleColumns,
    TNullToNan,
)
from .helpers import estimate_background, safe_arcsinh, safe_log, zscale_limits
from .normalize import (
    BackgroundSubtract,
    GlobalScalarNorm,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    ZScaleNormalize,
)
from .spectral import (
    BandMath,
    ContinuumNormalize,
    ContinuumRemoval,
    DopplerShift,
    PhaseFold,
    SpectralBinning,
)
from .stretch import ArcsinhStretch, LogStretch, SqrtStretch

__all__ = [
    "FITSTransform",
    "Compose",
    "ArcsinhStretch",
    "LogStretch",
    "SqrtStretch",
    "ZScaleNormalize",
    "RobustNormalize",
    "BackgroundSubtract",
    "PercentileClipNormalize",
    "MinMaxNormalize",
    "FITSHeaderScale",
    "FITSScaleColumns",
    "TNullToNan",
    "ContinuumNormalize",
    "DopplerShift",
    "PhaseFold",
    "SpectralBinning",
    "ContinuumRemoval",
    "BandMath",
    "GlobalScalarNorm",
    "SavitzkyGolayFilter",
    "RunningPercentile",
    "UpperEnvelopeContinuum",
    "WaveletDecompose",
    "AsymmetricLeastSquares",
    "AlphaShapeContinuum",
    "SigmaClip",
    "AsymmetricSigmaClip",
    "FITSHeaderNormalize",
    "safe_arcsinh",
    "safe_log",
    "estimate_background",
    "zscale_limits",
]
