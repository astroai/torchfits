"""``torchfits compress`` / ``decompress`` — tile-compressed FITS I/O."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torchfits
from torchfits._io_engine._tile_compression import (
    QUANTIZE_METHODS,
    compression_options,
    resolve_tile_dims,
)
from torchfits._io_engine.paths import cfitsio_base_path, has_bz2_support

from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_file_jobs_arg,
    add_hdu_arg,
    add_jobs_arg,
    add_split_arg,
    configure_torch_jobs,
    ensure_unique_basenames,
    _hdu_width,
    _header_truthy,
    ensure_unique_split_stems,
    hdu_type_name,
    is_compressed_image_header,
    reject_same_path,
    staged_output,
    resolve_batch_io_pairs,
    resolve_file_jobs,
    run_file_jobs,
    selected_hdu_indices,
)

_CODECS = ("GZIP_1", "GZIP_2", "RICE_1", "HCOMPRESS_1", "PLIO_1", "BZIP2_1")


def _available_codecs() -> tuple[str, ...]:
    return _CODECS if has_bz2_support() else _CODECS[:-1]


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths",
        nargs="+",
        help="INPUT [OUTPUT], or multiple INPUTs with --out-dir",
    )
    parser.add_argument("-o", "--out", default=None, help="output FITS path")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for outputs (required for multiple inputs or --split hdu)",
    )
    add_split_arg(parser)
    add_hdu_arg(
        parser,
        help="comma-separated HDU indices (default: all; with --split hdu)",
    )
    add_jobs_arg(parser)
    add_file_jobs_arg(parser)


def add_compress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "compress",
        help="write tile-compressed FITS",
        description=(
            "Tile-compress image HDUs (lossless GZIP_1 by default). "
            "-j = PyTorch intra-op threads; -J = parallel file workers. "
            "Use --split hdu --out-dir for one file per image HDU; "
            "RICE_1, GZIP_2, HCOMPRESS_1, and BZIP2_1 are lossy for floats; "
            "PLIO_1 rejects floats."
        ),
    )
    _add_shared_args(parser)
    parser.add_argument(
        "--algorithm",
        choices=_available_codecs(),
        default="GZIP_1",
        help=(
            "codec (default: GZIP_1, lossless for integers and floats); "
            "other codecs use CFITSIO float quantization and are lossy"
        ),
    )
    parser.add_argument(
        "-t",
        "--tile",
        default=None,
        metavar="N1,N2,...",
        help=(
            "tile size per axis in FITS order (NAXIS1 first), like fpack -t; "
            "omitted axes get 1, -1 = whole axis (write --tile=-1,-1). "
            "Default: one image row per tile (16 rows for HCOMPRESS_1)"
        ),
    )
    parser.add_argument(
        "-q",
        "--quantize-level",
        type=float,
        default=None,
        metavar="Q",
        help=(
            "float quantization, like fpack -q: Q > 0 = noise sigma / Q step, "
            "Q < 0 = absolute step |Q|, 0 = lossless (GZIP_1/GZIP_2 only). "
            "Default: lossless for GZIP_1, 4 otherwise"
        ),
    )
    parser.add_argument(
        "--quantize-method",
        choices=sorted(QUANTIZE_METHODS),
        default=None,
        help="float quantization dithering (default: subtractive_dither_1)",
    )
    parser.add_argument(
        "--dither-seed",
        type=int,
        default=None,
        metavar="N",
        help="dither seed 1..10000 for reproducible output (default: from the clock)",
    )
    parser.add_argument(
        "--int-to-float",
        action="store_true",
        help=(
            "compress integer images as quantized floats, like fpack -i2f "
            "(lossy; output images become BITPIX=-32; default level 4)"
        ),
    )
    parser.set_defaults(func=run_compress)


def add_decompress_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "decompress",
        help="write uncompressed FITS",
        description=(
            "Expand tile-compressed image HDUs. "
            "-j = PyTorch intra-op threads; -J = parallel file workers. "
            "Use --split hdu --out-dir for one file per image HDU."
        ),
    )
    _add_shared_args(parser)
    parser.set_defaults(func=run_decompress)


def _resolve_file_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    # Shared resolver in common.py owns the INPUT [OUTPUT] / -o / --out-dir
    # matrix (including basename collision checks and CFITSIO-section naming).
    return resolve_batch_io_pairs(
        [str(p) for p in args.paths],
        out=args.out,
        out_dir=args.out_dir,
        refuse_same_path=True,
    )


# Cards a tile-compressed HDU uses to remember the uncompressed HDU's
# required keywords (CFITSIO imcomp_copy_comp2img). They mean nothing in an
# uncompressed HDU and must not leak into decompressed output.
_RESTORE_CARDS = (
    "ZSIMPLE",
    "ZTENSION",
    "ZEXTEND",
    "ZBLOCKED",
    "ZPCOUNT",
    "ZGCOUNT",
    "ZHECKSUM",
    "ZDATASUM",
)


def _naxis_has_data(header: Any) -> bool:
    try:
        naxis = int(header.get("NAXIS", 0))
        return naxis > 0 and all(
            int(header.get(f"NAXIS{axis}", 0)) > 0 for axis in range(1, naxis + 1)
        )
    except (TypeError, ValueError):
        return False


def _refresh_checksums(path: str, hdu: int) -> None:
    header = torchfits.read_header(path, hdu)
    if "CHECKSUM" in header or "DATASUM" in header:
        torchfits.write_checksums(path, hdu)
        torchfits.io._invalidate_path_caches(path)


_QUANTIZE_OPTIONS = ("quantize_level", "quantize_method", "dither_seed")


def _cli_option_names(message: str) -> str:
    """Spell library keyword names in a validation message as CLI flags."""
    for key in ("tile_shape", "int_to_float", *_QUANTIZE_OPTIONS):
        flag = "--tile" if key == "tile_shape" else "--" + key.replace("_", "-")
        message = message.replace(f"{key}=", f"{flag} ").replace(key, flag)
    return message.replace("compress=", "--algorithm ")


def _parse_tile(text: str | None) -> tuple[int, ...] | None:
    """``--tile N1,N2,...`` (FITS axis order) as a tensor-order tile_shape."""
    if text is None:
        return None
    try:
        dims = [int(part) for part in str(text).split(",")]
    except ValueError:
        raise UsageError(
            f"invalid --tile {text!r}: expected comma-separated integers, "
            "e.g. --tile 100,100"
        ) from None
    if any(size == 0 or size < -1 for size in dims):
        raise UsageError(
            f"invalid --tile {text!r}: tile sizes must be positive (-1 = whole axis)"
        )
    # FITS order (NAXIS1 = fastest axis first) -> tensor order.
    return tuple(reversed(dims))


def _compress_file(
    input_path: str, output_path: str, *, algorithm: str, options: dict[str, Any]
) -> None:
    # Write and post-edit a temp file; publish only once it is complete.
    with staged_output(output_path) as temp_path:
        _compress_to(input_path, temp_path, algorithm=algorithm, options=options)


def _compress_to(
    input_path: str, output_path: str, *, algorithm: str, options: dict[str, Any]
) -> None:
    with torchfits.open(input_path) as hdul:
        primary_image = len(hdul) > 0 and _naxis_has_data(hdul[0].header)
        torchfits.write(
            output_path, hdul, overwrite=True, compress=algorithm, **options
        )
    if not primary_image:
        return
    # A FITS primary cannot be compressed: the image moves to HDU 1 behind an
    # empty primary. Record that with ZSIMPLE (as fpack / CFITSIO do) so
    # decompress can put it back as the primary.
    with torchfits.open(output_path) as out:
        moved = (
            len(out) > 1
            and not _naxis_has_data(out[0].header)
            and is_compressed_image_header(out[1].header)
        )
    if moved:
        torchfits.io._write_header_cards_if_supported(
            output_path, 1, {"ZSIMPLE": (True, "file does conform to FITS standard")}
        )
        _refresh_checksums(output_path, 1)


def _decompress_file(input_path: str, output_path: str) -> None:
    # Write and post-edit a temp file; publish only once it is complete.
    with staged_output(output_path) as temp_path:
        _decompress_to(input_path, temp_path)


def _decompress_to(input_path: str, output_path: str) -> None:
    with torchfits.open(input_path) as hdul:
        hdus = [hdul[index] for index in range(len(hdul))]
        compressed = [is_compressed_image_header(hdu.header) for hdu in hdus]
        # funpack layout restore: a compressed HDU 1 flagged ZSIMPLE behind an
        # empty primary was the original primary image.
        restore_primary = (
            len(hdus) > 1
            and compressed[1]
            and not _naxis_has_data(hdus[0].header)
            and _header_truthy(hdus[1].header.get("ZSIMPLE"))
        )
        if restore_primary:
            hdus = hdus[1:]
            compressed = compressed[1:]
        torchfits.write(
            output_path, torchfits.HDUList(hdus), overwrite=True, compress=False
        )
    for index, was_compressed in enumerate(compressed):
        if not was_compressed:
            continue
        header = torchfits.read_header(output_path, index)
        leaked = [key for key in _RESTORE_CARDS if key in header]
        for key in leaked:
            torchfits.io._delete_header_key_if_supported(output_path, index, key)
        if leaked:
            _refresh_checksums(output_path, index)


def _rewrite_file(
    input_path: str,
    output_path: str,
    *,
    compress: bool | str,
    options: dict[str, Any],
) -> None:
    try:
        if isinstance(compress, str):
            _compress_file(input_path, output_path, algorithm=compress, options=options)
        else:
            _decompress_file(input_path, output_path)
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{input_path}: {exc}") from exc


def _options_for_image(options: dict[str, Any], tensor: Any) -> dict[str, Any]:
    """Options for one image HDU written on its own (``--split hdu``).

    Integer images are compressed losslessly (unless --int-to-float); the
    per-file check already made sure the quantize options reach a
    floating-point image.
    """
    if tensor.is_floating_point() or options.get("int_to_float"):
        return options
    return {k: v for k, v in options.items() if k not in _QUANTIZE_OPTIONS}


def _hdu_output_path(
    out_dir: Path, input_path: str, hdu_index: int, *, width: int
) -> str:
    stem = Path(cfitsio_base_path(input_path)).stem
    return str(out_dir / f"{stem}_hdu{hdu_index:0{width}d}.fits")


def _rewrite_one_input_split_hdu(
    input_path: str,
    out_dir: Path,
    hdu: str | None,
    *,
    compress: bool | str,
    options: dict[str, Any],
    batch_inputs: tuple[str, ...] = (),
) -> int:
    """Compress/decompress image HDUs from one input; return count written."""
    import torch

    try:
        with torchfits.open(input_path) as hdul:
            indices = selected_hdu_indices(len(hdul), hdu, path=input_path)
            image_indices = [
                index
                for index in indices
                if hdu_type_name(hdul[index].header, hdul[index]) == "IMAGE"
            ]
    except UsageError:
        raise
    except Exception as exc:
        raise IoError(f"{input_path}: {exc}") from exc

    width = _hdu_width(image_indices)
    written = 0
    for index in image_indices:
        try:
            tensor = torchfits.read_tensor(input_path, hdu=index)
            if not isinstance(tensor, torch.Tensor):
                raise IoError(
                    f"{input_path}:{index} read_tensor did not return a tensor"
                )
            header = torchfits.read_header(input_path, index)
            output_path = _hdu_output_path(out_dir, input_path, index, width=width)
            # A generated split name may coincide with another input of the
            # batch (re-splitting into the same directory): refuse instead of
            # silently clobbering an unrelated input.
            for other in batch_inputs:
                reject_same_path(other, output_path)
            torchfits.write(
                output_path,
                tensor,
                header=header,
                overwrite=True,
                compress=compress,
                **_options_for_image(options, tensor),
            )
            written += 1
        except UsageError:
            raise
        except IoError:
            raise
        except Exception as exc:
            raise IoError(f"{input_path}:{index}: {exc}") from exc
    return written


def _rewrite_split_hdu(
    args: argparse.Namespace, *, compress: bool | str, options: dict[str, Any]
) -> None:
    if not args.out_dir:
        raise UsageError("--split hdu requires --out-dir")
    if args.out:
        raise UsageError("--split hdu uses --out-dir (not -o/--out)")

    out_dir = Path(args.out_dir)
    inputs = [str(p) for p in args.paths]
    ensure_unique_basenames(inputs, directory=out_dir)
    ensure_unique_split_stems(inputs, directory=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(inputs))
    configure_torch_jobs(int(args.jobs))
    counts = run_file_jobs(
        inputs,
        lambda path: _rewrite_one_input_split_hdu(
            path,
            out_dir,
            args.hdu,
            compress=compress,
            options=options,
            batch_inputs=tuple(inputs),
        ),
        file_jobs,
        torch_runtime=True,
    )
    if sum(counts) == 0:
        raise IoError("no image HDUs to process")


def _image_shape(header: Any) -> tuple[int, ...]:
    """Image shape (tensor order) from a plain or tile-compressed header."""
    prefix = "ZNAXIS" if is_compressed_image_header(header) else "NAXIS"
    naxis = int(header.get(prefix, 0) or 0)
    return tuple(
        int(header.get(f"{prefix}{axis}", 0) or 0) for axis in range(naxis, 0, -1)
    )


# BZERO of the integer conventions torchfits reads as an integer dtype.
_INTEGER_BZERO = {8: -128.0, 16: 32768.0, 32: 2147483648.0, 64: 9223372036854775808.0}


def _reads_as_float(header: Any, bitpix: int) -> bool:
    """True when the image reads (and so recompresses) as floating point."""
    if bitpix < 0:
        return True
    try:
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
    except (TypeError, ValueError):
        return False
    return bscale != 1.0 or bzero not in (0.0, _INTEGER_BZERO.get(bitpix))


def _validate_compression_inputs(
    inputs: list[str], algorithm: str, hdu: str | None, options: dict[str, Any]
) -> None:
    if algorithm == "BZIP2_1" and not has_bz2_support():
        raise UsageError("BZIP2_1 is unavailable in this torchfits build")
    tile_shape = options.get("tile_shape")
    quantize = any(key in options for key in _QUANTIZE_OPTIONS)
    int_to_float = bool(options.get("int_to_float"))
    if algorithm != "PLIO_1" and tile_shape is None and not quantize:
        return
    for path in inputs:
        has_float = False
        try:
            with torchfits.open(path) as hdul:
                for index in selected_hdu_indices(len(hdul), hdu, path=path):
                    header = hdul[index].header
                    if hdu_type_name(header, hdul[index]) != "IMAGE":
                        continue
                    shape = _image_shape(header)
                    if not shape or 0 in shape:
                        continue  # empty primary: nothing to compress
                    bitpix = int(header.get("ZBITPIX", header.get("BITPIX", 0)) or 0)
                    if algorithm == "PLIO_1" and bitpix < 0:
                        raise UsageError(
                            f"PLIO_1 does not support floating-point image data: {path}:{index}"
                        )
                    # --int-to-float makes every image HDU a float one.
                    has_float = (
                        has_float or int_to_float or _reads_as_float(header, bitpix)
                    )
                    if tile_shape is not None:
                        try:
                            resolve_tile_dims(tile_shape, shape, algorithm, index)
                        except ValueError as exc:
                            raise UsageError(f"--tile: {path}: {exc}") from None
        except UsageError:
            raise
        except Exception as exc:
            raise IoError(f"{path}: {exc}") from exc
        if quantize and not has_float:
            raise UsageError(
                f"{path}: --quantize-level/--quantize-method/--dither-seed apply "
                "to floating-point images, but every selected image HDU is "
                "integer (integer images are compressed losslessly unless "
                "--int-to-float)"
            )


def _run_rewrite(
    args: argparse.Namespace,
    *,
    compress: bool | str,
    options: dict[str, Any] | None = None,
) -> int:
    options = options or {}
    split = getattr(args, "split", "file")
    if split == "file" and args.hdu is not None:
        raise UsageError("--hdu applies only with --split hdu")
    if split == "hdu":
        if isinstance(compress, str):
            _validate_compression_inputs(
                [str(path) for path in args.paths], compress, args.hdu, options
            )
        _rewrite_split_hdu(args, compress=compress, options=options)
        return EXIT_OK

    pairs = _resolve_file_pairs(args)
    if isinstance(compress, str):
        _validate_compression_inputs(
            [pair[0] for pair in pairs], compress, None, options
        )
    file_jobs = resolve_file_jobs(int(args.file_jobs), len(pairs))
    configure_torch_jobs(int(args.jobs))
    run_file_jobs(
        pairs,
        lambda pair: _rewrite_file(
            pair[0], pair[1], compress=compress, options=options
        ),
        file_jobs,
        torch_runtime=True,
    )
    return EXIT_OK


def run_compress(args: argparse.Namespace) -> int:
    algo = str(getattr(args, "algorithm", "GZIP_1") or "GZIP_1").strip().upper()
    if algo not in _available_codecs():
        raise UsageError(
            f"invalid --algorithm {algo!r}; choose from {', '.join(_available_codecs())}"
        )
    options: dict[str, Any] = {}
    tile_shape = _parse_tile(getattr(args, "tile", None))
    if tile_shape is not None:
        options["tile_shape"] = tile_shape
    for key in _QUANTIZE_OPTIONS:
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if getattr(args, "int_to_float", False):
        options["int_to_float"] = True
    try:
        compression_options(algo, **options)
    except (TypeError, ValueError) as exc:
        raise UsageError(_cli_option_names(str(exc))) from None
    return _run_rewrite(args, compress=algo, options=options)


def run_decompress(args: argparse.Namespace) -> int:
    return _run_rewrite(args, compress=False)
