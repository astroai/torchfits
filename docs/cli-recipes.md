# CLI recipes

Shell workflows inspired by IRAF / CFITSIO / astropy FITS tutorials.
Samples come from `examples/_sample_data.py` (cached under
`~/.cache/torchfits/samples/`).

```bash
pixi run python -c "from examples._sample_data import ensure_sample; print(ensure_sample('horsehead'))"
export HH=~/.cache/torchfits/samples/horsehead.fits
```

Or run the bundled script:

```bash
bash examples/cli/imstat_imarith.sh
```

---

## Image inventory and stats (`imstat` / `fitsinfo`)

```bash
torchfits info "$HH"
torchfits stats "$HH" --hdu 0
torchfits header "$HH" --keyword OBJECT --keyword BITPIX
```

## Header edit (`hedit` / `modhead`)

```bash
torchfits copy "$HH" /tmp/hh_edit.fits
torchfits setkey /tmp/hh_edit.fits --key OBJECT --value HORSEHEAD
torchfits header /tmp/hh_edit.fits --keyword OBJECT
```

## Multi-file keyword table (`dfits | fitsort`)

```bash
torchfits header "$HH" /tmp/hh_edit.fits --fitsort --keyword OBJECT --keyword NAXIS1
```

## Constant arith (`imarith`)

```bash
torchfits arith "$HH" --op add --value 100 --out /tmp/hh_plus.fits
torchfits stats /tmp/hh_plus.fits --hdu 0
```

## Cutout / copy (`imcopy` section)

```bash
torchfits cutout "$HH" /tmp/hh_cut.fits --hdu 0 --box 100,100,256,256
torchfits copy "$HH" /tmp/hh_copy.fits
```

## Stretch via transform (`imfunction`-ish)

Default constructors only (no kwargs on the CLI):

```bash
torchfits transform "$HH" --name LogStretch --out /tmp/hh_log.fits
torchfits transform "$HH" --name SqrtStretch --out /tmp/hh_sqrt.fits
torchfits transform "$HH" --name ZScaleNormalize --out /tmp/hh_z.fits
```

Continuum / Doppler with parameters → Python gallery
([Transform gallery](examples-transforms.md)).

## Compression (`fpack` / `funpack`)

```bash
torchfits compress "$HH" /tmp/hh_packed.fits
torchfits decompress /tmp/hh_packed.fits /tmp/hh_unpacked.fits
```

## Verify checksums

```bash
torchfits verify "$HH"
```

## Tables (`astconvertt` / `tablist`)

```bash
pixi run python -c "from examples._sample_data import ensure_sample; print(ensure_sample('chandra_events'))"
export EV=~/.cache/torchfits/samples/chandra_events.fits
torchfits info "$EV"
torchfits table "$EV" --hdu 1 --preview 5
torchfits convert "$EV" /tmp/events.parquet --to parquet --hdu 1
torchfits convert "$EV" /tmp/events.csv --to csv --hdu 1
```

## Static RGB preview (Imviz-style export)

```bash
torchfits convert "$HH" /tmp/hh.png --to png --bands 0,0,0
```

---

## Familiar-tool map (extended)

| Classic | torchfits |
|---------|-----------|
| `imstat` | `stats` |
| `imarith` (constant) | `arith` |
| `imcopy` | `copy` / `cutout` |
| `imheader` / `hedit` | `header` / `setkey` |
| `hselect` / `fitsort` | `header --fitsort` |
| `imfunction` | `transform --name …` |
| `fitsinfo` | `info` |
| `fitsverify` | `verify` |
| `fpack` / `funpack` | `compress` / `decompress` |
| `astconvertt` | `convert --to parquet\|csv\|tsv\|arrow` |

Full flags: [CLI guide](cli.md).
