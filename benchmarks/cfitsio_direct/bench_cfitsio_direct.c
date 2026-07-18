/* Pure-C CFITSIO microbench against vendored libcfitsio.
 *
 * Reads a TSV job manifest and times the best-adapted CFITSIO call path per op:
 *
 *   read_full     fits_read_img (datatype from fits_get_img_equivtype)
 *   cutout        fits_read_subset at fixed 0-based (x0,y0) on given HDU
 *   cutout_rep    open once + N× fits_read_subset from coords file
 *   header_read   fits_get_hdrspace + fits_read_record on given HDU
 *   random_ext    open once + M× (movabs_hdu + fits_read_img); HDU seq =
 *                 ((i*3)%10)+1 in 0-based / astropy index space
 *   table_read    all cols via fits_read_col (ASCII_TBL / VLA / binary)
 *   table_proj    fits_read_col for first K columns only
 *   table_slice   fits_read_col row window (firstrow, nelements)
 *   table_scan    fits_get_num_rows only (header count; no column I/O)
 *   table_pred    fits_read_col one column + compact kept values (value > 0)
 *   table_header  fits_get_hdrspace + fits_read_keyn on first table HDU
 *
 * Job TSV columns (tab-separated, # comments allowed):
 *   case_id  op  path  a  b  c  d  e
 *
 * Op-specific args (HDUs are CFITSIO 1-based absolute):
 *   read_full / header_read   a=hdu
 *   cutout                    a=hx b=hy c=hdu d=x0 e=y0  (0-based origin)
 *   cutout_rep                a=coords_file  (lines: x0 y0 hx hy)
 *   random_ext                a=nreads
 *   table_proj                a=ncols
 *   table_slice               a=start_row(1-based) b=nrows
 *   table_scan / table_pred   a=col_name (scan ignores col; kept for TSV shape)
 *   table_read                (no args)
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fitsio.h"

typedef enum {
  OP_READ_FULL = 0,
  OP_CUTOUT,
  OP_CUTOUT_REP,
  OP_HEADER,
  OP_RANDOM_EXT,
  OP_TABLE_READ,
  OP_TABLE_PROJ,
  OP_TABLE_SLICE,
  OP_TABLE_SCAN,
  OP_TABLE_PRED,
  OP_TABLE_HEADER,
  OP_UNKNOWN
} op_t;

typedef struct {
  char case_id[256];
  char opname[64];
  char path[1024];
  char a[512], b[128], c[128], d[128], e[128];
  op_t op;
  /* Preloaded cutout_rep coords: [x0,y0,hx,hy] * ncoords (outside timed path). */
  long *coords;
  int ncoords;
} job_t;

static double monotonic_s(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
  }
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
  const double da = *(const double *)a;
  const double db = *(const double *)b;
  return (da > db) - (da < db);
}

static double median_s(double *xs, int n) {
  if (n <= 0) {
    return NAN;
  }
  qsort(xs, (size_t)n, sizeof(double), cmp_double);
  if (n & 1) {
    return xs[n / 2];
  }
  return 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

static op_t parse_op(const char *s) {
  if (strcmp(s, "read_full") == 0) {
    return OP_READ_FULL;
  }
  if (strcmp(s, "cutout") == 0 || strcmp(s, "cutout_100x100") == 0) {
    return OP_CUTOUT;
  }
  if (strcmp(s, "cutout_rep") == 0 || strncmp(s, "repeated_cutouts", 16) == 0) {
    return OP_CUTOUT_REP;
  }
  if (strcmp(s, "header_read") == 0) {
    return OP_HEADER;
  }
  if (strcmp(s, "random_ext") == 0 || strcmp(s, "random_ext_full_reads_200") == 0) {
    return OP_RANDOM_EXT;
  }
  if (strcmp(s, "table_read") == 0 || strcmp(s, "table_read_full") == 0) {
    return OP_TABLE_READ;
  }
  if (strcmp(s, "table_proj") == 0 || strcmp(s, "projection") == 0) {
    return OP_TABLE_PROJ;
  }
  if (strcmp(s, "table_slice") == 0 || strcmp(s, "row_slice") == 0) {
    return OP_TABLE_SLICE;
  }
  if (strcmp(s, "table_scan") == 0 || strcmp(s, "scan_count") == 0) {
    return OP_TABLE_SCAN;
  }
  if (strcmp(s, "table_pred") == 0 || strcmp(s, "predicate_filter") == 0) {
    return OP_TABLE_PRED;
  }
  if (strcmp(s, "table_header") == 0 || strcmp(s, "header_read") == 0) {
    return OP_TABLE_HEADER;
  }
  return OP_UNKNOWN;
}

static const char *op_out_name(const job_t *j) {
  switch (j->op) {
    case OP_READ_FULL:
      return "read_full";
    case OP_CUTOUT:
      return "cutout_100x100";
    case OP_CUTOUT_REP:
      return j->opname[0] ? j->opname : "repeated_cutouts";
    case OP_HEADER:
      return "header_read";
    case OP_RANDOM_EXT:
      return "random_ext_full_reads_200";
    case OP_TABLE_READ:
      return "read_full";
    case OP_TABLE_PROJ:
      return "projection";
    case OP_TABLE_SLICE:
      return "row_slice";
    case OP_TABLE_SCAN:
      return "scan_count";
    case OP_TABLE_PRED:
      return "predicate_filter";
    case OP_TABLE_HEADER:
      return "header_read";
    default:
      return "unknown";
  }
}

static void csv_escape(const char *in, char *out, size_t out_n) {
  if (strchr(in, ',') == NULL && strchr(in, '"') == NULL) {
    snprintf(out, out_n, "%s", in);
    return;
  }
  size_t j = 0;
  if (j + 1 < out_n) {
    out[j++] = '"';
  }
  for (const char *p = in; *p && j + 2 < out_n; ++p) {
    if (*p == '"') {
      out[j++] = '"';
      if (j + 1 >= out_n) {
        break;
      }
    }
    out[j++] = *p;
  }
  if (j + 1 < out_n) {
    out[j++] = '"';
  }
  out[j] = '\0';
}

static int move_hdu(fitsfile *fptr, int hdu_1based, int *status) {
  int hdutype = 0;
  if (hdu_1based < 1) {
    *status = BAD_HDU_NUM;
    return *status;
  }
  return fits_movabs_hdu(fptr, hdu_1based, &hdutype, status);
}

static int move_first_image(fitsfile *fptr, int *status) {
  int nhdu = 0, hdutype = 0;
  if (fits_get_num_hdus(fptr, &nhdu, status)) {
    return *status;
  }
  for (int i = 1; i <= nhdu; ++i) {
    if (fits_movabs_hdu(fptr, i, &hdutype, status)) {
      return *status;
    }
    if (hdutype == IMAGE_HDU) {
      int naxis = 0;
      if (fits_get_img_dim(fptr, &naxis, status)) {
        return *status;
      }
      if (naxis > 0) {
        return 0;
      }
    }
  }
  *status = BAD_HDU_NUM;
  return *status;
}

static int move_first_table(fitsfile *fptr, int *status) {
  int nhdu = 0, hdutype = 0;
  if (fits_get_num_hdus(fptr, &nhdu, status)) {
    return *status;
  }
  for (int i = 1; i <= nhdu; ++i) {
    if (fits_movabs_hdu(fptr, i, &hdutype, status)) {
      return *status;
    }
    if (hdutype == BINARY_TBL || hdutype == ASCII_TBL) {
      return 0;
    }
  }
  *status = NOT_TABLE;
  return *status;
}

/* Map image BITPIX / equivtype → CFITSIO datatype + element size. */
static int bitpix_to_dtype(int bitpix, int *datatype, size_t *elsz) {
  switch (bitpix) {
    case BYTE_IMG:
      *datatype = TBYTE;
      *elsz = 1;
      return 0;
    case SBYTE_IMG:
      *datatype = TSBYTE;
      *elsz = 1;
      return 0;
    case SHORT_IMG:
      *datatype = TSHORT;
      *elsz = 2;
      return 0;
    case USHORT_IMG:
      *datatype = TUSHORT;
      *elsz = 2;
      return 0;
    case LONG_IMG:
      *datatype = TINT;
      *elsz = 4;
      return 0;
    case ULONG_IMG:
      /* CFITSIO TULONG is unsigned long (8B on LP64); use TUINT for 32-bit. */
      *datatype = TUINT;
      *elsz = 4;
      return 0;
    case LONGLONG_IMG:
      *datatype = TLONGLONG;
      *elsz = 8;
      return 0;
    case ULONGLONG_IMG:
      *datatype = TULONGLONG;
      *elsz = 8;
      return 0;
    case FLOAT_IMG:
      *datatype = TFLOAT;
      *elsz = 4;
      return 0;
    case DOUBLE_IMG:
      *datatype = TDOUBLE;
      *elsz = 8;
      return 0;
    default:
      *datatype = TFLOAT;
      *elsz = 4;
      return 0;
  }
}

/* Prefer fits_get_img_equivtype (unsigned + true float scaling). */
static int image_read_dtype(fitsfile *fptr, int *datatype, size_t *elsz, int *status) {
  int bitpix = 0, naxis = 0;
  long naxes[9] = {0};
  if (fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, status)) {
    return *status;
  }
  int equiv = bitpix;
  int est = 0;
  fits_get_img_equivtype(fptr, &equiv, &est);
  if (est == 0) {
    bitpix = equiv;
  }
  (void)naxis;
  return bitpix_to_dtype(bitpix, datatype, elsz);
}

static int img_npix(fitsfile *fptr, long *npix, long naxes[9], int *naxis, int *status) {
  int bitpix = 0;
  if (fits_get_img_param(fptr, 9, &bitpix, naxis, naxes, status)) {
    return *status;
  }
  *npix = 1;
  for (int i = 0; i < *naxis; ++i) {
    *npix *= naxes[i];
  }
  return 0;
}

static int do_cutout_xy(fitsfile *fptr, long x1_0, long y1_0, long hx, long hy, int datatype,
                        size_t elsz, int naxis, long naxes[9], int *status);

static int do_read_full(fitsfile *fptr, int *status) {
  int datatype = 0;
  size_t elsz = 0;
  int naxis = 0;
  long naxes[9] = {0};
  long npix = 0;
  if (image_read_dtype(fptr, &datatype, &elsz, status)) {
    return *status;
  }
  if (img_npix(fptr, &npix, naxes, &naxis, status)) {
    return *status;
  }
  void *buf = malloc((size_t)npix * elsz);
  if (!buf) {
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  int anynul = 0;
  if (fits_read_img(fptr, datatype, 1, npix, NULL, buf, &anynul, status)) {
    free(buf);
    return *status;
  }
  if (npix > 0) {
    volatile unsigned char sink = ((unsigned char *)buf)[0] ^ ((unsigned char *)buf)[npix * (long)elsz - 1];
    (void)sink;
  }
  free(buf);
  return 0;
}

static int do_cutout(fitsfile *fptr, long hx, long hy, long x0, long y0, int *status) {
  int datatype = 0;
  size_t elsz = 0;
  int naxis = 0;
  long naxes[9] = {0};
  if (image_read_dtype(fptr, &datatype, &elsz, status)) {
    return *status;
  }
  int bitpix = 0;
  if (fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, status)) {
    return *status;
  }
  if (naxis < 2) {
    *status = BAD_NAXIS;
    return *status;
  }
  long nx = naxes[0], ny = naxes[1];
  if (hx < 1) {
    hx = 1;
  }
  if (hy < 1) {
    hy = 1;
  }
  if (hx > nx) {
    hx = nx;
  }
  if (hy > ny) {
    hy = ny;
  }
  if (x0 < 0) {
    x0 = (nx - hx) / 2;
  }
  if (y0 < 0) {
    y0 = (ny - hy) / 2;
  }
  if (x0 + hx > nx) {
    x0 = nx - hx;
  }
  if (y0 + hy > ny) {
    y0 = ny - hy;
  }
  return do_cutout_xy(fptr, x0, y0, hx, hy, datatype, elsz, naxis, naxes, status);
}

static int do_cutout_xy(fitsfile *fptr, long x1_0, long y1_0, long hx, long hy, int datatype,
                        size_t elsz, int naxis, long naxes[9], int *status) {
  long blc[9], trc[9], inc[9];
  for (int i = 0; i < naxis; ++i) {
    blc[i] = 1;
    trc[i] = naxes[i];
    inc[i] = 1;
  }
  blc[0] = x1_0 + 1;
  blc[1] = y1_0 + 1;
  trc[0] = blc[0] + hx - 1;
  trc[1] = blc[1] + hy - 1;
  if (trc[0] > naxes[0]) {
    trc[0] = naxes[0];
  }
  if (trc[1] > naxes[1]) {
    trc[1] = naxes[1];
  }
  long npix = (trc[0] - blc[0] + 1) * (trc[1] - blc[1] + 1);
  for (int i = 2; i < naxis; ++i) {
    npix *= naxes[i];
  }
  void *buf = malloc((size_t)npix * elsz);
  if (!buf) {
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  int anynul = 0;
  if (fits_read_subset(fptr, datatype, blc, trc, inc, NULL, buf, &anynul, status)) {
    free(buf);
    return *status;
  }
  if (npix > 0) {
    volatile unsigned char sink = ((unsigned char *)buf)[0];
    (void)sink;
  }
  free(buf);
  return 0;
}

/* Preloaded coords: [x0,y0,hx,hy] * ncoords. */
static int do_cutout_rep(fitsfile *fptr, const long *coords, int ncoords, int *status) {
  int datatype = 0;
  size_t elsz = 0;
  int naxis = 0;
  long naxes[9] = {0};
  if (image_read_dtype(fptr, &datatype, &elsz, status)) {
    return *status;
  }
  int bitpix = 0;
  if (fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, status)) {
    return *status;
  }
  if (naxis < 2) {
    *status = BAD_NAXIS;
    return *status;
  }
  for (int i = 0; i < ncoords; ++i) {
    long x0 = coords[4 * i + 0];
    long y0 = coords[4 * i + 1];
    long hx = coords[4 * i + 2];
    long hy = coords[4 * i + 3];
    if (do_cutout_xy(fptr, x0, y0, hx, hy, datatype, elsz, naxis, naxes, status)) {
      return *status;
    }
  }
  return 0;
}

static int do_header_read(fitsfile *fptr, int *status) {
  /* Parsed key/value/comment path — matches TorchFits/fitsio peers. */
  int nkeys = 0, more = 0;
  if (fits_get_hdrspace(fptr, &nkeys, &more, status)) {
    return *status;
  }
  char key[FLEN_KEYWORD], value[FLEN_VALUE], comment[FLEN_COMMENT];
  volatile int sink = 0;
  for (int i = 1; i <= nkeys; ++i) {
    if (fits_read_keyn(fptr, i, key, value, comment, status)) {
      return *status;
    }
    sink ^= (int)(unsigned char)key[0] ^ (int)(unsigned char)value[0];
  }
  (void)sink;
  return 0;
}

/* Match bench_fits_io: python_hdu = ((i*3)%10)+1; CFITSIO = python_hdu+1. */
static int do_random_ext(fitsfile *fptr, int nreads, int *status) {
  for (int i = 0; i < nreads; ++i) {
    int py_hdu = ((i * 3) % 10) + 1;
    int h = py_hdu + 1;
    if (move_hdu(fptr, h, status)) {
      return *status;
    }
    if (do_read_full(fptr, status)) {
      return *status;
    }
  }
  return 0;
}

static int table_has_vla(fitsfile *fptr, int ncols, int *status) {
  for (int c = 1; c <= ncols; ++c) {
    char tform[FLEN_VALUE] = {0};
    char key[16];
    snprintf(key, sizeof(key), "TFORM%d", c);
    int ks = 0;
    if (fits_read_key(fptr, TSTRING, key, tform, NULL, &ks) == 0) {
      if (strchr(tform, 'P') || strchr(tform, 'Q') || strchr(tform, 'p') || strchr(tform, 'q')) {
        return 1;
      }
    }
    ks = 0;
  }
  return 0;
}

static int do_table_read_tblbytes(fitsfile *fptr, long nrows, int *status) {
  /* Bulk row bytes — best CFITSIO path for fixed-width BINARY_TBL. */
  long naxis1 = 0;
  if (fits_read_key(fptr, TLONG, "NAXIS1", &naxis1, NULL, status)) {
    return *status;
  }
  if (naxis1 <= 0 || nrows <= 0) {
    return 0;
  }
  /* Read in chunks to avoid huge allocs on 1e6×wide tables. */
  const long chunk_rows = 8192;
  unsigned char *buf = (unsigned char *)malloc((size_t)chunk_rows * (size_t)naxis1);
  if (!buf) {
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  volatile unsigned char sink = 0;
  for (long start = 1; start <= nrows; start += chunk_rows) {
    long n = chunk_rows;
    if (start + n - 1 > nrows) {
      n = nrows - start + 1;
    }
    if (fits_read_tblbytes(fptr, start, 1, n * naxis1, buf, status)) {
      free(buf);
      return *status;
    }
    sink ^= buf[0];
  }
  (void)sink;
  free(buf);
  return 0;
}

/* Map CFITSIO col typecode → datatype + element size for fits_read_col.
 * Bool/bit → TBYTE; numeric → native; strings handled separately. */
/* Map TFORM typecode → safe CFITSIO datatype + host element size.
 * TLONG/TULONG are platform-width; force fixed-width TINT/TUINT. */
static int col_read_dtype(int typecode, int *datatype, size_t *elsz) {
  switch (typecode) {
    case TLOGICAL:
      *datatype = TBYTE;
      *elsz = 1;
      return 0;
    case TBYTE:
    case TSBYTE:
      *datatype = typecode;
      *elsz = 1;
      return 0;
    case TSHORT:
    case TUSHORT:
      *datatype = typecode;
      *elsz = 2;
      return 0;
    case TINT:
    case TFLOAT:
      *datatype = typecode;
      *elsz = 4;
      return 0;
    case TLONG:
      *datatype = TINT;
      *elsz = 4;
      return 0;
    case TULONG:
      *datatype = TUINT;
      *elsz = 4;
      return 0;
    case TLONGLONG:
    case TULONGLONG:
    case TDOUBLE:
      *datatype = typecode;
      *elsz = 8;
      return 0;
    case TCOMPLEX:
      *datatype = TCOMPLEX;
      *elsz = 8;
      return 0;
    case TDBLCOMPLEX:
      *datatype = TDBLCOMPLEX;
      *elsz = 16;
      return 0;
    case TSTRING:
    case TBIT:
      return 1; /* handled specially */
    default:
      *datatype = TDOUBLE;
      *elsz = 8;
      return 0;
  }
}

static int col_is_simple_numeric(int typecode) {
  return typecode != TSTRING && typecode != TBIT && typecode != TCOMPLEX &&
         typecode != TDBLCOMPLEX;
}

static int col_is_vla(fitsfile *fptr, int col) {
  char tform[FLEN_VALUE] = {0};
  char key[16];
  snprintf(key, sizeof(key), "TFORM%d", col);
  int ks = 0;
  if (fits_read_key(fptr, TSTRING, key, tform, NULL, &ks) != 0) {
    return 0;
  }
  return strchr(tform, 'P') || strchr(tform, 'Q') || strchr(tform, 'p') || strchr(tform, 'q');
}

static int read_col_range(fitsfile *fptr, int col, long firstrow, long nrows, int *status) {
  int typecode = 0;
  long repeat = 0, width = 0;
  if (fits_get_coltype(fptr, col, &typecode, &repeat, &width, status)) {
    return *status;
  }
  if (nrows <= 0) {
    return 0;
  }

  /* VLA: bulk descriptors + flat native buffer; one fits_read_col per row segment. */
  if (col_is_vla(fptr, col)) {
    int datatype = TDOUBLE;
    size_t elsz = 8;
    if (col_read_dtype(typecode, &datatype, &elsz) != 0) {
      datatype = TDOUBLE;
      elsz = 8;
    }
    LONGLONG *lens = (LONGLONG *)malloc((size_t)nrows * sizeof(LONGLONG));
    LONGLONG *addrs = (LONGLONG *)malloc((size_t)nrows * sizeof(LONGLONG));
    if (!lens || !addrs) {
      free(lens);
      free(addrs);
      *status = MEMORY_ALLOCATION;
      return *status;
    }
    if (fits_read_descriptsll(fptr, col, (LONGLONG)firstrow, (LONGLONG)nrows, lens, addrs,
                              status)) {
      free(lens);
      free(addrs);
      return *status;
    }
    LONGLONG total = 0;
    for (long i = 0; i < nrows; ++i) {
      if (lens[i] > 0) {
        total += lens[i];
      }
    }
    void *flat = total > 0 ? malloc((size_t)total * elsz) : NULL;
    if (total > 0 && !flat) {
      free(lens);
      free(addrs);
      *status = MEMORY_ALLOCATION;
      return *status;
    }
    volatile unsigned char sink = 0;
    char *cursor = (char *)flat;
    for (long i = 0; i < nrows; ++i) {
      if (lens[i] <= 0) {
        continue;
      }
      int anynul = 0;
      if (fits_read_col(fptr, datatype, col, firstrow + i, 1, (long)lens[i], NULL, cursor,
                        &anynul, status)) {
        free(flat);
        free(lens);
        free(addrs);
        return *status;
      }
      sink ^= (unsigned char)cursor[0];
      cursor += (size_t)lens[i] * elsz;
    }
    (void)sink;
    (void)addrs;
    free(flat);
    free(lens);
    free(addrs);
    return 0;
  }

  long rep = repeat > 0 ? repeat : 1;

  if (typecode == TSTRING) {
    long w = width > 0 ? width : 8;
    if (w > 1024) {
      w = 1024;
    }
    char **arr = (char **)malloc((size_t)nrows * sizeof(char *));
    char *blob = (char *)malloc((size_t)nrows * (size_t)(w + 1));
    if (!arr || !blob) {
      free(arr);
      free(blob);
      *status = MEMORY_ALLOCATION;
      return *status;
    }
    for (long i = 0; i < nrows; ++i) {
      arr[i] = blob + i * (w + 1);
      arr[i][0] = '\0';
    }
    int anynul = 0;
    char nulstr[2] = "";
    if (fits_read_col(fptr, TSTRING, col, firstrow, 1, nrows, nulstr, arr, &anynul, status)) {
      free(arr);
      free(blob);
      return *status;
    }
    volatile char sink = arr[0][0];
    (void)sink;
    free(arr);
    free(blob);
    return 0;
  }

  if (typecode == TBIT) {
    /* CFITSIO TBIT → ffgcx writes one char per bit, not packed bytes. */
    long nbits = nrows * rep;
    if (nbits < 1) {
      return 0;
    }
    char *buf = (char *)malloc((size_t)nbits);
    if (!buf) {
      *status = MEMORY_ALLOCATION;
      return *status;
    }
    int anynul = 0;
    if (fits_read_col(fptr, TBIT, col, firstrow, 1, nbits, NULL, buf, &anynul, status)) {
      free(buf);
      return *status;
    }
    volatile char sink = buf[0];
    (void)sink;
    free(buf);
    return 0;
  }

  long nelem = nrows * rep;
  int datatype = 0;
  size_t elsz = 0;
  if (col_read_dtype(typecode, &datatype, &elsz)) {
    *status = BAD_DATATYPE;
    return *status;
  }
  (void)width;
  void *buf = malloc((size_t)nelem * elsz);
  if (!buf) {
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  int anynul = 0;
  if (fits_read_col(fptr, datatype, col, firstrow, 1, nelem, NULL, buf, &anynul, status)) {
    free(buf);
    return *status;
  }
  if (nelem > 0) {
    volatile unsigned char sink = ((unsigned char *)buf)[0];
    (void)sink;
  }
  free(buf);
  return 0;
}

static int read_col_all(fitsfile *fptr, int col, long nrows, int *status) {
  return read_col_range(fptr, col, 1, nrows, status);
}

/* Prefer fits_read_cols (row-chunked multi-column; one file pass). Fall back to
 * fits_get_rowsize windows when any column is BIT/string/VLA. */
static int table_read_window(fitsfile *fptr, int *cols, int ncols, long firstrow,
                             long nrows, int *status) {
  if (nrows <= 0 || ncols <= 0) {
    return 0;
  }
  int all_simple = 1;
  int *datatypes = (int *)calloc((size_t)ncols, sizeof(int));
  size_t *elsz = (size_t *)calloc((size_t)ncols, sizeof(size_t));
  long *reps = (long *)calloc((size_t)ncols, sizeof(long));
  if (!datatypes || !elsz || !reps) {
    free(datatypes);
    free(elsz);
    free(reps);
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  for (int i = 0; i < ncols; ++i) {
    int typecode = 0;
    long width = 0;
    if (fits_get_coltype(fptr, cols[i], &typecode, &reps[i], &width, status)) {
      free(datatypes);
      free(elsz);
      free(reps);
      return *status;
    }
    if (col_is_vla(fptr, cols[i]) || !col_is_simple_numeric(typecode) ||
        col_read_dtype(typecode, &datatypes[i], &elsz[i])) {
      all_simple = 0;
      break;
    }
    if (reps[i] < 1) {
      reps[i] = 1;
    }
  }

  if (all_simple) {
    /* fits_read_cols already chunks via fits_get_rowsize — one call / full window.
     * CRITICAL: anynul is per-column (anynul[i]), not a single flag. */
    void **arrays = (void **)calloc((size_t)ncols, sizeof(void *));
    void **nulvals = (void **)calloc((size_t)ncols, sizeof(void *));
    int *anynuls = (int *)calloc((size_t)ncols, sizeof(int));
    if (!arrays || !nulvals || !anynuls) {
      free(arrays);
      free(nulvals);
      free(anynuls);
      free(datatypes);
      free(elsz);
      free(reps);
      *status = MEMORY_ALLOCATION;
      return *status;
    }
    for (int i = 0; i < ncols; ++i) {
      arrays[i] = malloc((size_t)nrows * (size_t)reps[i] * elsz[i]);
      if (!arrays[i]) {
        for (int j = 0; j < i; ++j) {
          free(arrays[j]);
        }
        free(arrays);
        free(nulvals);
        free(anynuls);
        free(datatypes);
        free(elsz);
        free(reps);
        *status = MEMORY_ALLOCATION;
        return *status;
      }
    }
    if (fits_read_cols(fptr, ncols, datatypes, cols, (LONGLONG)firstrow, (LONGLONG)nrows,
                       nulvals, arrays, anynuls, status)) {
      for (int i = 0; i < ncols; ++i) {
        free(arrays[i]);
      }
      free(arrays);
      free(nulvals);
      free(anynuls);
      free(datatypes);
      free(elsz);
      free(reps);
      return *status;
    }
    volatile unsigned char sink = 0;
    for (int i = 0; i < ncols; ++i) {
      sink ^= ((unsigned char *)arrays[i])[0];
      free(arrays[i]);
    }
    (void)sink;
    free(arrays);
    free(nulvals);
    free(anynuls);
    free(datatypes);
    free(elsz);
    free(reps);
    return 0;
  }

  free(datatypes);
  free(elsz);
  free(reps);
  /* Mixed / VLA / string: CFITSIO rowsize windows, all requested cols per window. */
  long rowchunk = 0;
  int st = 0;
  fits_get_rowsize(fptr, &rowchunk, &st);
  if (st || rowchunk < 1) {
    rowchunk = 1024;
    *status = 0;
  }
  for (long base = 0; base < nrows; base += rowchunk) {
    long n = rowchunk;
    if (base + n > nrows) {
      n = nrows - base;
    }
    for (int i = 0; i < ncols; ++i) {
      if (read_col_range(fptr, cols[i], firstrow + base, n, status)) {
        return *status;
      }
    }
  }
  return 0;
}

static int do_table_read_cols(fitsfile *fptr, int kncols, long firstrow, long nrows,
                              int *status) {
  long nrows_all = 0;
  int ncols_all = 0;
  if (fits_get_num_rows(fptr, &nrows_all, status) ||
      fits_get_num_cols(fptr, &ncols_all, status)) {
    return *status;
  }
  if (nrows_all <= 0 || ncols_all <= 0) {
    return 0;
  }
  if (firstrow < 1) {
    firstrow = 1;
  }
  if (firstrow > nrows_all) {
    return 0;
  }
  if (nrows < 0 || firstrow + nrows - 1 > nrows_all) {
    nrows = nrows_all - firstrow + 1;
  }
  int ncols = ncols_all;
  if (kncols > 0 && kncols < ncols) {
    ncols = kncols;
  }
  int *cols = (int *)malloc((size_t)ncols * sizeof(int));
  if (!cols) {
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  for (int i = 0; i < ncols; ++i) {
    cols[i] = i + 1;
  }
  int rc = table_read_window(fptr, cols, ncols, firstrow, nrows, status);
  free(cols);
  return rc;
}

static int do_table_read(fitsfile *fptr, int *status) {
  (void)do_table_read_tblbytes;
  (void)read_col_all;
  return do_table_read_cols(fptr, /*kncols=*/-1, /*firstrow=*/1, /*nrows=*/-1, status);
}

static int do_table_proj(fitsfile *fptr, int kncols, int *status) {
  if (kncols < 1) {
    kncols = 1;
  }
  return do_table_read_cols(fptr, kncols, 1, -1, status);
}

static int do_table_slice(fitsfile *fptr, long start_row, long nrows_win, int *status) {
  return do_table_read_cols(fptr, -1, start_row, nrows_win, status);
}

static int colnum_by_name(fitsfile *fptr, const char *name, int *col, int *status) {
  if (fits_get_colnum(fptr, CASEINSEN, (char *)name, col, status)) {
    return *status;
  }
  return 0;
}

static int do_table_scan(fitsfile *fptr, const char *colname, int *status) {
  /* Scorecard scan_count smart path: header nrows only (no column I/O). */
  (void)colname;
  long nrows = 0;
  if (fits_get_num_rows(fptr, &nrows, status)) {
    return *status;
  }
  volatile long sink = nrows;
  (void)sink;
  return 0;
}

static int do_table_pred(fitsfile *fptr, const char *colname, int *status) {
  int col = 0;
  if (colnum_by_name(fptr, colname, &col, status)) {
    return *status;
  }
  long nrows = 0;
  if (fits_get_num_rows(fptr, &nrows, status)) {
    return *status;
  }
  if (nrows <= 0) {
    return 0;
  }
  int typecode = 0;
  long repeat = 0, width = 0;
  if (fits_get_coltype(fptr, col, &typecode, &repeat, &width, status)) {
    return *status;
  }
  int datatype = TDOUBLE;
  size_t elsz = 8;
  if (!col_is_vla(fptr, col) && col_is_simple_numeric(typecode)) {
    col_read_dtype(typecode, &datatype, &elsz);
  }
  long rowchunk = 0;
  int st = 0;
  fits_get_rowsize(fptr, &rowchunk, &st);
  if (st || rowchunk < 1) {
    rowchunk = 4096;
    *status = 0;
  }
  long rep = repeat > 0 ? repeat : 1;
  /* In-place compact of kept scalar values (scorecard smart peer). */
  double *kept = (double *)malloc((size_t)nrows * sizeof(double));
  if (!kept) {
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  long nkept = 0;
  void *chunk = malloc((size_t)rowchunk * (size_t)rep * elsz);
  if (!chunk) {
    free(kept);
    *status = MEMORY_ALLOCATION;
    return *status;
  }
  for (long base = 0; base < nrows; base += rowchunk) {
    long n = rowchunk;
    if (base + n > nrows) {
      n = nrows - base;
    }
    int anynul = 0;
    if (fits_read_col(fptr, datatype, col, base + 1, 1, n * rep, NULL, chunk, &anynul,
                      status)) {
      free(chunk);
      free(kept);
      return *status;
    }
    if (datatype == TDOUBLE) {
      double *d = (double *)chunk;
      for (long i = 0; i < n * rep; ++i) {
        if (d[i] > 0.0) {
          kept[nkept++] = d[i];
        }
      }
    } else if (datatype == TFLOAT) {
      float *d = (float *)chunk;
      for (long i = 0; i < n * rep; ++i) {
        if (d[i] > 0.0f) {
          kept[nkept++] = (double)d[i];
        }
      }
    } else if (datatype == TINT || datatype == TLONG) {
      int *d = (int *)chunk;
      for (long i = 0; i < n * rep; ++i) {
        if (d[i] > 0) {
          kept[nkept++] = (double)d[i];
        }
      }
    } else if (datatype == TSHORT) {
      short *d = (short *)chunk;
      for (long i = 0; i < n * rep; ++i) {
        if (d[i] > 0) {
          kept[nkept++] = (double)d[i];
        }
      }
    } else {
      /* Fallback: re-read window as TDOUBLE for exotic types. */
      double *tmp = (double *)malloc((size_t)n * (size_t)rep * sizeof(double));
      if (!tmp) {
        free(chunk);
        free(kept);
        *status = MEMORY_ALLOCATION;
        return *status;
      }
      if (fits_read_col(fptr, TDOUBLE, col, base + 1, 1, n * rep, NULL, tmp, &anynul,
                        status)) {
        free(tmp);
        free(chunk);
        free(kept);
        return *status;
      }
      for (long i = 0; i < n * rep; ++i) {
        if (tmp[i] > 0.0) {
          kept[nkept++] = tmp[i];
        }
      }
      free(tmp);
    }
  }
  volatile double sink = nkept > 0 ? kept[0] : 0.0;
  (void)sink;
  free(chunk);
  free(kept);
  return 0;
}

static int run_job_once(const job_t *job, int *status) {
  fitsfile *fptr = NULL;
  if (fits_open_file(&fptr, job->path, READONLY, status)) {
    return *status;
  }
  int rc = 0;
  switch (job->op) {
    case OP_READ_FULL: {
      int hdu = job->a[0] ? atoi(job->a) : 0;
      rc = (hdu > 0 ? move_hdu(fptr, hdu, status) : move_first_image(fptr, status)) ||
           do_read_full(fptr, status);
      break;
    }
    case OP_CUTOUT: {
      long hx = job->a[0] ? atol(job->a) : 100;
      long hy = job->b[0] ? atol(job->b) : 100;
      int hdu = job->c[0] ? atoi(job->c) : 0;
      long x0 = job->d[0] ? atol(job->d) : -1;
      long y0 = job->e[0] ? atol(job->e) : -1;
      rc = (hdu > 0 ? move_hdu(fptr, hdu, status) : move_first_image(fptr, status)) ||
           do_cutout(fptr, hx, hy, x0, y0, status);
      break;
    }
    case OP_CUTOUT_REP: {
      /* a=coords_file; optional b=hdu (CFITSIO 1-based); coords preloaded. */
      int hdu = job->b[0] ? atoi(job->b) : 0;
      rc = (hdu > 0 ? move_hdu(fptr, hdu, status) : move_first_image(fptr, status)) ||
           do_cutout_rep(fptr, job->coords, job->ncoords, status);
      break;
    }
    case OP_HEADER: {
      int hdu = job->a[0] ? atoi(job->a) : 0;
      rc = (hdu > 0 ? move_hdu(fptr, hdu, status) : move_first_image(fptr, status)) ||
           do_header_read(fptr, status);
      break;
    }
    case OP_RANDOM_EXT: {
      int nreads = job->a[0] ? atoi(job->a) : 200;
      rc = do_random_ext(fptr, nreads, status);
      break;
    }
    case OP_TABLE_READ:
      rc = move_first_table(fptr, status) || do_table_read(fptr, status);
      break;
    case OP_TABLE_PROJ: {
      int k = job->a[0] ? atoi(job->a) : 3;
      rc = move_first_table(fptr, status) || do_table_proj(fptr, k, status);
      break;
    }
    case OP_TABLE_SLICE: {
      long start = job->a[0] ? atol(job->a) : 1;
      long n = job->b[0] ? atol(job->b) : 1000;
      rc = move_first_table(fptr, status) || do_table_slice(fptr, start, n, status);
      break;
    }
    case OP_TABLE_SCAN:
      rc = move_first_table(fptr, status) ||
           do_table_scan(fptr, job->a[0] ? job->a : "flux", status);
      break;
    case OP_TABLE_PRED:
      rc = move_first_table(fptr, status) ||
           do_table_pred(fptr, job->a[0] ? job->a : "flux", status);
      break;
    case OP_TABLE_HEADER:
      rc = move_first_table(fptr, status) || do_header_read(fptr, status);
      break;
    default:
      *status = BAD_OPTION;
      rc = *status;
      break;
  }
  int close_st = 0;
  fits_close_file(fptr, &close_st);
  if (rc && !*status) {
    *status = rc;
  }
  return *status;
}

static int split_tsv(char *line, char *fields[], int maxf) {
  int n = 0;
  char *p = line;
  while (n < maxf) {
    fields[n++] = p;
    char *tab = strchr(p, '\t');
    if (!tab) {
      break;
    }
    *tab = '\0';
    p = tab + 1;
  }
  return n;
}

static int load_jobs(const char *path, job_t **out_jobs, int *out_n) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno));
    return 1;
  }
  int cap = 256, n = 0;
  job_t *jobs = (job_t *)calloc((size_t)cap, sizeof(job_t));
  char line[2048];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') {
      continue;
    }
    size_t L = strlen(line);
    while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) {
      line[--L] = '\0';
    }
    char *fields[10] = {0};
    int nf = split_tsv(line, fields, 10);
    if (nf < 3) {
      continue;
    }
    if (n >= cap) {
      cap *= 2;
      jobs = (job_t *)realloc(jobs, (size_t)cap * sizeof(job_t));
    }
    job_t *j = &jobs[n];
    memset(j, 0, sizeof(*j));
    snprintf(j->case_id, sizeof(j->case_id), "%s", fields[0]);
    snprintf(j->opname, sizeof(j->opname), "%s", fields[1]);
    snprintf(j->path, sizeof(j->path), "%s", fields[2]);
    if (nf > 3 && fields[3]) {
      snprintf(j->a, sizeof(j->a), "%s", fields[3]);
    }
    if (nf > 4 && fields[4]) {
      snprintf(j->b, sizeof(j->b), "%s", fields[4]);
    }
    if (nf > 5 && fields[5]) {
      snprintf(j->c, sizeof(j->c), "%s", fields[5]);
    }
    if (nf > 6 && fields[6]) {
      snprintf(j->d, sizeof(j->d), "%s", fields[6]);
    }
    if (nf > 7 && fields[7]) {
      snprintf(j->e, sizeof(j->e), "%s", fields[7]);
    }
    j->op = parse_op(j->opname);
    if (j->op == OP_UNKNOWN) {
      fprintf(stderr, "skip unknown op %s for %s\n", j->opname, j->case_id);
      continue;
    }
    /* Preload cutout_rep coords outside the timed path. */
    if (j->op == OP_CUTOUT_REP && j->a[0]) {
      FILE *cf = fopen(j->a, "r");
      if (!cf) {
        fprintf(stderr, "fopen coords %s: %s\n", j->a, strerror(errno));
        free(jobs);
        fclose(f);
        return 1;
      }
      int ccap = 64, cn = 0;
      long *coords = (long *)malloc((size_t)ccap * 4 * sizeof(long));
      char cline[256];
      while (fgets(cline, sizeof(cline), cf)) {
        if (cline[0] == '#' || cline[0] == '\n') {
          continue;
        }
        long x0 = 0, y0 = 0, hx = 0, hy = 0;
        if (sscanf(cline, "%ld %ld %ld %ld", &x0, &y0, &hx, &hy) != 4) {
          continue;
        }
        if (cn >= ccap) {
          ccap *= 2;
          coords = (long *)realloc(coords, (size_t)ccap * 4 * sizeof(long));
        }
        coords[4 * cn + 0] = x0;
        coords[4 * cn + 1] = y0;
        coords[4 * cn + 2] = hx;
        coords[4 * cn + 3] = hy;
        ++cn;
      }
      fclose(cf);
      j->coords = coords;
      j->ncoords = cn;
    }
    ++n;
  }
  fclose(f);
  *out_jobs = jobs;
  *out_n = n;
  return 0;
}

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s --jobs FILE [--runs N] [--warmup N] [--csv PATH|-]\n",
          argv0);
}

int main(int argc, char **argv) {
  const char *jobs_path = NULL;
  const char *csv_path = "-";
  int runs = 7, warmup = 2;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--jobs") == 0 && i + 1 < argc) {
      jobs_path = argv[++i];
    } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
      runs = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
  }
  if (!jobs_path) {
    usage(argv[0]);
    return 2;
  }
  if (runs < 1) {
    runs = 1;
  }
  if (warmup < 0) {
    warmup = 0;
  }

  job_t *jobs = NULL;
  int njobs = 0;
  if (load_jobs(jobs_path, &jobs, &njobs)) {
    return 1;
  }
  if (njobs == 0) {
    fprintf(stderr, "no jobs in %s\n", jobs_path);
    free(jobs);
    return 1;
  }

  FILE *out = stdout;
  if (strcmp(csv_path, "-") != 0) {
    out = fopen(csv_path, "w");
    if (!out) {
      fprintf(stderr, "fopen(%s): %s\n", csv_path, strerror(errno));
      free(jobs);
      return 1;
    }
  }

  float vnum = 0.0f;
  fits_get_version(&vnum);
  char version[64];
  snprintf(version, sizeof(version), "%.3f", (double)vnum);

  fprintf(out,
          "library,method,case_id,operation,status,time_s,n_runs,warmup,cfitsio_version,"
          "api_note\n");

  double *samples = (double *)malloc((size_t)runs * sizeof(double));
  int overall = 0;
  for (int ji = 0; ji < njobs; ++ji) {
    job_t *j = &jobs[ji];
    char case_esc[300];
    csv_escape(j->case_id, case_esc, sizeof(case_esc));
    const char *opname = op_out_name(j);
    const char *api = "fits_read_img";
    switch (j->op) {
      case OP_CUTOUT:
      case OP_CUTOUT_REP:
        api = "fits_read_subset";
        break;
      case OP_HEADER:
        api = "fits_read_keyn";
        break;
      case OP_RANDOM_EXT:
        api = "fits_movabs_hdu+fits_read_img";
        break;
      case OP_TABLE_READ:
      case OP_TABLE_PROJ:
      case OP_TABLE_SLICE:
        api = "fits_read_cols+fits_get_rowsize";
        break;
      case OP_TABLE_SCAN:
        api = "fits_get_num_rows";
        break;
      case OP_TABLE_PRED:
        api = "fits_read_col+compact";
        break;
      case OP_TABLE_HEADER:
        api = "fits_read_keyn";
        break;
      default:
        break;
    }

    int status = 0;
    for (int w = 0; w < warmup; ++w) {
      status = 0;
      if (run_job_once(j, &status)) {
        char err[FLEN_ERRMSG] = {0};
        fits_get_errstatus(status, err);
        fprintf(stderr, "%s/%s warmup fail %d (%s)\n", j->case_id, opname, status, err);
        fprintf(out, "cfitsio,cfitsio_direct,%s,%s,ERROR,,,,%s,%s\n", case_esc, opname, version,
                api);
        overall = 1;
        goto next;
      }
    }
    for (int r = 0; r < runs; ++r) {
      status = 0;
      double t0 = monotonic_s();
      if (run_job_once(j, &status)) {
        char err[FLEN_ERRMSG] = {0};
        fits_get_errstatus(status, err);
        fprintf(stderr, "%s/%s run fail %d (%s)\n", j->case_id, opname, status, err);
        fprintf(out, "cfitsio,cfitsio_direct,%s,%s,ERROR,,,,%s,%s\n", case_esc, opname, version,
                api);
        overall = 1;
        goto next;
      }
      samples[r] = monotonic_s() - t0;
    }
    double med = median_s(samples, runs);
    fprintf(out, "cfitsio,cfitsio_direct,%s,%s,OK,%.9g,%d,%d,%s,%s\n", case_esc, opname, med, runs,
            warmup, version, api);
  next:;
  }

  free(samples);
  for (int i = 0; i < njobs; ++i) {
    free(jobs[i].coords);
  }
  free(jobs);
  if (out != stdout) {
    fclose(out);
  }
  return overall;
}
