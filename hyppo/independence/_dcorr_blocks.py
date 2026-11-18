"""
Memory-budgeted, exact Dcorr computations.

The standard Dcorr path (:func:`hyppo.independence.dcorr._dcorr`) materializes
two fully centered :math:`n \\times n` distance matrices and, for every
permutation, an additional full copy of the permuted distance matrix. These
routines instead accumulate the row sums, grand total and centered inner
products over row blocks so that the largest temporary array is a single
:math:`b \\times n` block.

The arithmetic reproduces the biased and unbiased statistics defined in
:mod:`hyppo.independence.dcorr` exactly; nothing is approximated and no
observations are dropped. Distance matrices are only read (through fancy
indexing, which returns fresh block-sized copies), so read-only
:class:`numpy.memmap` inputs and permutation index views are supported without
ever copying the whole matrix.
"""

import re

import numpy as np

from sklearn.utils import check_random_state

from ..tools import check_perm_blocks
from ..tools.common import _PermGroups


_MEMORY_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
}


def _parse_memory_budget(memory_budget):
    """Parse a memory budget given as bytes (int) or a string (e.g. ``"256MB"``)."""
    if isinstance(memory_budget, (int, np.integer)) and not isinstance(
        memory_budget, bool
    ):
        budget = int(memory_budget)
    elif isinstance(memory_budget, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", memory_budget)
        if match is None:
            raise ValueError(
                "Could not parse memory_budget "
                f"{memory_budget!r}; use a number of bytes or a string "
                "such as '512MB'."
            )
        value, unit = match.groups()
        unit = unit.lower()
        if unit not in _MEMORY_UNITS:
            raise ValueError(
                f"Unknown memory unit {unit!r} in memory_budget={memory_budget!r}."
            )
        budget = int(float(value) * _MEMORY_UNITS[unit])
    else:
        raise TypeError(
            "memory_budget must be an int (number of bytes) or a str, got "
            f"{type(memory_budget)}."
        )

    if budget <= 0:
        raise ValueError("memory_budget must be strictly positive.")

    return budget


def _block_rows(n, memory_budget, itemsize):
    """
    Largest row-block size whose two block-sized working arrays fit in the
    budget. Each block pass holds two ``b x n`` blocks (one per distance
    matrix), so the dominant temporary footprint is ``2 * b * n * itemsize``
    bytes; small fixed-size vectors (row sums, index arrays) are accounted for
    with slack. The result is always in ``[1, n]``: a budget that cannot even
    hold two single-row blocks falls back to single-row blocks, which is the
    minimum footprint of the exact computation.
    """
    slack = 4 * n * max(itemsize, 8)
    usable = max(memory_budget - slack, 0)
    b = int(usable // (2 * n * itemsize))
    return max(1, min(b, n))


def _block_row_sums(dist, block_rows):
    """
    Row sums and grand total of a distance matrix, accumulated over row
    blocks. Reads the input through row slices only and never allocates an
    ``n x n`` temporary, so read-only memmaps are supported.
    """
    n = dist.shape[0]
    row_sums = np.empty(n, dtype=np.float64)
    total = 0.0
    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        block = dist[start:end, :]
        row_sums[start:end] = block.sum(axis=1, dtype=np.float64)
        total += block.sum(dtype=np.float64)
    return row_sums, float(total)


def _block_centered_products(
    distx,
    disty,
    orderx,
    ordery,
    row_sumsx,
    row_sumsy,
    totalx,
    totaly,
    bias,
    block_rows,
):
    r"""
    Centered inner products :math:`P_{xy}, P_{xx}, P_{yy}` accumulated over
    row blocks, where :math:`P_{ab} = \langle C^a, C^b \rangle` and
    :math:`C^\cdot` is the (biased or unbiased) U-centered distance matrix.

    ``orderx``/``ordery`` are permutation index views into the original
    matrices (``np.arange(n)`` for the unpermuted case). A permuted distance
    matrix is read block by block as ``dist[order[rows]][:, order]`` instead
    of being materialized in full. For a symmetric distance matrix the row and
    column sums of the permuted matrix are both ``row_sums[order]``.

    Only two ``block_rows x n`` blocks and a handful of length-``n`` vectors
    are live at once; the input matrices are never written to.
    """
    n = distx.shape[0]

    rsumx = row_sumsx[orderx]
    rsumy = row_sumsy[ordery]
    col_idx_x = orderx
    col_idx_y = ordery

    if bias:
        center_denom = n
        grand_denom = n * n
    else:
        center_denom = n - 2
        grand_denom = (n - 1) * (n - 2)

    pxy = 0.0
    pxx = 0.0
    pyy = 0.0
    inv_center = 1.0 / center_denom
    grand_term_x = totalx / grand_denom
    grand_term_y = totaly / grand_denom

    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        local_rows = np.arange(end - start)
        block_rows_idx = np.arange(start, end)

        # fancy indexing on a read-only memmap returns a fresh writable copy
        blockx = distx[np.ix_(orderx[block_rows_idx], col_idx_x)].copy()
        blocky = disty[np.ix_(ordery[block_rows_idx], col_idx_y)].copy()

        # in-place centering: C_ij = D_ij - row_i/c - col_j/c + total/grand
        blockx -= (rsumx[block_rows_idx, np.newaxis] * inv_center)
        blockx -= (rsumx[np.newaxis, :] * inv_center)
        blockx += grand_term_x
        blocky -= (rsumy[block_rows_idx, np.newaxis] * inv_center)
        blocky -= (rsumy[np.newaxis, :] * inv_center)
        blocky += grand_term_y

        if not bias:
            # unbiased centering zeros the diagonal, 1(i != j)
            blockx[local_rows, block_rows_idx] = 0.0
            blocky[local_rows, block_rows_idx] = 0.0

        flatx = blockx.ravel()
        flaty = blocky.ravel()
        pxy += np.dot(flatx, flaty)
        pxx += np.dot(flatx, flatx)
        pyy += np.dot(flaty, flaty)

    return float(pxy), float(pxx), float(pyy)


def _stat_from_products(pxy, pxx, pyy, n, bias):
    """Assemble the Dcorr statistic from centered products, matching ``_dcorr``."""
    if bias:
        covar = pxy / (n * n)
        varx = pxx / (n * n)
        vary = pyy / (n * n)
        # only biased Dcov is square-rooted
        if varx <= 0 or vary <= 0 or covar <= 0:
            return 0.0
        return float(np.sqrt(covar / np.sqrt(varx * vary)))

    covar = pxy / (n * (n - 3))
    varx = pxx / (n * (n - 3))
    vary = pyy / (n * (n - 3))
    if varx <= 0 or vary <= 0:
        return 0.0
    return float(covar / np.sqrt(varx * vary))


def _dcorr_block_stat(
    distx,
    disty,
    orderx,
    ordery,
    row_sumsx,
    row_sumsy,
    totalx,
    totaly,
    bias,
    block_rows,
):
    """Block-based Dcorr statistic for one (possibly permuted) pairing."""
    n = distx.shape[0]
    pxy, pxx, pyy = _block_centered_products(
        distx,
        disty,
        orderx,
        ordery,
        row_sumsx,
        row_sumsy,
        totalx,
        totaly,
        bias,
        block_rows,
    )
    return _stat_from_products(pxy, pxx, pyy, n, bias)


def _permutation_stat(
    distx,
    disty,
    ordery,
    row_sumsx,
    row_sumsy,
    totalx,
    totaly,
    bias,
    block_rows,
):
    """Dcorr statistic for one permutation of ``disty`` (``distx`` held fixed)."""
    n = distx.shape[0]
    orderx = np.arange(n)
    pxy, pxx, pyy = _block_centered_products(
        distx,
        disty,
        orderx,
        ordery,
        row_sumsx,
        row_sumsy,
        totalx,
        totaly,
        bias,
        block_rows,
    )
    return _stat_from_products(pxy, pxx, pyy, n, bias)


def _perm_chunk(
    distx,
    disty,
    seeds,
    permuter,
    row_sumsx,
    row_sumsy,
    totalx,
    totaly,
    bias,
    block_rows,
):
    """Compute permutation statistics for a chunk of random seeds."""
    null_chunk = np.empty(len(seeds), dtype=np.float64)
    for i, seed in enumerate(seeds):
        rng = check_random_state(seed)
        order = permuter(rng=rng)
        null_chunk[i] = _permutation_stat(
            distx,
            disty,
            order,
            row_sumsx,
            row_sumsy,
            totalx,
            totaly,
            bias,
            block_rows,
        )
    return null_chunk


def _prepare_block_run(distx, disty, memory_budget):
    """Validate inputs and derive block size plus per-matrix sums from the budget."""
    if (
        distx.ndim != 2
        or distx.shape[0] != distx.shape[1]
        or distx.shape != disty.shape
    ):
        raise ValueError(
            "distx and disty must be square distance matrices of equal shape."
        )
    # exact statistics are accumulated in float64; floating inputs (including
    # read-only memmaps) are used as-is, only non-floating matrices are copied
    if distx.dtype.kind != "f":
        distx = np.asarray(distx, dtype=np.float64)
    if disty.dtype.kind != "f":
        disty = np.asarray(disty, dtype=np.float64)
    n = distx.shape[0]
    budget = _parse_memory_budget(memory_budget)
    itemsize = max(distx.dtype.itemsize, disty.dtype.itemsize, 8)
    block_rows = _block_rows(n, budget, itemsize)
    row_sumsx, totalx = _block_row_sums(distx, block_rows)
    row_sumsy, totaly = _block_row_sums(disty, block_rows)
    return (
        distx,
        disty,
        n,
        block_rows,
        row_sumsx,
        row_sumsy,
        totalx,
        totaly,
    )


def dcorr_block_statistic(distx, disty, bias=False, memory_budget=None):
    """
    Block-based observed Dcorr statistic for two distance matrices.

    Convenience wrapper used by :meth:`Dcorr.statistic`; reads the inputs
    without allocating centered ``n x n`` matrices.
    """
    (
        distx,
        disty,
        n,
        block_rows,
        row_sumsx,
        row_sumsy,
        totalx,
        totaly,
    ) = _prepare_block_run(distx, disty, memory_budget)
    identity = np.arange(n)
    return _dcorr_block_stat(
        distx,
        disty,
        identity,
        identity,
        row_sumsx,
        row_sumsy,
        totalx,
        totaly,
        bias,
        block_rows,
    )


def dcorr_block_test(
    distx,
    disty,
    reps=1000,
    workers=1,
    bias=False,
    memory_budget=None,
    perm_blocks=None,
    random_state=None,
):
    r"""
    Exact Dcorr statistic and permutation p-value under a memory budget.

    Both distance matrices are treated as read-only data: row sums and grand
    totals are accumulated in a single block pass per matrix, and every
    permutation is evaluated through permutation index views, one row block at
    a time. No permutation ever allocates a full :math:`n \times n` copy. The
    biased and unbiased statistics are identical to those produced by the
    standard in-memory path.

    Parameters
    ----------
    distx, disty : ndarray of float
        Square distance matrices of shape ``(n, n)``. Read-only
        :class:`numpy.memmap` arrays are supported.
    reps : int, default: 1000
        Number of permutation replications. ``0`` returns the observed
        statistic only (p-value of 1).
    workers : int, default: 1
        Number of processes used for the permutation replications. ``-1`` uses
        all available cores. Each worker re-opens memmap-backed inputs rather
        than receiving matrix copies.
    bias : bool, default: False
        Whether to use the biased or unbiased statistic.
    memory_budget : int, str, or None, default: None
        Upper bound on the size of the temporary arrays used by this routine,
        in bytes, or as a string such as ``"256MB"``. The block size is chosen
        from the budget; it never changes the result.
    perm_blocks : ndarray or None, default: None
        Restricted-permutation blocks, as in :func:`hyppo.tools.perm_test`.
    random_state : int or None, default: None
        Seed controlling the permutations.

    Returns
    -------
    stat : float
        The observed Dcorr statistic.
    pvalue : float
        The permutation p-value.
    null_dist : ndarray of float
        Null distribution of shape ``(reps,)``.
    """
    (
        distx,
        disty,
        n,
        block_rows,
        row_sumsx,
        row_sumsy,
        totalx,
        totaly,
    ) = _prepare_block_run(distx, disty, memory_budget)

    identity = np.arange(n)
    stat = _dcorr_block_stat(
        distx,
        disty,
        identity,
        identity,
        row_sumsx,
        row_sumsy,
        totalx,
        totaly,
        bias,
        block_rows,
    )

    if reps == 0:
        return stat, 1.0, np.empty(0, dtype=np.float64)

    # match the seeding convention of hyppo.tools.perm_test
    rng = check_random_state(random_state)
    seeds = rng.randint(np.iinfo(np.int32).max, size=reps)

    checked_blocks = check_perm_blocks(perm_blocks)
    permuter = _PermGroups(disty, checked_blocks)

    if workers == 1:
        null_dist = _perm_chunk(
            distx,
            disty,
            seeds,
            permuter,
            row_sumsx,
            row_sumsy,
            totalx,
            totaly,
            bias,
            block_rows,
        )
    else:
        from joblib import Parallel, delayed

        # split seeds into a modest number of chunks so memmap inputs are
        # re-opened per worker instead of being serialized per replication
        n_jobs = None if workers == -1 else workers
        chunks = np.array_split(seeds, min(reps, _n_workers(n_jobs)))
        null_chunks = Parallel(n_jobs=n_jobs)(
            delayed(_perm_chunk)(
                distx,
                disty,
                chunk,
                permuter,
                row_sumsx,
                row_sumsy,
                totalx,
                totaly,
                bias,
                block_rows,
            )
            for chunk in chunks
        )
        null_dist = np.concatenate(null_chunks)

    pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)
    return stat, float(pvalue), null_dist


def _n_workers(n_jobs):
    """Resolve joblib's effective worker count for chunking."""
    if isinstance(n_jobs, int) and n_jobs > 0:
        return n_jobs
    import os

    return os.cpu_count() or 1
