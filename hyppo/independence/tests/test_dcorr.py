import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import linear, power
from .. import Dcorr
from .._dcorr_blocks import (
    _block_row_sums,
    _block_rows,
    _parse_memory_budget,
    dcorr_block_statistic,
    dcorr_block_test,
)


class TestDcorrStat:
    @pytest.mark.parametrize("n", [100, 200])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat1, pvalue1 = Dcorr().test(x, y)
        stat2 = Dcorr().statistic(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(stat2, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y = linear(n, 1)
        stat1, pvalue1 = Dcorr().test(x, y, random_state=2)
        stat2, pvalue2 = Dcorr().test(x, y, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2

    def test_dcorr_sqrt_bug(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 9, 4, 4])
        stat = Dcorr(bias=True).test(x, y, reps=0)[0]

        assert_almost_equal(stat, 0.762676242417, decimal=2)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_dtypes(self, dtype):
        np.random.seed(123456789)
        x, y = linear(100, 1)
        x_cast = (x * 100).astype(dtype)
        y_cast = (y * 100).astype(dtype)
        stat = Dcorr().statistic(x_cast, y_cast)
        assert_almost_equal(stat, 1.0, decimal=2)


class TestDcorrTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "Dcorr",
            sim_type="indep",
            sim="multimodal_independence",
            n=100,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)

    def test_oned_fast(self):
        np.random.seed(123456789)
        est_power = power(
            "Dcorr",
            sim_type="indep",
            sim="multimodal_independence",
            n=100,
            p=1,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)

    def test_threed_fast(self):
        np.random.seed(123456789)
        est_power = power(
            "Dcorr",
            sim_type="indep",
            sim="multimodal_independence",
            n=100,
            p=3,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)


class TestDcorrMemoryBudget:
    """Exact, block-budgeted path must match the standard in-memory path."""

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("budget", [4096, "32KB", "1MB"])
    def test_matches_standard_raw_samples(self, bias, budget):
        np.random.seed(123456789)
        x, y = linear(80, 2)
        stat_std, p_std = Dcorr(bias=bias).test(
            x, y, auto=False, reps=200, random_state=42
        )
        stat_blk, p_blk = Dcorr(bias=bias, memory_budget=budget).test(
            x, y, auto=False, reps=200, random_state=42
        )
        assert_almost_equal(stat_blk, stat_std, decimal=12)
        assert_almost_equal(p_blk, p_std, decimal=12)

    @pytest.mark.parametrize("bias", [True, False])
    def test_matches_standard_precomputed(self, bias):
        np.random.seed(123456789)
        x, y = linear(60, 2)
        from sklearn.metrics import pairwise_distances

        distx = pairwise_distances(x)
        disty = pairwise_distances(y)
        stat_std, p_std = Dcorr(compute_distance=None, bias=bias).test(
            distx, disty, auto=False, reps=150, random_state=11
        )
        stat_blk, p_blk = Dcorr(
            compute_distance=None, bias=bias, memory_budget="64KB"
        ).test(distx, disty, auto=False, reps=150, random_state=11)
        assert_almost_equal(stat_blk, stat_std, decimal=12)
        assert_almost_equal(p_blk, p_std, decimal=12)

    def test_statistic_matches_on_distance_matrices(self):
        np.random.seed(123456789)
        x, y = linear(60, 2)
        from sklearn.metrics import pairwise_distances

        distx = pairwise_distances(x)
        disty = pairwise_distances(y)
        stat_std = Dcorr(compute_distance=None).statistic(distx, disty)
        for budget in (2048, "16KB"):
            stat_blk = Dcorr(
                compute_distance=None, memory_budget=budget
            ).statistic(distx, disty)
            assert_almost_equal(stat_blk, stat_std, decimal=12)

    def test_block_size_independence(self):
        np.random.seed(123456789)
        x, y = linear(70, 2)
        from sklearn.metrics import pairwise_distances

        distx = pairwise_distances(x)
        disty = pairwise_distances(y)
        ref = dcorr_block_statistic(distx, disty, memory_budget="4MB")
        # budgets forcing block sizes of 1 and everything in between
        for budget in (512, 4096, "32KB", "256KB"):
            got = dcorr_block_statistic(distx, disty, memory_budget=budget)
            assert_almost_equal(got, ref, decimal=12)

    def test_readonly_memmap_inputs(self, tmp_path):
        np.random.seed(123456789)
        n = 80
        x, y = linear(n, 2)
        from sklearn.metrics import pairwise_distances

        distx = pairwise_distances(x)
        disty = pairwise_distances(y)

        px = tmp_path / "distx.dat"
        py = tmp_path / "disty.dat"
        mx = np.memmap(px, dtype=np.float64, mode="w+", shape=(n, n))
        my = np.memmap(py, dtype=np.float64, mode="w+", shape=(n, n))
        mx[:] = distx
        my[:] = disty
        del mx, my

        rx = np.memmap(px, dtype=np.float64, mode="r", shape=(n, n))
        ry = np.memmap(py, dtype=np.float64, mode="r", shape=(n, n))
        assert not rx.flags.writeable and not ry.flags.writeable

        stat_std, p_std = Dcorr(compute_distance=None).test(
            distx, disty, auto=False, reps=120, random_state=7
        )
        stat_blk, p_blk, null_dist = dcorr_block_test(
            rx, ry, reps=120, memory_budget="64KB", random_state=7
        )
        assert_almost_equal(stat_blk, stat_std, decimal=12)
        assert_almost_equal(p_blk, p_std, decimal=12)
        assert null_dist.shape == (120,)
        # the source files must not have been modified
        assert_almost_equal(np.asarray(rx), distx)
        assert_almost_equal(np.asarray(ry), disty)

        # same through the Dcorr API with precomputed read-only matrices
        stat_api, p_api = Dcorr(
            compute_distance=None, memory_budget="128KB"
        ).test(rx, ry, auto=False, reps=120, random_state=7)
        assert_almost_equal(stat_api, stat_std, decimal=12)
        assert_almost_equal(p_api, p_std, decimal=12)

    def test_memmap_parallel_workers(self, tmp_path):
        np.random.seed(123456789)
        n = 60
        x, y = linear(n, 3)
        from sklearn.metrics import pairwise_distances

        distx = pairwise_distances(x)
        disty = pairwise_distances(y)
        px = tmp_path / "dx.dat"
        py = tmp_path / "dy.dat"
        np.memmap(px, dtype=np.float64, mode="w+", shape=(n, n))[:] = distx
        np.memmap(py, dtype=np.float64, mode="w+", shape=(n, n))[:] = disty
        rx = np.memmap(px, dtype=np.float64, mode="r", shape=(n, n))
        ry = np.memmap(py, dtype=np.float64, mode="r", shape=(n, n))

        stat_1, p_1, _ = dcorr_block_test(
            rx, ry, reps=200, workers=1, memory_budget="64KB", random_state=5
        )
        stat_2, p_2, _ = dcorr_block_test(
            rx, ry, reps=200, workers=2, memory_budget="64KB", random_state=5
        )
        assert_almost_equal(stat_1, stat_2, decimal=12)
        assert_almost_equal(p_1, p_2, decimal=12)

    def test_perm_blocks_match_standard(self):
        rng = np.random.default_rng(0)
        n = 60
        x = rng.normal(size=(n, 2))
        y = rng.normal(size=(n, 1)) + x[:, :1]
        perm_blocks = np.tile(np.arange(6), 10)
        stat_std, p_std = Dcorr().test(
            x,
            y,
            auto=False,
            reps=100,
            random_state=5,
            perm_blocks=perm_blocks,
        )
        stat_blk, p_blk = Dcorr(memory_budget="128KB").test(
            x,
            y,
            auto=False,
            reps=100,
            random_state=5,
            perm_blocks=perm_blocks,
        )
        assert_almost_equal(stat_blk, stat_std, decimal=12)
        assert_almost_equal(p_blk, p_std, decimal=12)

    def test_reps_zero(self):
        np.random.seed(123456789)
        x, y = linear(50, 1)
        stat, pvalue = Dcorr(memory_budget="1MB").test(
            x, y, auto=False, reps=0
        )
        assert pvalue == 1.0
        stat_ref = Dcorr().test(x, y, auto=False, reps=0)[0]
        assert_almost_equal(stat, stat_ref, decimal=12)

    def test_seeded_reproducibility(self):
        np.random.seed(123456789)
        x, y = linear(60, 2)
        first = Dcorr(memory_budget="64KB").test(
            x, y, auto=False, reps=100, random_state=2
        )
        second = Dcorr(memory_budget="64KB").test(
            x, y, auto=False, reps=100, random_state=2
        )
        assert first == second

    def test_budget_parser(self):
        assert _parse_memory_budget(1024) == 1024
        assert _parse_memory_budget("256MB") == 256 * 1024**2
        assert _parse_memory_budget(" 8 kb ") == 8 * 1024
        assert _parse_memory_budget("1024") == 1024
        with pytest.raises(ValueError):
            _parse_memory_budget("not-a-budget")
        with pytest.raises(ValueError):
            _parse_memory_budget(0)
        with pytest.raises(ValueError):
            _parse_memory_budget(-1)
        with pytest.raises(TypeError):
            _parse_memory_budget(1.5)
        with pytest.raises(TypeError):
            _parse_memory_budget(None)

    def test_block_rows_within_bounds(self):
        n = 100
        # the smallest possible budgets still produce a valid single-row block
        assert _block_rows(n, 1, 8) == 1
        assert _block_rows(n, 64, 8) == 1
        # a budget sized for exactly b rows returns b
        b = 10
        budget = 2 * b * n * 8 + 4 * n * 8
        assert _block_rows(n, budget, 8) == b
        # generous budgets cap at the full matrix
        assert _block_rows(n, 10**12, 8) == n
        all_budgets = [_block_rows(n, bud, 8) for bud in (512, 4096, 65536)]
        assert all(1 <= rows <= n for rows in all_budgets)

    def test_block_row_sums_match(self):
        rng = np.random.default_rng(3)
        dist = rng.normal(size=(40, 40))
        for block_rows in (1, 7, 40):
            row_sums, total = _block_row_sums(dist, block_rows)
            assert_almost_equal(row_sums, dist.sum(axis=1))
            assert_almost_equal(total, dist.sum())

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_dtypes_match_standard(self, dtype):
        np.random.seed(123456789)
        x, y = linear(50, 2)
        from sklearn.metrics import pairwise_distances

        distx = (pairwise_distances(x) * 10).astype(dtype)
        disty = (pairwise_distances(y) * 10).astype(dtype)
        stat_std = Dcorr(compute_distance=None).statistic(
            distx.astype(np.float64), disty.astype(np.float64)
        )
        stat_blk = dcorr_block_statistic(distx, disty, memory_budget="64KB")
        assert_almost_equal(stat_blk, stat_std, decimal=5)

    def test_readonly_float32_memmap(self, tmp_path):
        rng = np.random.default_rng(4)
        n = 40
        distx = rng.normal(size=(n, n)).astype(np.float32)
        disty = rng.normal(size=(n, n)).astype(np.float32)
        px = tmp_path / "fx.dat"
        py = tmp_path / "fy.dat"
        np.memmap(px, dtype=np.float32, mode="w+", shape=(n, n))[:] = distx
        np.memmap(py, dtype=np.float32, mode="w+", shape=(n, n))[:] = disty
        rx = np.memmap(px, dtype=np.float32, mode="r", shape=(n, n))
        ry = np.memmap(py, dtype=np.float32, mode="r", shape=(n, n))
        stat_blk = dcorr_block_statistic(rx, ry, memory_budget="32KB")
        stat_ref = dcorr_block_statistic(
            distx.astype(np.float64), disty.astype(np.float64),
            memory_budget="32KB",
        )
        assert_almost_equal(stat_blk, stat_ref, decimal=5)

    def test_no_full_matrix_temporary(self):
        # At a fixed tight budget the traced peak must stay bounded regardless
        # of sample size, whereas the standard path allocates centered n x n
        # matrices and therefore grows with n^2.
        import tracemalloc

        budget = "32KB"
        peaks = []
        for n in (200, 300):
            np.random.seed(123456789)
            x, y = linear(n, 2)
            from sklearn.metrics import pairwise_distances

            distx = pairwise_distances(x)
            disty = pairwise_distances(y)

            tracemalloc.start()
            dcorr_block_test(
                distx, disty, reps=5, memory_budget=budget, random_state=1
            )
            _, peak_budgeted = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak_budgeted)

            full_centered_size = 2 * n * n * 8
            assert peak_budgeted < full_centered_size / 4

        # peak stays effectively flat while n grows (no n^2 temporaries);
        # the larger run may reuse cached blocks, so allow a generous factor
        assert max(peaks) <= 2 * min(peaks)

        # standard in-memory path on the larger problem exceeds the budgeted peak
        np.random.seed(123456789)
        x, y = linear(300, 2)
        from sklearn.metrics import pairwise_distances

        distx = pairwise_distances(x)
        disty = pairwise_distances(y)
        tracemalloc.start()
        Dcorr(compute_distance=None).test(
            distx, disty, auto=False, reps=5, random_state=1
        )
        _, peak_standard = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak_standard > max(peaks)

    def test_invalid_matrix_shape(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            dcorr_block_test(
                rng.normal(size=(10, 8)),
                rng.normal(size=(10, 10)),
                memory_budget="1MB",
            )

