import math
from collections import namedtuple

import numpy as np
from typeguard import typechecked

SummaryStatistics = namedtuple(
    "SummaryStatistics", ("n", "min", "max", "sum", "mean", "var", "sd")
)


@typechecked
class OnlineStatistics:
    """
    @class Statistics:
        A class for performing basic statistical calculations.

    @attribute min:
        The minimum value encountered so far. Initialized to positive infinity.

    @attribute max:
        The maximum value encountered so far. Initialized to negative infinity.

    @attribute n:
        The number of values encountered so far. Initialized to 0.

    @attribute sum:
        The sum of all values encountered so far. Initialized to 0.0.

    @attribute mean:
        The mean value calculated based on the encountered values. Initialized
        to 0.0.

    @attribute var:
        The variance value calculated based on the encountered values.
        Initialized to 0.0.

    @attribute sd:
        The standard deviation value calculated based on the encountered
        values. Initialized to 0.0.

    @attribute M2n:
        The intermediate value used in calculating variance. Initialized to
        0.0.

    @method update(vals: np.ndarray):
        Updates the statistics with an array of values.

    Args:
        vals (np.ndarray): An array of values to update the statistics.

    Returns:
        None.
    """

    min: float = math.inf
    max: float = -math.inf
    n: int = 0
    sum: float = 0.0
    mean: float = 0.0

    M2n: float = 0.0
    ddof: float = 1.0

    @property
    def var(self) -> float:
        return self.M2n / (self.n - self.ddof)

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    def __init__(self, unbiased: bool = True):
        if unbiased:
            self.ddof = 1  # Use unbiased variance estimator
        else:
            self.ddof = (
                0  # Use maximum likelihood (best L2) variance estimator
            )

    def update(self, vals: np.ndarray) -> None:
        """
        Update the statistics with an array of values.

        For updating the variance a variant of Welford's algo is used (see e.g.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).

        Parameters
        ----------
        vals : np.ndarray
            The array of values to update the statistics with.
        """  # noqa: E501

        self.min = min(self.min, vals.min())
        self.max = max(self.max, vals.max())
        self.n += len(vals)
        self.sum += vals.sum()
        old_mean = self.mean
        self.mean = self.sum / self.n
        self.M2n += ((vals - old_mean) * (vals - self.mean)).sum()

    def update_single(self, val):
        # Note: type checking is too tricky due to all the different numeric
        # data type in vanilla python and in numpy
        self.min = min(self.min, val)
        self.max = max(self.max, val)
        self.n += 1
        self.sum += val
        old_mean = self.mean
        self.mean = self.sum / self.n
        self.M2n += (val - old_mean) * (val - self.mean)

    def describe(self):
        return SummaryStatistics(
            self.n, self.min, self.max, self.sum, self.mean, self.var, self.sd
        )


@typechecked
class HistData:
    bin_edges: np.ndarray[float]
    counts: np.ndarray[int]

    def __init__(self, bin_edges: np.ndarray[float], counts: np.ndarray[int]):
        if len(bin_edges) != len(counts) + 1:
            raise ValueError(
                "`bin_edges` must have one more element than `counts` "
                f"({len(bin_edges)=}, {len(counts)=})"
            )

        self.bin_edges = bin_edges
        self.counts = counts

    @property
    def bin_centers(self) -> np.ndarray[float]:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def density(self) -> np.ndarray[float]:
        dx = np.diff(self.bin_edges)
        counts = self.counts.astype(float)
        return counts / (dx * counts.sum())

    @staticmethod
    def bin_size_sturges(stats: OnlineStatistics) -> float:
        return (stats.max - stats.min) / (np.log2(stats.n) + 1.0)

    @staticmethod
    def bin_size_scott(stats: OnlineStatistics) -> float:
        factor = (24 * 24 * np.pi) ** (1.0 / 6.0)
        return factor * stats.sd * stats.n ** (-1.0 / 3.0)

    @staticmethod
    def bin_size_terrell_scott(stats: OnlineStatistics) -> float:
        num_bins = np.ceil((2.0 * stats.n) ** (1.0 / 3.0))
        return (stats.max - stats.min) / num_bins

    @staticmethod
    def get_bin_edges(
        stats: OnlineStatistics,
        name="scott",
        clip: tuple[int, int] | None = None,
        extend: bool = False,
    ):
        if name == "scott":
            bin_size = HistData.bin_size_scott(stats)
        elif name == "terrell_scott":
            bin_size = HistData.bin_size_terrell_scott(stats)
        elif name == "sturges":
            bin_size = HistData.bin_size_sturges(stats)
        elif name == "auto":
            bin_size = bin_size = HistData.bin_size_scott(stats)
        else:
            raise ValueError(f"Unrecognized binning algorithm name: {name}")

        range = (stats.min, stats.max)
        num_bins = int(np.ceil((range[1] - range[0]) / bin_size))
        if clip is not None:
            num_bins = np.clip(num_bins, *clip)

        if extend:
            bin_size = (range[1] - range[0]) / num_bins
            num_bins += 1
            if clip is not None:
                num_bins = np.clip(num_bins, *clip)
            range = (stats.min - 0.5 * bin_size, stats.max + 0.5 * bin_size)

        bin_edges = np.histogram_bin_edges([], bins=num_bins, range=range)
        return bin_edges


def gaussian_iqr(mu: float, sigma: float) -> tuple[float, float]:
    # Quartiles for the standard normal distribution are about +-0.67.
    # Get the exact value with `scipy.stats.norm.isf(0.25)`.
    alpha = 0.6744897501960817
    return (mu - alpha * sigma, mu + alpha * sigma)
