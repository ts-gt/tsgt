import collections
import os
import pickle
import time

import gin
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import tqdm
from matplotlib.gridspec import GridSpec

series_axis = -3
preds_axis = -2
horizon_axis = -1
windows_axis = 0


def violin_stats(X, method, points=100, quantiles=None):
    """
    Returns a list of dictionaries of data which can be used to draw a series
    of violin plots.
    See the Returns section below to view the required keys of the dictionary.
    Users can skip this function and pass a user-defined set of dictionaries
    with the same keys to `~.axes.Axes.violinplot` instead of using Matplotlib
    to do the calculations. See the *Returns* section below for the keys
    that must be present in the dictionaries.
    Parameters
    ----------
    X : array-like
        Sample data that will be used to produce the gaussian kernel density
        estimates. Must have 2 or fewer dimensions.
    method : callable
        The method used to calculate the kernel density estimate for each
        column of data. When called via `method(v, coords)`, it should
        return a vector of the values of the KDE evaluated at the values
        specified in coords.
    points : int, default = 100
        Defines the number of points to evaluate each of the gaussian kernel
        density estimates at.
    quantiles : array-like, default = None
        Defines (if not None) a list of floats in interval [0, 1] for each
        column of data, which represents the quantiles that will be rendered
        for that column of data. Must have 2 or fewer dimensions. 1D array will
        be treated as a singleton list containing them.
    Returns
    -------
    vpstats : list of dict
        A list of dictionaries containing the results for each column of data.
        The dictionaries contain at least the following:
        - coords: A list of scalars containing the coordinates this particular
          kernel density estimate was evaluated at.
        - vals: A list of scalars containing the values of the kernel density
          estimate at each of the coordinates given in `coords`.
        - mean: The mean value for this column of data.
        - median: The median value for this column of data.
        - min: The minimum value for this column of data.
        - max: The maximum value for this column of data.
        - quantiles: The quantile values for this column of data.
    """

    # List of dictionaries describing each of the violins.
    vpstats = []

    # Want X to be a list of data sequences
    X = matplotlib.cbook._reshape_2D(X, "X")

    # Want quantiles to be as the same shape as data sequences
    if quantiles is not None and len(quantiles) != 0:
        quantiles = matplotlib.cbook._reshape_2D(quantiles, "quantiles")
    # Else, mock quantiles if is none or empty
    else:
        quantiles = [[]] * np.shape(X)[0]

    # quantiles should has the same size as dataset
    if np.shape(X)[:1] != np.shape(quantiles)[:1]:
        raise ValueError(
            "List of violinplot statistics and quantiles values"
            " must have the same length"
        )

    def _shift(data, eval_point, delta):
        shift = delta
        value_range = method(data, data)
        max_value = np.max(value_range)
        while True:
            y = method(data, eval_point + shift)
            if y[0] > 0.2 * max_value:
                shift -= delta
                break
            shift += delta
        return shift

    ci = 0.995
    # Zip x and quantiles
    for x, q in zip(X, quantiles):
        # Dictionary of results for this distribution
        stats = {}

        # Calculate basic stats for the distribution
        min_val = np.min(x)
        max_val = np.max(x)
        quantile_val = np.percentile(x, 100 * q)

        # Evaluate the kernel density estimate
        # shift_max = _shift(x, max_val, -2.0)  # 0.001)
        # shift_min = _shift(x, min_val, 2.0)  #-0.001)
        # q_05 = np.quantile(x, 0.05)
        # q_95 = np.quantile(x, 0.95)
        q_05 = np.quantile(x, 1 - ci)
        q_95 = np.quantile(x, ci)
        coords = np.linspace(q_05, q_95, points)
        # coords = np.linspace(min_val + shift_min, max_val + shift_max, points)
        # print(max_val, max_val + shift_max)
        # print(min_val, min_val + shift_min)
        # coords = np.linspace(min_val, max_val, points)
        stats["vals"] = method(x, coords)
        stats["coords"] = coords

        # Store additional statistics for this distribution
        stats["mean"] = np.mean(x)
        stats["median"] = np.median(x)
        stats["min"] = min_val
        stats["max"] = max_val
        stats["quantiles"] = np.atleast_1d(quantile_val)

        # Append to output
        vpstats.append(stats)

    return vpstats


matplotlib.cbook.violin_stats = violin_stats


def plot_trajectory(tsi, gt, window, series, output_dir, ci=0.95, suffix=""):
    """
    Args:
        tsi: shape [predictions, horizon].
        gt: shape [1, horizon].
        series: int,
    """

    H = tsi.shape[-1]  # tsi.shape = [series, horizon]
    x_labels = list(range(H))

    space_per_violin = 0.8
    spv = space_per_violin

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the ground truth data.
    ax.plot(gt, color="red", ls="--")
    # Get the mean of the prediction.
    pred_mean = np.median(tsi, axis=0)  # [horizon, ]
    # Plot the mean of the prediction.
    ax.plot(pred_mean, color="green")
    # Shade out the [q(1-ci), q(ci)] area.
    lower = np.quantile(tsi, q=1 - ci, axis=0)
    upper = np.quantile(tsi, q=ci, axis=0)
    ax.fill_between(x=range(H), y1=lower, y2=upper, alpha=0.3)

    means = np.mean(tsi, axis=0)

    v = ax.violinplot(
        dataset=tsi,
        positions=list(range(H)),
        widths=0.7,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        points=100,
    )
    # Plot mean tick.
    for mu, p in zip(means, range(H)):
        ax.plot([p - 0.15, p], [mu] * 2, c="black")
    # Only keep one half of the violin plot.
    for b in v["bodies"]:
        # Get the center.
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Plot the right half: modify the paths to not go further left than m.
        # To plot the left half set `-np.inf, m`.
        b.get_paths()[0].vertices[:, 0] = np.clip(
            b.get_paths()[0].vertices[:, 0], m, np.inf
        )
        b.set_color("blue")
        b.set_alpha(0.8)
    ax.set_ylabel(f"{int(series)}")
    if not os.path.exists(os.path.join(output_dir, "trajectories")):
        os.mkdir(os.path.join(output_dir, "trajectories"))
    plt.savefig(
        os.path.join(
            output_dir, "trajectories", f"trajectory_{suffix}_{window}_{series}.png"
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_trajectories(preds, targets, draw_series_ids, horizon, output_dir, suffix=""):
    """
    Args:
        preds: shape [windows, series, predictions, horizon].
        targets: shape [windows, series, 1, horizon].
        draw_series_ids: None = all, [] = no series, otherwise draw the
            series with the given ids.
    """
    style = {
        "axes.labelsize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "axes.titlesize": 20,
        "font.serif": "Times New Roman",
    }
    plt.rcParams.update(style)

    num_windows = preds.shape[windows_axis]
    num_series = preds.shape[series_axis]

    if draw_series_ids is None:
        draw_series_ids = range(num_series)

    for window in tqdm.tqdm(range(num_windows)):
        for series in draw_series_ids:
            tsi = preds[window, series, :, :horizon]
            gt = targets[window, series, 0, :horizon]
            plot_trajectory(
                tsi, gt, window, series, output_dir, ci=0.975, suffix=suffix
            )


def plot_trajectory_histogram(tsi, gt, window, var_level, output_dir, ci):
    """
    Args:
        tsi: shape [predictions, horizon].
        gt: shape [horizon, ].
        var_level: float, a quantile level
    """

    series_color = "blue"
    quantiles_color = "red"

    quantiles = np.quantile(tsi, q=var_level, axis=0)
    df = pd.DataFrame(
        {
            "target": gt,
            f"q({var_level:.2f})": quantiles,
            "median": np.median(tsi, axis=0),
            "mean": np.mean(tsi, axis=0),
        }
    )

    horizon_length = len(gt)
    palette = ["blue", "red", "green", "orange"]

    # Create a grid of plots. First, time-series with predictions.
    cols = 6
    rows = (horizon_length // cols) + 1 + (horizon_length % cols > 0)
    fig = plt.figure(figsize=(cols, 0.8 * rows))  # , constrained_layout=True)

    gs = GridSpec(rows + 1, cols)
    gs.update(left=0.05, right=0.9, wspace=0.2, hspace=0.2, top=0.9)
    ax = plt.subplot(gs[:2, :])

    sns.lineplot(data=df, ax=ax, palette=palette)
    ax.set_xlabel("")
    # Move legend above the lineplot.
    h, l = ax.get_legend_handles_labels()
    kw = dict(ncol=4, loc="lower center", frameon=False)
    legend = ax.legend(h, l, bbox_to_anchor=[0.5, 1.0], **kw)
    ax.add_artist(legend)

    # Highlight voilations.
    voil_idx = list(np.where(quantiles <= gt)[0])
    for x in voil_idx:
        ax.axvline(x, color="gray", linestyle="--")
        ax.plot(x, quantiles[x], marker="*", color="black", markersize=10)
    delta = 5
    xticks = list(range(delta, horizon_length, delta))
    xticks += voil_idx

    ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))  # ax.set_xticks(xticks)
    ax.set_xticklabels([f"{xt}" for xt in xticks])

    # Second, create density plots for each timestep.
    series = 0
    for i in range(2, rows + 1):
        for j in range(cols):
            if series >= horizon_length:
                break

            ax = plt.subplot(gs[i, j])

            # Needed to translate axis distance to inches.
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width = bbox.width

            # Compute minimal distance between labels: eps.
            samples = tsi[:, series]
            l, r = np.quantile(samples, 0.005), np.quantile(samples, 0.995)
            eps = 0.09 / width * (r - l)

            # Quantile from the model, q, and the actual realization, s.
            q = np.quantile(samples, var_level)
            s = gt[i]
            if q > s:
                col_mn, col_mx = palette[0], palette[1]
            else:
                col_mn, col_mx = palette[1], palette[0]
            sns.histplot(data=samples, kde=True, ax=ax)
            ax.axvline(s, color=palette[0])
            ax.axvline(q, color=palette[1])
            ax.xaxis.set_major_locator(
                mticker.FixedLocator([])
            )
            ax.set_xticklabels([])

            # Remove y-axis annotations, since it is not that relevant for the density.
            ax.yaxis.set_major_locator(mticker.FixedLocator([]))
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            # If the interquantile distance too small, leave the default settings.
            if r - l > 0.1:
                ax.set_xlim([l, r])
            # Annotate timestep.
            ax.text(
                0.05, 0.95, f"t={series+1}", transform=ax.transAxes, ha="left", va="top"
            )
            series += 1

    if not os.path.exists(os.path.join(output_dir, "histograms")):
        os.mkdir(os.path.join(output_dir, "histograms"))

    plt.savefig(
        os.path.join(output_dir, "histograms", f"histograms_{window}_{series}.png"),
        # bbox_inches='tight')  # to keep the legend
    )
    plt.close()


def plot_trajectories_histograms(
    preds, targets, draw_series_ids, horizon, var_level, output_dir
):
    """
    Args:
        preds: shape [windows, series, predictions, horizon].
        targets: shape [windows, series, 1, horizon]
    """
    style = {
        "axes.labelsize": 8,  # 20,
        "legend.fontsize": 8,  # 14,
        "xtick.labelsize": 8,  # 17,
        "ytick.labelsize": 8,  # 17,
        "axes.titlesize": 8,  # 20,
        "xtick.major.pad": 1,  # distance between xtick and xticklabel
        "font.serif": "Times New Roman",
    }
    plt.rcParams.update(style)

    num_windows = preds.shape[windows_axis]
    num_series = preds.shape[series_axis]

    if draw_series_ids is None:
        draw_series_ids = range(num_series)

    for window in tqdm.tqdm(range(num_windows)):
        for series in draw_series_ids:
            tsi = preds[window, series, :, :horizon]
            gt = targets[window, series, 0, :horizon]
            plot_trajectory_histogram(
                tsi, gt, window, var_level, output_dir, ci=0.975
            )


def coverage(viols_array, var_level=0.9, test="POF"):
    """
    voils_array.shape = [windows, series, quantiles, horizon]

    Coverage tests. They use the fact that when the model is correct, the
    violations are independent identically distributed Bernoulli random variables.
    The tests are two-sided.

    Z test: goodness of fit (`(S - mu) / std` is approximatelly normal).

    Kupiec TUFF (Time Until First Failure) test: likelihood ratio of goemetric
    distributions, evaluated at V, which is the first time the violation happens.

    Kupiec POF (Proportion of Failures) test: likelihood ratio of binomial
    distributions, evaluated at S, which is the number of violations.

    INFO: `var_level` should be adjusted to horizon_length. The expected number of
    voilation (if the model is true) equals `(1.- var_level) * horizon_length`.

    INFO: here we estimate quantiles using MC samples. The quality of the
    estimator depends on number of samples (`pred.shape[0]`), the shape of the
    model's distribution, and the quantile level `var_level`.

    INFO: The tests are asymptotic, hence their quality depend on the length of
    the series.

    INFO: see papers, wiki, mcneil, frey, embrechts,
    or mathworks.com/help/risk/overview-of-var-backtesting.html

    Args:
      prediction: np.array, prediction of the model
      ground_truth: np.array, the actual time series
      var_level: float, VaR confidence level
      test: which test to run
    Returns:
      pvalue: float
    """
    assert test in ("POF", "TUFF", "Z"), "Test not supported."
    if isinstance(var_level, float):
        var_level = np.array([var_level])
    var_level = np.expand_dims(var_level, axis=(0, 1, -1))
    assert viols_array.shape[2] == var_level.shape[2]

    h0_distribution_dict = dict(
        POF=sps.chi2(df=1),
        TUFF=sps.chi2(df=1),
        Z=sps.norm(),
    )
    h0_distribution = h0_distribution_dict[test]

    # Number of windows is axis=0
    num_windows = viols_array.shape[0]
    # Theoretical rate of voilations.
    prob = 1 - var_level  # [1, 1, quantiles, 1]
    # Compute violations over windows axis (axis=0).
    # Shape [1, series, quantiles, horizon].
    viols = np.sum(viols_array, axis=0, keepdims=True).astype(np.float64)
    # Compute frequency.
    freq = viols / num_windows  # [1, series, quantiles, horizon].

    if test == "POF":
        # Binomial distribution (without the binomial coefficient).
        likelihood_data = (freq**viols) * ((1 - freq) ** (num_windows - viols))
        likelihood_binom = (prob**viols) * ((1 - prob) ** (num_windows - viols))
        T = -2.0 * np.log(likelihood_binom / likelihood_data)
    elif test == "TUFF":
        # Time until first violation.
        # Added 1, since tuff=0 results in NaNs
        tuff = (
            np.argmax(viols_array == 1, axis=0)[None, ...] + 1
        )  # (1, series, quantiles, horizon)

        # Geometric distribution.
        likelihood_data = prob * (1 - prob) ** tuff
        likelihood_geom = (1 / tuff) * (1.0 - 1 / tuff) ** (tuff - 1)
        T = -2.0 * np.log(likelihood_geom / likelihood_data)
    else:
        T = np.sqrt(num_windows) * (freq - prob) / np.sqrt(prob * (1 - prob))
    # p_value = 2.0 * np.minimum(h0_distribution.cdf(T), h0_distribution.sf(T))
    p_value = h0_distribution.sf(T)
    return p_value


# Define non-quantile metrics
def mean(preds, targets, windows_axis=None):
    # preds shape: [windows, series, predictions, horizon]
    # targets shape: [windows, series, 1, horizon]
    # output shape: [windows, series, 1, horizon]
    del targets, windows_axis  # Not useful.
    return np.mean(preds, axis=preds_axis, keepdims=True)


def std(preds, targets, windows_axis=None):
    # preds shape: [windows, series, predictions, horizon]
    # targets shape: [windows, series, 1, horizon]
    # output shape: [windows, series, 1, horizon]
    del targets, windows_axis  # Not useful.
    return np.std(preds, axis=preds_axis, keepdims=True)


def mse(preds, targets, windows_axis=None):
    # preds shape: [windows, series, predictions, horizon]
    # targets shape: [windows, series, 1, horizon]
    axes = (preds_axis,)
    if windows_axis is not None:
        axes = axes + (windows_axis,)
    return np.mean((preds - targets) ** 2, axis=axes, keepdims=True)


def mad(preds, targets, windows_axis=None):
    # preds shape: [windows, series, predictions, horizon]
    # targets shape: [windows, series, 1, horizon]
    axes = (preds_axis,)
    if windows_axis is not None:
        axes = axes + (windows_axis,)
    return np.mean(np.abs(preds - targets), axis=axes, keepdims=True)


def elemwise_quantile_loss(preds, targets, alphas):
    """Compute the loss for each quantile level according to
    https://en.wikipedia.org/wiki/Quantile_regression#Sample_quantile.

    Args:
        preds: shape [windows, series, predictions, horizon].
        targets: shape [windows, series, 1, horizon].
        alphas: list of floats, quantile levels.

    Returns:
        loss: shape [windows, series, alphas, horizon].
    """

    quantiles = np.stack(
        [np.quantile(preds, q=alpha, axis=preds_axis) for alpha in alphas],
        axis=preds_axis,
    )  # [windows, series, alphas, horizon]

    delta = targets - quantiles  # [windows, series, alphas, horizon]

    # Indicator if delta is negative.
    ind_delta_neg = (delta < 0).astype(np.float32)  # [windows, series, alphas, horizon]

    # Make alphas the correct shape
    alphas = np.array(alphas)[None, None, :, None]  # [1, 1, alphas, 1]

    # Weighted l1 loss (signs always agree).
    loss = delta * (alphas - ind_delta_neg)  # [windows, series, alphas, horizon]

    return loss


def quantile_loss(normalized_preds, normalized_targets, alphas):
    """Section 5.1 of the Enhancing Locality paper:
    https://arxiv.org/pdf/1907.00235.pdf.

    QR_{s,a,h} = 2 * sum_w L_{w,s,a,h} / normalize(target_{w,s,h}),
    where target normalization is done in `compute_metrics` (see there
    for possible options).

    TODO: comment regarding CRPS.

    Args:
        preds shape: [..., series, predictions, horizon]
        targets shape: [..., series, 1, horizon]

    Returns:
        loss shape: [..., series, alphas, horizon]
    """
    loss = 2 * elemwise_quantile_loss(
        normalized_preds, normalized_targets, alphas
    )  # [series, alphas, horizon]
    return loss


def crps(normalized_preds, normalized_targets):
    alphas = np.linspace(0, 1, num=21, endpoint=True)[1:-1]
    start = time.time()
    ql = quantile_loss(normalized_preds, normalized_targets, alphas)
    print("CRPS computation took ", time.time() - start)
    return np.mean(ql, axis=preds_axis, keepdims=True)


def compute_metrics(
    metrics,
    metric_names,
    sim_data,
    train_data,
    reduce_series=None,
    reduce_horizon=None,
    alphas=None,
    normalize=None,
):
    """
    Args:
        sim_data: (List[Tuple[np.array, np.array): sequence of prediction-target pairs
        reduce_series: list of functions to reduce over series axis.
            E.g., we might be interested in portfolio-level metrics.
        reduce_horizon: list of functions to reduce over horizon axis.
            E.g., we might be interested in predicting max, min, or other
            function of the trajectory.
        reduce_preds: list of functions to reduce over prediction axis.
            We might be interested in various statistics, such as quantiles.
    """
    # Initialize dict of results.
    results = collections.defaultdict(list)

    # Iterate over each simulations for each timewindow.
    assert len(sim_data) == len(train_data)
    for (preds, targets), train_arr in tqdm.tqdm(
        zip(sim_data, train_data), total=len(sim_data)
    ):
        if normalize is not None:
            # train_arr shape: [series, num_train_samples], e.g., `num_train_samples=24*365-1`
            if normalize == "std_train":
                norm_factor = (
                    np.expand_dims(np.std(train_arr, axis=-1), axis=(-1, -2)) + 1e-9
                )  # [series, 1, 1]
            elif normalize == "absmean_train":
                norm_factor = (
                    np.expand_dims(np.mean(np.abs(train_arr), axis=-1), axis=(-1, -2))
                    + 1e-9
                )  # [series, 1, 1]
            elif normalize == "absmean_horizon":
                # targest shape: [series, 1, horizon]
                norm_factor = (
                    np.mean(np.abs(targets), axis=horizon_axis, keepdims=True) + 1e-9
                )  # [series, 1, 1]
            else:
                raise ValueError(f"Normalize method {normalize} does not exist!")

            results["norm_factor"].append(norm_factor)
            preds /= norm_factor
            targets /= norm_factor

        preds = apply_series_and_horizon_reductions(
            preds, reduce_series, reduce_horizon
        )
        targets = apply_series_and_horizon_reductions(
            targets, reduce_series, reduce_horizon
        )

        # Compute non-quantile metrics.
        for metric, metric_name in zip(metrics, metric_names):
            if "quantile" in metric_name:
                result = metric(preds, targets, alphas=alphas)
            else:
                result = metric(preds, targets)
            results[metric_name].append(result)

    for key, value in results.items():
        results[key] = np.stack(value)

    return results


def compute_backtest_metrics(quantiles, targets, alphas, output_dir, plot_strips=False):
    """
    Args:
        quantiles: shape [windows, series, quantiles, horizon]
        targets: shape [windows, series, 1, horizon]
    Returns:
        dict with p-values for each series, quantile, and horizon
        Saves a plot of the p-values.
    """
    violations = (quantiles < targets).astype(
        np.float64
    )  # (windows, series, quantiles, horizon)
    print(f"{violations.shape=}")  # (windows, series, quantiles, horizon)
    pof_predictions = coverage(
        violations, var_level=alphas, test="POF"
    )  # (1, series, quantiles, horizon)

    style = {
        "ytick.labelsize": 8,
        # 'font.serif': 'Times New Roman',
    }
    plt.rcParams.update(style)

    below5pp = mpl.colormaps["RdYlGn"](np.linspace(0, 0.3, 256))
    above5pp = mpl.colormaps["RdYlGn"](np.linspace(0.5, 1, 256))
    all_colors = np.vstack((below5pp, above5pp))
    mix_colors = mcolors.LinearSegmentedColormap.from_list("mix_colors", all_colors)
    divnorm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.05, vmax=1.0)

    print(f"{pof_predictions.shape=}")  # (1, 24, 2, 27)

    num_heatmaps = len(alphas)
    num_series = pof_predictions.shape[series_axis]

    if plot_strips:
        fig = plt.figure(figsize=(20, num_heatmaps))
        gs = GridSpec(num_heatmaps, num_series, figure=fig)

        p = []
        for idx, alpha in enumerate(alphas):
            ax = plt.subplot(gs[idx, :])
            ax.set_ylabel(f"{alpha:.0%}")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(
                pof_predictions[0, :, idx, :].T,
                interpolation="nearest",
                cmap=mix_colors,
                norm=divnorm,
            )
            p.append(ax.get_position().get_points().flatten())
            width = ax.get_position().width  # Width of one heatmap.
            height = ax.get_position().height  # Height of one heatmap.
            # Bbox(x0=0.20162070602451473, y0=0.53, x1=0.8257873726911814, y1=0.88)
            # bbox.x0 = xmin, # bbox.y0 = ymin, # bbox.width = width, # bbox.height = height

        hi = p[0][1] + height - p[1][1]  # Height of the colorbar.
        wi = hi / 100  # Width of the colorbar.
        x0, y0 = p[1][0] + width + wi, p[1][1]  # Left lower corner of the colorbar.
        ax_cbar = fig.add_axes(
            [x0, y0, wi, hi]
        )  # [lower_left_corner_x, lower_left_corner_y, width, height]
        # [a,b,c,d]: (a,b) is the point in southwest corner of the rectangle which we create. c represents width and d represents height of the respective rectangle.
        # ax_cbar.tick_params(rotation=90)
        ticks = [0.0, 0.01, 0.05, 0.3, 0.5, 0.8, 1.00]
        cbar = plt.colorbar(
            im,
            cax=ax_cbar,
            orientation="vertical",
            ticks=ticks,
            norm=divnorm,
            cmap=mix_colors,
        )
        cbar.ax.set_yticklabels([f"{x:.0%}" for x in ticks])

        plt.savefig(os.path.join(output_dir, "pof_predictions.png"))
    return {"pof_p_values": pof_predictions[0]}


def probe_p_values(p_values, probe_at, threshold=0.05):
    """
    Args:
        p_values: a tensor of p_values with shape [series, alphas, horizon].
        probe_at: list of indices at which we compute p_values.
        threshold: float, indicates what a good p_value means (5% by default).
    """
    probed_p_values = p_values[..., probe_at]  # [series, alphas, probe_at]
    # [1, alphas, probe_at]
    probed_p_values = np.mean(
        probed_p_values >= threshold, axis=series_axis, keepdims=True
    )
    return {"probed_p_values": probed_p_values}


def reduce(data, funcs, axis):
    new_data = [np.apply_along_axis(func, axis=axis, arr=data) for func in funcs]
    new_data = np.stack(new_data, axis=axis)
    return new_data


def apply_series_and_horizon_reductions(data, reduce_series, reduce_horizon):
    # Reduce over series (e.g., a portfolio).
    if reduce_series is not None:
        # [reduced_series, preds, horizon]
        data = reduce(data, reduce_series, axis=series_axis)

    # Reduce over horizon (e.g., max, min, etc.).
    if reduce_horizon is not None:
        data = reduce(data, reduce_horizon, axis=horizon_axis)

    return data


def reduce_data_and_compute_quantiles(sim_data, reduce_series, reduce_horizon, alphas):
    preds_list, targets_list, quantiles_list = [], [], []

    # Iterate over each simulations for each timewindow.
    for preds, targets in tqdm.tqdm(sim_data, total=len(sim_data)):
        preds = apply_series_and_horizon_reductions(
            preds, reduce_series, reduce_horizon
        )
        targets = apply_series_and_horizon_reductions(
            targets, reduce_series, reduce_horizon
        )

        preds_list.append(preds)
        targets_list.append(targets)

        # Reduce over predictions (e.g., quantiles).
        quantile_functions = [lambda x, q=alpha: np.quantile(x, q) for alpha in alphas]
        preds_quantiles = reduce(
            preds, quantile_functions, axis=preds_axis
        )  # (series, quantiles, horizon)
        quantiles_list.append(preds_quantiles)

    return np.stack(preds_list), np.stack(targets_list), np.stack(quantiles_list)


def process_simulation_data(dataframes, average_preds=False):
    """
    Returns:
        sim_data (List[Tuple[np.array, np.array): sequence of prediction-target pairs
        horizon: (int): Prediction horizon read from dataframe"""

    # TODO: refactor `horizon`.
    horizon = len([col for col in dataframes[0].columns if "pred" in col]) - 1
    # Preprocess data -> returns List[Tuple[np.array, np.array]]
    N = len(dataframes[0]["ts_id"].unique())
    H = len([col for col in dataframes[0].columns if "pred" in col]) - 1
    P = len(dataframes[0]["pred_id"].unique())

    def extract_preds_targets(df):
        df_numpy = df.to_numpy()
        # df.column() = ['Unnamed: 0', 'ts_id', 'pred_id', 'pred_0', ..., 'pred_23',
        # 'gt_0', ..., 'gt_23', 'eval_start', 'eval_end', 'horizon']
        # Hence, correction by `3` below.
        preds = df_numpy[:, 3 : -(H + 3)].reshape((N, P, H))  # (series, preds, horizon)
        preds = preds.astype(np.float64)
        targets = df_numpy[:, -(H + 3) : -3].reshape((N, P, H))[
            :, :1, :
        ]  # (series, 1, horizon)
        targets = targets.astype(np.float64)

        if average_preds:
            preds = np.mean(preds, axis=preds_axis, keepdims=True)

        return preds, targets

    sim_data = [extract_preds_targets(df) for df in dataframes]

    return sim_data, horizon


@gin.configurable(module="code.evaluation")
def evaluate(
    train_data,
    sim_dataframes,
    metrics=(mean, std, mse, mad, quantile_loss, crps),
    metric_names=(
        "mean",
        "std",
        "mse",
        "mad",
        "quantile_loss",
        "crps",
    ),
    reduce_series=None,
    reduce_horizon=None,
    alphas=(0.5, 0.75, 0.9, 0.95, 0.99),
    save_preds=False,
    output_dir="./out",
    normalize=None,
    average_preds=False,
):
    """
    This function computes metrics, saves them to a file, and draws plots.

    Shapes:
        preds: (windows, series, predictions, horizon)
        targets: (windows, series, 1, horizon)
        non-quantile metrics: (windows, series, 1, horizon)
        quantile metrics: (windows, series, 2, horizon)
        p_values: (series, 2, horizon)

    Args:
        save_preds: Controls if preds as save do a file. Use with caution!
            For a shape (100, 370, 1024, 27) it takes almost 8G of space.
        normalize: str.
    """

    # INFO: dirty hack related to gin. Executable code is kept in the string.
    if isinstance(reduce_series, str):
        reduce_series = eval(reduce_series)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    start = time.time()    
    sim_data, horizon = process_simulation_data(
        sim_dataframes,
        average_preds=average_preds,
    )
    print(f"Finished processing {len(sim_dataframes)} experiments in {time.time() - start} sec.")

    # Metrics and predictions.
    start = time.time()

    if reduce_horizon is not None:
        reduce_horizon = (
            tuple([lambda x, idx=i: x[idx] for i in range(horizon)])
            + reduce_horizon
        )

    print("Computing metrics and predictions...")
    results = compute_metrics(
        metrics=metrics,
        metric_names=metric_names,
        sim_data=sim_data,
        train_data=train_data,
        reduce_series=reduce_series,
        reduce_horizon=reduce_horizon,
        normalize=normalize,
        alphas=alphas,
    )
    end = time.time()
    print(f"Finished computing metrics in {end - start} sec.")

    preds_arr, targets_arr, quantiles_arr = reduce_data_and_compute_quantiles(
        sim_data=sim_data,
        reduce_series=reduce_series,
        reduce_horizon=reduce_horizon,
        alphas=alphas,
    )

    # Backtest metrics (p-values and plots).
    start = time.time()
    print("Plotting p-value strips...")
    p_values = compute_backtest_metrics(
        quantiles=quantiles_arr,
        targets=targets_arr,
        alphas=alphas,
        output_dir=output_dir,
    )
    results.update(p_values)

    # Compute fraction of p-values above 5% probed at
    # short, mid, and long horizons.
    print("Computing statistics for p_values...")
    freq_p_values = probe_p_values(
        results["pof_p_values"],  # [series, alphas, horizon]
        probe_at=(0, horizon // 2, horizon - 1),
        threshold=0.05,
    )
    results.update(freq_p_values)

    print("Saving the resulting dict...")
    if not os.path.exists(os.path.join(output_dir, "results")):
        os.mkdir(os.path.join(output_dir, "results"))
    filename = os.path.join(output_dir, "results", "results.pkl")

    results["preds_quantiles"] = quantiles_arr
    if save_preds:
        print("Saving the predictions...")
        results["preds"] = preds_arr
        results["targets"] = targets_arr

    with open(filename, "wb") as file:
        pickle.dump(results, file, protocol=4)

    end = time.time()
    print(f"Finished plotting in {end - start} sec.")

    # Single-out series that did not pass backtest.
    pof_predictions = results["pof_p_values"]  # [series, alphas, horizon]
    frequency_of_bad_p_values = np.mean(
        pof_predictions < 0.05, axis=horizon_axis, keepdims=True
    )  # [series, alphas, 1]
    num_series = pof_predictions.shape[series_axis]
    num_quantiles = pof_predictions.shape[preds_axis]
    # Create a priority queue with the worst p_values
    queue = []
    for series in range(num_series):
        for idx in range(num_quantiles):
            queue.append((-frequency_of_bad_p_values[series, idx, 0], series))
    queue.sort()
    # Limit the number of plots, e.g., setting first 5 series
    cnt = 0
    bad_indices = []
    for p, v in queue:
        if bad_indices.count(v) == 0:
            bad_indices.append(v)
            cnt += 1
        if cnt >= 5:
            break

    cnt = 0
    good_indices = []
    for p, v in reversed(queue):
        if good_indices.count(v) == 0:
            good_indices.append(v)
            cnt += 1
        if cnt >= 5:
            break

    # Plotting block.
    # Plotting trajectories with violin plots
    start = time.time()
    print("Plotting trajectories with violin plots...")
    plot_trajectories(
        preds=preds_arr,
        targets=targets_arr,
        draw_series_ids=bad_indices,
        horizon=horizon,
        output_dir=output_dir,
        suffix="bad",
    )
    end = time.time()

    plot_trajectories(
        preds=preds_arr,
        targets=targets_arr,
        draw_series_ids=good_indices,
        horizon=horizon,
        output_dir=output_dir,
        suffix="good",
    )
    end = time.time()
    print(f"Finished plotting in {end - start} sec.")

    # Plotting trajectories with seperate histograms
    start = time.time()
    print("Plotting trajectories with individual histograms...")
    plot_trajectories_histograms(
        preds=preds_arr,
        targets=targets_arr,
        draw_series_ids=bad_indices[0:1],
        horizon=horizon,
        var_level=alphas[-1],
        output_dir=output_dir,
    )
    end = time.time()
    print(f"Finished plotting in {end - start} sec.")
