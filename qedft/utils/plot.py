"""Plotting utilities."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


class PlotStyle:
    """Reusable plotting style settings for scientific visualizations."""

    def __init__(
        self,
        palette_name="plasma_r",
        palette_size=10,
        font_size_medium=12,
        font_size_large=13,
        use_tex=False,
    ):
        """Initialize plot style parameters.

        Parameters
        ----------
        palette_name : str
            Name of the seaborn color palette
        palette_size : int
            Number of colors in the palette
        font_size_medium : int
            Standard font size for labels, ticks
        font_size_large : int
            Font size for titles
        use_tex : bool
            Whether to use LaTeX rendering
        """
        self.palette_name = palette_name
        self.palette_size = palette_size
        self.font_size_medium = font_size_medium
        self.font_size_large = font_size_large
        self.use_tex = use_tex
        self.palette = sns.color_palette(palette_name, palette_size)

        # Apply global style settings
        self.apply_global_style()

    def apply_global_style(self):
        """Apply global matplotlib style settings."""
        plt.rc("font", size=self.font_size_medium)
        plt.rc("axes", titlesize=self.font_size_large)
        plt.rc("axes", labelsize=self.font_size_medium)
        plt.rc("xtick", labelsize=self.font_size_medium)
        plt.rc("ytick", labelsize=self.font_size_medium)
        plt.rc("legend", fontsize=self.font_size_medium)
        plt.rc("figure", titlesize=self.font_size_medium)
        plt.rc("text", usetex=self.use_tex)

    def setup_axis(
        self,
        ax,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        xscale="linear",
        yscale="linear",
        xtick_spacing=None,
    ):
        """Apply consistent styling to an axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to style
        xlim, ylim : tuple, optional
            x and y axis limits
        xlabel, ylabel : str, optional
            Axis labels
        xscale, yscale : str, optional
            Scale for each axis ('linear', 'log', etc.)
        xtick_spacing : float, optional
            Spacing for major x ticks
        """
        # Set scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # Set limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Set tick directions
        ax.tick_params(axis="both", direction="in")
        if xscale == "log":
            ax.tick_params(axis="x", which="minor", direction="in")
        if yscale == "log":
            ax.tick_params(axis="y", which="minor", direction="in")

        # Set tick spacing
        if xtick_spacing is not None:
            ax.xaxis.set_major_locator(mticker.MultipleLocator(base=xtick_spacing))

        if yscale == "log":
            ax.yaxis.set_major_locator(mticker.LogLocator(numticks=3))
            ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=3, subs="auto"))

    def add_reference_line(
        self,
        ax,
        horizontal=None,
        vertical=None,
        color="k",
        linestyle="--",
        linewidth=0.5,
        alpha=0.3,
    ):
        """Add reference lines to the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to add lines to
        horizontal : float or list, optional
            y-values for horizontal lines
        vertical : float or list, optional
            x-values for vertical lines
        color, linestyle, linewidth, alpha : float
            Line style parameters
        """
        # Handle horizontal lines
        if horizontal is not None:
            if not isinstance(horizontal, (list, tuple, np.ndarray)):
                horizontal = [horizontal]
            for h in horizontal:
                ax.axhline(
                    y=h,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                )

        # Handle vertical lines
        if vertical is not None:
            if not isinstance(vertical, (list, tuple, np.ndarray)):
                vertical = [vertical]
            for v in vertical:
                ax.axvline(
                    x=v,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                )

    def add_highlight_region(
        self,
        ax,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        color="0.85",
        alpha=0.5,
        label=None,
    ):
        """Add a highlighted region to the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to add highlight to
        xmin, xmax, ymin, ymax : float, optional
            Region boundaries
        color : str
            Color for the highlighted region
        alpha : float
            Transparency
        label : str, optional
            Label for the legend
        """
        if ymin is not None and ymax is not None:
            ax.axhspan(ymin, ymax, color=color, alpha=alpha, label=label)
        elif xmin is not None and xmax is not None:
            ax.axvspan(xmin, xmax, color=color, alpha=alpha, label=label)

    def get_palette_color(self, palette_name=None, color_idx=0, palette_size=None):
        """Get a color from a specified palette.

        Parameters
        ----------
        palette_name : str, optional
            Name of the seaborn color palette. If None, uses the instance's palette.
        color_idx : int, optional
            Index of the color in the palette.
        palette_size : int, optional
            Size of the palette. If None, uses the instance's palette_size.

        Returns
        -------
        tuple
            RGB color tuple
        """
        if palette_name is None:
            # Use the instance's palette
            if color_idx < len(self.palette):
                return self.palette[color_idx]
            else:
                raise IndexError(
                    f"Color index {color_idx} out of range for palette of size {len(self.palette)}",
                )
        else:
            # Create a new palette and return the color
            size = palette_size if palette_size is not None else self.palette_size
            palette = sns.color_palette(palette_name, size)
            if color_idx < len(palette):
                return palette[color_idx]
            else:
                raise IndexError(
                    f"Color index {color_idx} out of range for palette of size {len(palette)}",
                )

    def plot_curve(
        self,
        ax,
        x,
        y,
        label=None,
        color=None,
        color_idx=None,
        color_intensity=None,
        palette_name=None,
        linestyle="-",
        linewidth=2,
        marker=None,
        markersize=None,
    ):
        """Plot a curve with consistent styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to plot on
        x, y : array-like
            Data to plot
        label : str, optional
            Label for the legend
        color : str, optional
            Line color
        color_idx : int, optional
            Index into the palette to select a color
        color_intensity : float, optional
            Intensity factor for this specific curve (0.0-1.0)
            Lower values make colors less intense/lighter
            If None, uses the global color_intensity
        palette_name : str, optional
            Name of a seaborn palette to use instead of the default
        linestyle, linewidth, marker, markersize :
            Line and marker style parameters
        """
        # Determine color
        if color is None and color_idx is not None:
            if palette_name is not None:
                # Get color from specified palette
                base_color = self.get_palette_color(palette_name, color_idx)
            else:
                # Use the instance's palette
                base_color = self.palette[color_idx]

            # Apply per-curve intensity if specified
            if color_intensity is not None and color_intensity < 1.0:
                # Clamp intensity between 0 and 1
                intensity = max(0.0, min(1.0, color_intensity))
                # Adjust color intensity for this specific curve
                color = tuple(c * intensity + (1 - intensity) for c in base_color)
            else:
                color = base_color

        # Plot the curve
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
        )


if __name__ == "__main__":

    # Same as paper
    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = 0.5 * np.cos(x)
    y3 = np.exp(-0.2 * x) * np.sin(x)  # Damped sine

    # Error data
    error1 = 0.05 * np.random.rand(len(x))
    error2 = 0.1 * np.random.rand(len(x))

    # Initialize plot style
    style = PlotStyle(palette_name="plasma_r", palette_size=10)

    # Create figure with 3 subplots
    fig, axs = plt.subplots(
        3,
        sharex=False,
        sharey=False,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    fig.set_size_inches(4, 6)

    # Plot limits
    xlim = (0, 10)

    # Plot 1: Main curves
    style.setup_axis(
        axs[0],
        xlim=xlim,
        ylim=(-1.2, 1.2),
        ylabel="Signal",
        xtick_spacing=2.0,
    )

    style.plot_curve(
        axs[0],
        x,
        y1,
        label="Sine",
        color_idx=1,
        linewidth=3,
        palette_name="mako_r",
    )
    style.plot_curve(
        axs[0],
        x,
        y2,
        label="Cosine",
        color_idx=1,
        color_intensity=0.5,
        linewidth=2,
    )
    style.plot_curve(
        axs[0],
        x,
        y3,
        label="Damped",
        color="gray",
        linestyle="-.",
    )

    # Add reference points
    reference_points = [2, 5, 8]
    ref_idx = [np.argmin(np.abs(x - p)) for p in reference_points]

    style.plot_curve(
        axs[0],
        x[ref_idx],
        y1[ref_idx],
        label="Reference",
        color="black",
        marker="o",
        markersize=5,
        linestyle="",
    )

    axs[0].legend(bbox_to_anchor=(1.0, 0.8), framealpha=0.5, frameon=False)

    # Plot 2: Error 1
    style.setup_axis(
        axs[1],
        xlim=xlim,
        ylim=(0.001, 0.2),
        ylabel="Error 1",
        yscale="log",
        xtick_spacing=2.0,
    )

    style.plot_curve(
        axs[1],
        x,
        error1 + 0.01,
        color_idx=1,
        linewidth=2,
    )

    style.add_reference_line(axs[1], horizontal=0.03)
    style.add_highlight_region(
        axs[1],
        ymin=0,
        ymax=0.03,
        label="threshold",
    )
    style.add_reference_line(axs[1], vertical=reference_points)

    # Plot 3: Error 2
    style.setup_axis(
        axs[2],
        xlim=xlim,
        ylim=(0.01, 0.3),
        ylabel="Error 2",
        xlabel="X-axis",
        yscale="log",
        xtick_spacing=2.0,
    )

    style.plot_curve(
        axs[2],
        x,
        error2 + 0.02,
        color_idx=3,
        linewidth=2,
    )

    style.add_reference_line(axs[2], vertical=reference_points)

    # Finalize and save
    fig.tight_layout()
    fig.savefig("simple_example.pdf", bbox_inches="tight")
    plt.show()
