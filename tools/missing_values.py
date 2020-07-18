import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from common.file_system import ensure_directory


# Matplotlib settings
plt.style.use("classic")

# Seaborn settings
sns.set()
sns.set(font_scale=1.0)
sns.set_style({"font.family": "serif", "font.serif": ["Times New Roman"]})


def explore_missing_data(df, fig_show=False, save_as="tmp.pdf", **fig_settings):
    """
    Explore missing data values

    This function explores missing values in an input pandas DataFrame. It assumes
    that <df> has the following structure: rows: observations, columns: features.
    Names of the observations are set as the index. It stores a graph with the
    missing values to <save_as> path, and if <fig_show> is set to True, it also
    shows it. Figure settings can be specified in <fig_settings>. If it is not
    specified, the following settings are used:

        {
            "fig_size": (16, 16),
            "fig_cmap": "Greys",

            "fig_ticks_x": np.arange(0, len(df.columns), 1),
            "fig_ticks_y": np.arange(0, len(df.index), 1),

            "line_width": 0.3,
            "line_color": "#c8d6e5"
        }

    Parameters
    ----------

    df : pandas DataFrame
        pandas DataFrame (rows=observations, cols=features), index=observations

    fig_show : bool, optional, default True
        boolean flag for figure showing

    fig_settings : dict, optional, default (see above)
        dict with the figure settings

    save_as : str, optional, default "tmp.pdf"
        str with the full-path to store the figure with the missing values
    """

    # Set the figure settings
    fig_settings = fig_settings if fig_settings else {

        # Figure
        "fig_size": (16, 16),
        "fig_cmap": "Greys",

        # Ticks
        "fig_ticks_x": np.arange(0, len(df.columns), 1),
        "fig_ticks_y": np.arange(0, len(df.index), 1),

        # Lines
        "line_width": 0.3,
        "line_color": "#c8d6e5"
    }

    if "fig_size" not in fig_settings:
        fig_settings.update({"fig_size": (16, 16)})
    if "fig_cmap" not in fig_settings:
        fig_settings.update({"fig_cmap": "Greys"})
    if "fig_ticks_x" not in fig_settings:
        fig_settings.update({"fig_ticks_x": np.arange(0, len(df.columns), 1)})
    if "fig_ticks_y" not in fig_settings:
        fig_settings.update({"fig_ticks_y": np.arange(0, len(df.index), 1)})
    if "line_width" not in fig_settings:
        fig_settings.update({"line_width": 0.3})
    if "line_color" not in fig_settings:
        fig_settings.update({"line_color": "#c8d6e5"})

    # Prepare the missing values
    missing_values = df.isnull()

    # Create the figure
    fig = plt.figure(figsize=fig_settings.get("fig_size"))

    # Add the axes
    ax = fig.add_subplot(1, 1, 1)

    # Plot the graph
    sns.heatmap(
        missing_values,
        cbar=False,
        cmap=fig_settings.get("fig_cmap"),
        linewidths=fig_settings.get("line_width"),
        linecolor=fig_settings.get("line_color"))

    # Set up the ticks and labels
    ax.set_xticks(fig_settings.get("fig_ticks_x"))
    ax.set_yticks(fig_settings.get("fig_ticks_y"))
    ax.set_xticklabels(list(df.columns))
    ax.set_yticklabels(list(df.index))
    plt.tick_params(top=False, bottom=True, left=True, right=False)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Set up the final adjustments
    ax.set_title("Missing values")
    plt.tight_layout()
    ax.grid()

    # Save the graph
    ensure_directory(save_as)
    plt.savefig(save_as)

    # Show the graph
    if fig_show and np.sum(np.sum(missing_values.values)):
        plt.show()
    else:
        plt.close()

    # Return the figure and axes
    return fig, ax
