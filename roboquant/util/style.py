
import matplotlib.pyplot as plt


def __set_defaults():
    plt.rcParams['axes.grid'] = True
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["lines.linewidth"] = 1

def set_dark_style():
    """Set a dark style for matplotlib plots. This function modifies the default
    matplotlib style to use a dark background.

    It also sets some default parameters for the grid, figure size,
    and line width.

    This function should be called before creating any plots to ensure
    that the dark style is applied to all subsequent plots.
    """

    plt.style.use("dark_background")
    __set_defaults()

def set_light_style():
    """Set a llight style for matplotlib plots. This function modifies the default
    matplotlib style to use a dark background.

    It also sets some default parameters for the grid, figure size,
    and line width.

    This function should be called before creating any plots to ensure
    that the dark style is applied to all subsequent plots.
    """

    plt.style.use("default")
    __set_defaults()
