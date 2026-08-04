
import matplotlib.pyplot as plt

def set_dark_style():
    plt.style.use("dark_background")

    plt.rcParams['axes.grid'] = True
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["lines.linewidth"] = 1
