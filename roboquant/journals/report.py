import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import base64
import io


_open_html_snippet = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>
"""

_close_html_snippet = """
</body>
</html>
"""


class Report:
    """Collect matplotlib figures and save them into a single HTML or PDF file."""

    def __init__(self):
        self._figures: list[plt.Figure] = []

    def add_figure(self, fig: plt.Figure) -> None:
        """Add a matplotlib figure to the report."""
        self._figures.append(fig)

    def add_df(self, df, title: str | None = None) -> None:
        """Add a df to the report.
        The dataframe will be rendered using matplotlib.
        """
        fig, ax = plt.subplots(figsize=(11, 8))

        ax.table(cellText=df.values, colLabels=df.columns, loc="center")  # type: ignore
        ax.axis("tight")
        ax.axis("off")
        if title:
            fig.suptitle(title)
        self.add_figure(fig)

    def add_current_figure(self) -> None:
        """Add the current matplotlib figure to the report."""
        if plt.get_fignums():
            self._figures.append(plt.gcf())

    def save_as_pdf(self, filepath: str | Path, close_figures: bool = True) -> None:
        """Save all added figures into a single PDF file. Each chart will on its
        own page.

        Parameters
        ----------
        filepath: Destination path for the PDF file.
        close_figures: If True, close each figure after writing it to the PDF.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(str(filepath)) as pdf:
            for fig in self._figures:
                pdf.savefig(fig)
                if close_figures:
                    plt.close(fig)

    @staticmethod
    def __chart_to_html(fig, **kwargs):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", **kwargs)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return (
            f'<img class="chart" src="data:image/png;base64,{b64}" '
            + 'style="max-width:100%; margin-bottom:20px; display:block;" />'
        )

    def save_as_html(self, filepath: str | Path, close_figures: bool = True, **kwargs):
        """Save as a single HTML file"""

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(_open_html_snippet)
            for fig in self._figures:
                snippet = self.__chart_to_html(fig, **kwargs)
                f.write(snippet)
                if close_figures:
                    plt.close(fig)
            f.write(_close_html_snippet)

    def __len__(self) -> int:
        return len(self._figures)
