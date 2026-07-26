"""Tema visualisasi bersama untuk seluruh grafik FurnaceGuard AI.

Palet dan aturan mark mengikuti satu sistem desain yang tervalidasi:
slot kategorikal ditetapkan berurutan dan tidak pernah diputar ulang,
magnitudo memakai satu hue (biru) dari terang ke gelap, dan sumbu serta
grid dibuat resesif.

Tiga slot kategorikal pertama lolos seluruh gerbang keterbacaan buta
warna. Slot keempat masih aman untuk bentuk berdampingan (batang, garis)
selama setiap deret diberi label langsung — dua slot punya kontras di
bawah 3:1 terhadap permukaan terang, sehingga label langsung wajib ada,
bukan pilihan.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# --------------------------------------------------------------------
# Palet
# --------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Slot kategorikal, dipakai berurutan. Jangan diputar, jangan ditambah.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

#: Ramp sequential satu hue untuk magnitudo (terang -> gelap).
SEQUENTIAL = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

#: Warna status — tidak pernah dipakai sebagai warna deret.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Abu-abu untuk deret konteks pada grafik bertipe penekanan.
DEEMPHASIS = "#c3c2b7"

FONT_FAMILY = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def apply_theme() -> None:
    """Pasang gaya matplotlib global."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": 10,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.dpi": 120,
        }
    )


def sequential_colors(count: int) -> list[str]:
    """Ambil ``count`` langkah ramp sequential, terang ke gelap.

    Untuk bar chart magnitudo: batang terbesar mendapat langkah tergelap.
    """
    if count <= 0:
        return []
    if count == 1:
        return [SEQUENTIAL[4]]
    # Sisakan langkah paling terang untuk nilai mendekati nol saja.
    usable = SEQUENTIAL[1:]
    step = (len(usable) - 1) / (count - 1)
    return [usable[round(index * step)] for index in range(count)]


def style_axes(
    axes: plt.Axes,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid_axis: str | None = "y",
) -> None:
    """Terapkan judul, label, dan grid resesif pada satu sumbu."""
    if title:
        axes.set_title(
            title,
            loc="left",
            fontsize=12.5,
            fontweight="600",
            color=TEXT_PRIMARY,
            pad=18 if subtitle else 10,
        )
    if subtitle:
        # Subtitle dibungkus manual — matplotlib tidak melipat teks, dan
        # subtitle panjang akan meluber keluar gambar.
        wrapped = textwrap.fill(subtitle, width=104)
        line_count = wrapped.count("\n") + 1
        if title:
            axes.set_title(
                title,
                loc="left",
                fontsize=12.5,
                fontweight="600",
                color=TEXT_PRIMARY,
                pad=10 + 13 * line_count,
            )
        axes.text(
            0.0,
            1.015,
            wrapped,
            transform=axes.transAxes,
            fontsize=9.5,
            color=TEXT_SECONDARY,
            va="bottom",
            linespacing=1.35,
        )
    if xlabel:
        axes.set_xlabel(xlabel, fontsize=9.5, labelpad=8)
    if ylabel:
        axes.set_ylabel(ylabel, fontsize=9.5, labelpad=8)
    if grid_axis:
        axes.grid(axis=grid_axis, linewidth=0.8, color=GRIDLINE, zorder=0)
        axes.set_axisbelow(True)


def integer_axis(axes: plt.Axes, which: str = "y") -> None:
    """Paksa tick bilangan bulat — jumlah event tidak punya nilai pecahan."""
    locator = MaxNLocator(integer=True, nbins=6)
    if which == "y":
        axes.yaxis.set_major_locator(locator)
    else:
        axes.xaxis.set_major_locator(locator)


def label_bars_h(
    axes: plt.Axes,
    bars: Any,
    values: list[float],
    *,
    fmt: str = "{:,.0f}",
    pad: float = 0.0,
) -> None:
    """Label langsung di ujung batang horizontal.

    Wajib: dua slot kategorikal berada di bawah kontras 3:1 terhadap
    permukaan terang, sehingga identitas tidak boleh bergantung pada
    warna saja.
    """
    span = max(values) if values else 1.0
    offset = pad or span * 0.012
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_SECONDARY,
        )


def label_bars_v(
    axes: plt.Axes,
    bars: Any,
    values: list[float],
    *,
    fmt: str = "{:,.0f}",
) -> None:
    """Label langsung di atas batang vertikal."""
    span = max(values) if values else 1.0
    offset = span * 0.02
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )


def add_source_note(figure: plt.Figure, note: str) -> None:
    """Catatan sumber data di kaki gambar."""
    # Ditempatkan di bawah batas gambar; ``bbox_inches="tight"`` saat
    # penyimpanan akan menariknya masuk tanpa menabrak label sumbu.
    figure.text(
        0.01,
        -0.02,
        textwrap.fill(note, width=120),
        fontsize=7.5,
        color=TEXT_MUTED,
        va="top",
        ha="left",
    )


def save(figure: plt.Figure, path: Path) -> Path:
    """Simpan gambar dan tutup untuk membebaskan memori."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path
