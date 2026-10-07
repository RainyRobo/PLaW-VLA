# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table

_CONSOLE = Console(highlight=False)


def get_console() -> Console:
    return _CONSOLE


def create_progress(*, console: Console | None = None) -> Progress:
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green", pulse_style="bright_blue"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=False),
        console=console or _CONSOLE,
        expand=True,
    )


def render_key_value_panel(title: str, rows: Sequence[tuple[str, Any]]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right", no_wrap=True)
    table.add_column(style="white")
    for key, value in rows:
        table.add_row(str(key), str(value))
    return Panel(table, title=f"[bold]{title}[/bold]", border_style="cyan", box=ROUNDED)


def render_summary_table(title: str, rows: Sequence[tuple[str, Any]]) -> Table:
    table = Table(title=title, box=ROUNDED, header_style="bold cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")
    for key, value in rows:
        table.add_row(str(key), str(value))
    return table
