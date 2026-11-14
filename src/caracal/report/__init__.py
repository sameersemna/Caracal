"""Report model, evidence index and Markdown renderer (docs/03 §5, docs/04 §9)."""

from caracal.report.model import Report, ReportSummary
from caracal.report.render_md import render_markdown

__all__ = ["Report", "ReportSummary", "render_markdown"]
