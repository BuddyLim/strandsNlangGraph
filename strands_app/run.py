import argparse
import sys

from common.config import settings
from common.types import ResearchReport, ResearchRequest
from strands_app.research import run_research


def format_report(report: ResearchReport) -> str:
    """Render a report as readable CLI text."""
    lines = [f"# Research: {report.question}", "", "## Summary", report.summary, ""]
    for f in report.findings:
        status = "" if f.ok else " (failed)"
        lines += [f"## {f.subtopic}{status}", f.findings, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strands research assistant")
    parser.add_argument("question", help="the research question")
    parser.add_argument("--subtopics", type=int, default=settings.n_subtopics,
                        help="number of subtopics to fan out into")
    parser.add_argument("--grounded", action="store_true",
                        help="use Gemini native Google Search instead of the mock tool")
    args = parser.parse_args(argv)

    request = ResearchRequest(question=args.question, n_subtopics=args.subtopics)
    try:
        report = run_research(request, grounded=args.grounded)
    except RuntimeError as exc:
        # Config failures (e.g. missing key) — clean message, no stack trace.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: never dump a stack trace
        # Covers model throttling / transient provider errors surfaced by Strands.
        print(f"research failed: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
