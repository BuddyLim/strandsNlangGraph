import argparse
import sys

from common.config import settings
from common.types import ResearchReport, ResearchRequest
from strands_app.research import run_research


def format_findings(report: ResearchReport) -> str:
    """Render just the per-subtopic findings section.

    Kept separate from the summary because in streaming mode the summary has
    already streamed live, so only this recap is printed afterwards. Because the
    fan-out is LLM-driven, the coordinator may answer directly without delegating —
    in that case the section says so explicitly rather than rendering blank.
    """
    n = len(report.findings)
    if n == 0:
        return (
            "## Sub-agent findings\n"
            "_(coordinator answered directly without delegating to sub-agents)_"
        )
    lines = [f"## Sub-agent findings ({n} spawned)", ""]
    for f in report.findings:
        status = "" if f.ok else " (failed)"
        lines += [f"### {f.subtopic}{status}", f.findings, ""]
    return "\n".join(lines).rstrip()


def format_report(report: ResearchReport) -> str:
    """Render the full report: title, summary, and the per-subtopic findings."""
    return "\n".join(
        [
            "",
            f"# Research: {report.question}",
            "",
            "## Summary",
            report.summary,
            "",
            format_findings(report),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strands research assistant")
    parser.add_argument("question", help="the research question")
    parser.add_argument(
        "--subtopics",
        type=int,
        default=settings.n_subtopics,
        help="number of subtopics to fan out into",
    )
    parser.add_argument(
        "--grounded",
        action="store_true",
        help="use Gemini native Google Search instead of the mock tool",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show Strands' raw tool-call trace instead of the curated streaming output",
    )
    args = parser.parse_args(argv)

    request = ResearchRequest(question=args.question, n_subtopics=args.subtopics)
    try:
        report = run_research(request, grounded=args.grounded, verbose=args.verbose)
    except RuntimeError as exc:
        # Config failures (e.g. missing key) — clean message, no stack trace.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: never dump a stack trace
        # Covers model throttling / transient provider errors surfaced by Strands.
        print(f"research failed: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        # The raw trace already streamed above; print the full structured recap.
        print(format_report(report))
    else:
        # Subtopic progress + summary already streamed live; end the streamed line
        # with a blank separator, then print the per-subtopic findings recap.
        print("\n")
        print(format_findings(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
