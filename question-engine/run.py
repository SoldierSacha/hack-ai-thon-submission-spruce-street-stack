import argparse


def main():
    parser = argparse.ArgumentParser(prog="run.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Run the full enrichment pipeline.")
    p_build.add_argument("--limit", type=int, default=None,
                         help="Only enrich the first N reviews (smoke testing).")
    p_build.add_argument("--workers", type=int, default=8,
                         help="Parallel LLM workers.")

    p_ask = sub.add_parser("ask", help="Interactive: submit a review, answer follow-ups, see updates.")
    p_ask.add_argument("property_id", help="The eg_property_id to target (or 'list' to see all).")
    p_ask.add_argument("review_text", nargs="?", default="",
                       help="The review text (can be empty, use quotes).")
    p_ask.add_argument("--today", help="Override today's date (YYYY-MM-DD). Default = max review date in data.",
                       default=None)

    args = parser.parse_args()

    if args.cmd == "build":
        from src.question_engine import build
        build(limit_reviews=args.limit, max_workers=args.workers)
    elif args.cmd == "ask":
        from src.question_engine import run_ask
        run_ask(args.property_id, args.review_text, args.today)


if __name__ == "__main__":
    main()
