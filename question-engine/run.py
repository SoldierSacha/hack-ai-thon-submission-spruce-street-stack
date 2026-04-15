import argparse


def main():
    parser = argparse.ArgumentParser(prog="run.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Run the full enrichment pipeline.")
    p_build.add_argument("--limit", type=int, default=None,
                         help="Only enrich the first N reviews (smoke testing).")
    p_build.add_argument("--workers", type=int, default=8,
                         help="Parallel LLM workers.")

    args = parser.parse_args()

    if args.cmd == "build":
        from src.question_engine import build
        build(limit_reviews=args.limit, max_workers=args.workers)


if __name__ == "__main__":
    main()
