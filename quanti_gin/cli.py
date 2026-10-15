import argparse
import logging
from pathlib import Path

from quanti_gin.benchmarks import BENCHMARKS, get_benchmark
from quanti_gin.methods import METHODS, import_custom_method
from quanti_gin.runner import run_benchmark, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="quanti_gin", description="benchmark suite for sunrise methods"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list the predefined benchmarks and methods")

    run = subparsers.add_parser("run", help="run methods on a benchmark")
    run.add_argument(
        "benchmark",
        choices=list(BENCHMARKS),
        metavar="benchmark",
        help="name of the benchmark",
    )
    run.add_argument(
        "--methods",
        "-M",
        nargs="+",
        choices=list(METHODS),
        metavar="method",
        help="methods to run, the defaults of the benchmark if not given",
    )
    run.add_argument(
        "--custom-method",
        type=Path,
        action="append",
        default=[],
        help="python file with a run_optimization(molecule, coordinates, **kwargs) function",
    )
    run.add_argument(
        "--reference", default="fci", help="reference energy, 'fci' by default"
    )
    run.add_argument(
        "--limit", "-n", type=int, help="only run the first n entries of the benchmark"
    )
    run.add_argument(
        "--fidelity",
        "-F",
        action="store_true",
        help="calculate the fidelity between the true ground state and the optimized state",
    )
    run.add_argument(
        "--jobs", "-j", type=int, help="number of processes to execute the jobs in"
    )
    run.add_argument("--output", "-O", type=Path, help="output file name")

    generate = subparsers.add_parser(
        "generate", help="(re)compute the reference energies of benchmarks"
    )
    generate.add_argument(
        "benchmarks", nargs="*", help="benchmarks to generate, all if not given"
    )
    generate.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the datasets, the package data by default",
    )

    plot = subparsers.add_parser("plot", help="plot the results of a benchmark run")
    plot.add_argument("results", type=Path)
    plot.add_argument("--methods", "-M", nargs="+")

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.command == "list":
        print("benchmarks:")
        for name, benchmark in BENCHMARKS.items():
            status = "" if benchmark.path.exists() else " (no dataset)"
            print(
                f"  {name:18s} {benchmark.description}, references: {', '.join(benchmark.references)}{status}"
            )
        print("methods:")
        for name in METHODS:
            print(f"  {name}")

    elif args.command == "run":
        methods = (args.methods or []) + [
            import_custom_method(path) for path in args.custom_method
        ]
        results = run_benchmark(
            args.benchmark,
            methods=methods,
            reference=args.reference,
            limit=args.limit,
            calculate_fidelity=args.fidelity,
            threads=args.jobs,
        )
        path = args.output or Path(f"{args.benchmark}_results.csv")
        results.to_csv(path, index=False)
        if "error" in results:
            print(summarize(results).to_string())
        if "exception" in results:
            print(
                f"{results['exception'].notna().sum()} job(s) failed, see the exception column in {path}"
            )
        print(f"results written to {path}")

    elif args.command == "generate":
        names = args.benchmarks or list(BENCHMARKS)
        for name in names:
            benchmark = get_benchmark(name)
            path = None if args.output_dir is None else args.output_dir / f"{name}.csv"
            benchmark.generate(path=path)

    elif args.command == "plot":
        from quanti_gin.visualization import plot_results

        plot_results(str(args.results), methods=args.methods)


if __name__ == "__main__":
    main()
