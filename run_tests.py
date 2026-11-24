"""Run discovery and the original Theta* suite together from any directory."""
import argparse
import io
from pathlib import Path
import sys
from time import perf_counter
import unittest

ROOT = Path(__file__).resolve().parent


class MethodResult(unittest.TextTestResult):
    passed = 0

    def addSuccess(self, test):
        self.passed += 1
        super().addSuccess(test)


def owners(events):
    """Multiple unsuccessful subtests can belong to the same method."""
    return {getattr(test, "test_case", test) for test, _ in events}


def methods(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from methods(item)
        else:
            yield item


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verbose", action="store_true")
    args = cli.parse_args()
    begin = perf_counter()
    groups = {name: unittest.TestSuite() for name in ("A*", "Theta*", "ACO", "Integration")}
    discovered = unittest.TestLoader().discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    for test in methods(discovered):
        module = type(test).__module__
        group = {"tests.test_astar": "A*", "tests.test_aco": "ACO"}.get(module, "Integration")
        groups[group].addTest(test)
    # Discovery places the original suite's directory on sys.path for its local import.
    groups["Theta*"] = unittest.TestLoader().discover(str(ROOT / "src/theta_star"), "unit_test.py")

    print("PATHFINDING TEST REPORT")
    counts = []
    for name, suite in groups.items():
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2 if args.verbose else 0,
                                         resultclass=MethodResult)
        result = runner.run(suite)
        if args.verbose or not result.wasSuccessful():
            print(stream.getvalue())
        errors = owners(result.errors)
        failures = (owners(result.failures) | set(result.unexpectedSuccesses)) - errors
        skipped = owners(result.skipped) - failures - errors
        if result.expectedFailures:
            print(f"{name} expected failures: {len(result.expectedFailures)}")
        counts.append((name, result.passed, len(failures), len(errors), len(skipped)))
    print(f"{'Suite':14} {'Passed':>7} {'Failed':>7} {'Errors':>7} {'Skipped':>7}")
    totals = tuple(map(sum, zip(*(row[1:] for row in counts))))
    for name, passed, failed, errors, skipped in [*counts, ("TOTAL", *totals)]:
        print(f"{name:14} {passed:7} {failed:7} {errors:7} {skipped:7}")
    ok = not (totals[1] or totals[2])
    print(f"Status: {'PASS' if ok else 'FAIL'}\nElapsed time: {perf_counter()-begin:.3f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
