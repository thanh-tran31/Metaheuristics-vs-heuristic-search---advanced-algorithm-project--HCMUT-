"""Benchmark pathfinding on fixed MovingAI scenarios. Run as a module from the root."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import fmean, median, stdev
import xml.etree.ElementTree as ET
import subprocess
from time import perf_counter_ns

from src import astar, theta_star
from src.aco import ACOSolver
from src.common.parser import MovingAIParser
from src.common.validation import validate_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "movingai"
MANIFEST = ROOT / "experiments" / "configs" / "selected_scenarios.json"
MAPS = {"random": "random-32-32-20.map", "maze": "maze-32-32-2.map"}
ALGORITHMS = ("astar", "theta_star", "aco")
NAMES = {"astar": "A*", "theta_star": "Theta*", "aco": "ACO"}
FAMILIES = {"astar": "Heuristic", "theta_star": "Heuristic", "aco": "Metaheuristic"}
COLORS = {"astar": "#2563b0", "theta_star": "#087d75", "aco": "#b9540a"}
RESULTS = ROOT / "experiments" / "results"


def paths(case):
    family = DATA / case["family"]
    return family / "map" / case["map_name"], family / "scenarios" / case["scenario_file"]


def select_scenarios():
    """Take the midpoint rank of each of 12 equal-count bins per map."""
    cases = []
    for family, name in MAPS.items():
        directory = DATA / family
        records = {}
        for file in sorted((directory / "scenarios").glob("*.scen")):
            parser = MovingAIParser(directory / "map" / name, file)
            for index in range(parser.scenario_count):
                info = parser.get_scenario_info(index)
                records.setdefault((info["start"], info["goal"]),
                                   (info["optimal_length"], file.name, index))
        ranked = sorted(records.values())
        if len(ranked) < 12:
            raise ValueError(f"Too few distinct scenarios for {family}")
        for bin_index in range(12):
            _, filename, index = ranked[(2 * bin_index + 1) * len(ranked) // 24]
            cases.append(dict(family=family, map_name=name, scenario_file=filename,
                              scenario_index=index))
    return cases


def repetition_counts(profile, algorithms):
    deterministic, stochastic = (5, 30) if profile == "full" else (2, 2)
    return {a: stochastic if a == "aco" else deterministic for a in algorithms}


def run_algorithm(algorithm, case, parser, repetition):
    index = case["scenario_index"]
    grid, start, goal = parser.get_input(index)
    original = [row.copy() for row in grid]
    reference = parser.get_scenario_info(index)["optimal_length"]
    row = dict(algorithm=algorithm, algorithm_family=FAMILIES[algorithm],
               map_name=case["map_name"], scenario_file=case["scenario_file"],
               scenario_index=index, start_x=start[0], start_y=start[1], goal_x=goal[0],
               goal_y=goal[1], repetition=repetition, seed=None, runtime_ms=None, outcome="error",
               path_valid=None, path_length=None, reference_length=reference,
               matches_grid_reference=None, grid_optimality_gap_pct=None, relative_to_astar_pct=None,
               expanded_nodes=None, generated_nodes=None, evaluated_nodes=None,
               ants=None, rounds=None, error="")
    path = None
    try:
        if algorithm == "aco":
            solver = ACOSolver(seed=repetition-1)
            row.update(seed=solver.seed, ants=solver.ants, rounds=solver.rounds)
            solve = solver.solve
        else:
            solve = astar.solve if algorithm == "astar" else theta_star.solve
        begin = perf_counter_ns()
        try:
            path = solve(grid, start, goal)
        finally:
            row["runtime_ms"] = (perf_counter_ns() - begin) / 1_000_000

        if algorithm == "astar":
            stats = astar.get_last_stats()
            for key in ("expanded_nodes", "generated_nodes", "evaluated_nodes"):
                row[key] = getattr(stats, key)
        if grid != original:
            raise ValueError("Solver mutated its input grid")
        result = validate_path(original, start, goal, path, algorithm=algorithm)
        if result.status == "none":
            row["outcome"] = "none"
        elif result.is_valid:
            length = result.path_length
            row.update(outcome="found", path_valid=True, path_length=length)
            if algorithm == "astar":
                row["matches_grid_reference"] = math.isclose(length, reference, rel_tol=1e-7, abs_tol=1e-6)
            if algorithm in ("astar", "aco") and reference > 0:
                row["grid_optimality_gap_pct"] = 100 * (length - reference) / reference
        else:
            row.update(outcome="invalid_path", path_valid=False, error=", ".join(result.errors))
    except Exception as exc:
        row.update(outcome="error", error=f"{type(exc).__name__}: {exc}")
    if grid != original and "mutated" not in row["error"]:
        row["error"] += " (solver mutated its input grid)"
    return row, path if row["outcome"] == "found" else None


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values):
    if not values:
        return dict(mean=None, median=None, sd=None, p90=None)
    values = sorted(values)
    rank = 0.9 * (len(values) - 1)
    p90 = values[math.floor(rank)] + (rank % 1) * (values[math.ceil(rank)] - values[math.floor(rank)])
    return dict(mean=fmean(values), median=median(values),
                sd=stdev(values) if len(values) > 1 else None, p90=p90)


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def load_scenarios(profile):
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if profile == "full":
        return cases
    if profile == "smoke":
        return [next(c for c in cases if c["family"] == family) for family in MAPS]
    raise ValueError(f"Unknown profile {profile!r}")


def scenario_id(row):
    return row["map_name"], row["scenario_file"], row["scenario_index"]


def add_relative_lengths(attempts):
    """All attempts supplied here belong to one scenario."""
    baselines = [r["path_length"] for r in attempts if r["algorithm"] == "astar"
                 and r["outcome"] == "found"]
    baseline = median(baselines) if baselines else None
    for row in attempts:
        if baseline is not None and baseline > 0 and row["algorithm"] != "astar" and row["outcome"] == "found":
            row["relative_to_astar_pct"] = 100 * (row["path_length"] - baseline) / baseline


def aggregate_results(rows):
    """Return one statistical record per algorithm and exact MovingAI record."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"], scenario_id(row)].append(row)
    summary = []
    for (algorithm, case_id), runs in grouped.items():
        valid = [r for r in runs if r["outcome"] == "found"]
        returned = len(valid) + sum(r["outcome"] == "invalid_path" for r in runs)
        record = dict(algorithm=algorithm, case_id=case_id, attempts=len(runs), found=len(valid),
                      failed=len(runs)-len(valid), returned=returned,
                      success_rate=100 * len(valid) / len(runs))
        for label, field, sample in (
            ("runtime_all", "runtime_ms", runs), ("runtime_valid", "runtime_ms", valid),
            ("length", "path_length", valid), ("gap", "grid_optimality_gap_pct", valid),
            ("relative", "relative_to_astar_pct", valid),
        ):
            values = [r[field] for r in sample if r[field] is not None]
            for stat, value in _distribution(values).items():
                record[f"{label}_{stat}"] = value
            if label == "length":
                record.update(best=min(values, default=None), worst=max(values, default=None))
        summary.append(record)
    return summary


def matched_ids(records, algorithms):
    solved = [{r["case_id"] for r in records if r["algorithm"] == a and r["found"]}
              for a in algorithms]
    return set.intersection(*solved)


def average(records, field):
    values = [r[field] for r in records if r[field] is not None]
    return fmean(values) if values else None


def comparison(records, algorithms):
    matched = matched_ids(records, algorithms)
    result = {}
    for algorithm in algorithms:
        group = [r for r in records if r["algorithm"] == algorithm]
        paired = [r for r in group if r["case_id"] in matched]
        attempts = sum(r["attempts"] for r in group)
        found = sum(r["found"] for r in group)
        returned = sum(r["returned"] for r in group)
        values = dict(attempts=attempts, found=found, failed=attempts-found,
                      success_rate=average(group, "success_rate"),
                      validity=100 * found / returned if returned else None,
                      coverage=100 * sum(bool(r["found"]) for r in group) / len(group))
        for prefix in ("runtime_all", "runtime_valid", "length", "gap", "relative"):
            for stat in ("mean", "median", "sd", "p90"):
                key = f"{prefix}_{stat}"
                sample = group if prefix.startswith("runtime") else paired
                values[key] = average(sample, key)
        result[algorithm] = values
    return result


def fmt(value):
    if value is None:
        return "N/A"
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def svg_element(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attributes.items()})


def svg_text(parent, x, y, content, **attrs):
    svg_element(parent, "text", x=x, y=y, **attrs).text = str(content)


def svg_canvas(width, height, title):
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox=f"0 0 {width} {height}",
                      attrib={"font-family": "system-ui, sans-serif", "font-size": "13", "fill": "#182938"})
    svg_element(root, "title").text = title
    svg_element(root, "rect", width=width, height=height, fill="white")
    return root


def metric_figure(path, title, panels, overall):
    """Dot plots keep signed differences and missing measurements distinct."""
    panel_width = 320
    height = 100 + 38 * len(overall)
    root = svg_canvas(len(panels) * panel_width, height, title)
    for i, (label, field) in enumerate(panels):
        left = i * panel_width
        values = [r[field] for r in overall.values() if r[field] is not None]
        low, high = min([0, *values]), max([0, *values])
        log = field.startswith("runtime") and values and min(values) > 0 and max(values) / min(values) >= 20
        if log:
            low, high = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
        elif field == "success_rate":
            low, high = 0, 100
        if low == high:
            high = low + 1
        x1, x2 = left + 90, left + 221
        def position(v):
            return x1 + (v - low) / (high - low) * (x2 - x1)
        svg_text(root, left+12, 23, label + (" (log10)" if log else ""), **{"font-weight": "bold"})
        for j, (a, record) in enumerate(overall.items()):
            y = 51 + 38*j
            shade = "#fff3e9" if a == "aco" else "#edf5f7"
            svg_element(root, "rect", x=left+5, y=y-17, width=panel_width-10,
                        height=38, fill=shade)
            svg_text(root, left+12, y, NAMES[a], fill=COLORS[a], **{"font-weight": "bold"})
            svg_text(root, left+12, y+13, FAMILIES[a], **{"font-size": 10})
            svg_element(root, "line", x1=x1, x2=x2, y1=y, y2=y, stroke="#d5dce0")
            value = record[field]
            if value is not None:
                x = position(math.log10(value) if log else value)
                svg_element(root, "circle", cx=x, cy=y, r=5, fill=COLORS[a])
            svg_text(root, x2+10, y+4, fmt(value))
        if values:
            ticks = [low, high]
            if low < 0 < high and not log:
                ticks.insert(1, 0)
            for tick in ticks:
                svg_text(root, position(tick), height-12, f"{(10**tick if log else tick):.2g}",
                         **{"text-anchor": "middle", "font-size": 11})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def path_figure(path, case, grid, samples):
    panel, cell = 274, min(7, 228 / max(len(grid), len(grid[0])))
    root = svg_canvas(panel * len(samples), 340, case["scenario_file"])
    for i, (row, route) in enumerate(samples):
        left, top = i * panel + 15, 43
        a = row["algorithm"]
        heading = NAMES[a] + (f"  ·  seed {row['seed']}" if a == "aco" else "")
        svg_text(root, left, 24, heading, fill=COLORS[a], **{"font-weight": "bold"})
        svg_element(root, "rect", x=left, y=top, width=len(grid[0])*cell,
                    height=len(grid)*cell, fill="#f3f5f6", stroke="#a8b3bb")
        for y, line in enumerate(grid):
            for x, blocked in enumerate(line):
                if blocked:
                    svg_element(root, "rect", x=left+x*cell, y=top+y*cell, width=cell,
                                height=cell, fill="#515e69")
        def center(x, y):
            return left + (x+.5)*cell, top + (y+.5)*cell
        if route:
            points = " ".join(f"{cx},{cy}" for cx, cy in (center(x, y) for x, y in route))
            svg_element(root, "polyline", points=points, stroke=COLORS[a], fill="none",
                        **{"stroke-width": 2.3, "stroke-linejoin": "round"})
        for endpoint, color in (("start", "#14672c"), ("goal", "#952347")):
            cx, cy = center(row[endpoint+"_x"], row[endpoint+"_y"])
            svg_element(root, "circle", cx=cx, cy=cy, r=3.3, fill=color, stroke="white")
        status = "valid" if row["path_valid"] else row["outcome"]
        svg_text(root, left, 292, f"{status}  ·  length {fmt(row['path_length'])}  ·  {fmt(row['runtime_ms'])} ms")
        svg_text(root, left, 317, f"Start ({row['start_x']}, {row['start_y']})", fill="#14672c")
        svg_text(root, left+120, 317, f"Goal ({row['goal_x']}, {row['goal_y']})", fill="#952347")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate_figures(directory, overall, representatives):
    directory.mkdir(exist_ok=True)
    runtime = ("Mean solver runtime (ms)", "runtime_all_mean")
    relative = ("Length difference vs A* (%)", "relative_mean")
    success = ("Valid successes (%)", "success_rate")
    metric_figure(directory/"runtime_comparison.svg", "Solver runtime", [runtime], overall)
    quality = [("Matched mean path length (cells)", "length_mean")]
    if "astar" in overall and len(overall) > 1:
        quality.append(relative)
    metric_figure(directory/"path_quality_comparison.svg", "Matched path quality", quality, overall)
    if len(overall) > 1:
        panels = [runtime, *quality[1:], success]
        metric_figure(directory/"algorithm_families.svg", "Search families", panels, overall)
    for family, (case, grid, samples) in representatives.items():
        path_figure(directory/f"{family}_paths.svg", case, grid, samples)


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(v) for v in cells) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def generate_report(path, rows, records, metadata, representatives):
    algorithms = metadata["algorithms"]
    overall = comparison(records, algorithms)
    matched = matched_ids(records, algorithms)
    total = len(metadata["selected"])
    relative = "astar" in algorithms and len(algorithms) > 1
    gap = any(a in algorithms for a in ("astar", "aco"))
    report = ["# Pathfinding Benchmark Report", "## 1. Experimental Configuration",
              f"**{metadata['profile'].title()}** · {total} scenarios · {len(rows)} attempted executions.",
              table(["Algorithm", "Search family", "Repetitions per scenario"],
                    [(NAMES[a], FAMILIES[a], metadata["repetitions_per_instance"][a]) for a in algorithms]),
              ", ".join(f"{case['map_name']} ({len(grid[0])} × {len(grid)})"
                        for case, grid, _ in representatives.values()) + ".",
              metadata["timing"]]
    if "aco" in algorithms:
        config = metadata["config"]["aco"]
        params = ", ".join(f"{k}={v}" for k, v in config.items() if k != "seed")
        report.append(f"Each scenario uses fresh ACO solvers with seeds 0–{metadata['seeds'][-1]} and {params}.")
    report.extend(["## 2. Correctness and Success Summary",
                   table(["Algorithm", "Attempts", "Valid successes", "Failures", "Success (%)",
                          "Path validity (%)", "Scenario coverage (%)"],
                         [[NAMES[a], *(fmt(r[k]) for k in ("attempts", "found", "failed", "success_rate",
                                                           "validity", "coverage"))] for a, r in overall.items()]),
                   "Success includes every attempt in the denominator. The table averages scenario success "
                   "rates equally. Each algorithm has the same repetition count on each of its scenarios. "
                   "Path validity divides valid paths by valid plus invalid returned paths. Scenario coverage "
                   "counts scenarios with any valid solution. N/A indicates an unavailable measurement.",
                   "## 3. Runtime Comparison",
                   "Milliseconds. Each column averages the corresponding within-scenario statistic over "
                   "scenarios with a measurement. SD is the sample standard deviation. The within-scenario "
                   "p90 interpolates at rank 0.9(n−1).",
                   table(["Algorithm", "Attempts timed", "Mean of means", "Mean of medians", "Mean of SDs",
                          "Mean of p90s"],
                         [[NAMES[a], sample, *(fmt(r[f"{key}_{stat}"]) for stat in ("mean","median","sd","p90"))]
                          for a, r in overall.items() for key, sample in
                          (("runtime_all","All"), ("runtime_valid","Valid only"))]),
                   "![Mean solver runtime in milliseconds](figures/runtime_comparison.svg)",
                   "## 4. Path Quality Comparison",
                   f"Matched **{len(matched)} of {total}** selected scenarios. Excluded **{total-len(matched)}** "
                   "because at least one selected algorithm had no valid path. Lengths are in cell widths. "
                   "Each matched scenario has equal weight. Means, medians and SDs below use valid paths only."])
    fields = [("Mean of means", "length_mean"), ("Mean of medians", "length_median"),
              ("Mean of SDs", "length_sd")]
    if gap:
        fields.append(("Mean grid gap (%)", "gap_mean"))
    if relative:
        fields.append(("Mean difference vs A* (%)", "relative_mean"))
    report.append(table(["Algorithm", *(label for label, _ in fields)],
                        [[NAMES[a], *(fmt(r[k]) for _, k in fields)] for a, r in overall.items()]))
    if gap:
        report.append("Grid gap measures the signed percentage difference from a positive MovingAI grid "
                      "reference for A* and ACO. Decimal reference rounding can leave small residuals.")
    if relative:
        report.append("Signed differences use the median valid A* length on the same scenario. Negative "
                      "means shorter. A missing or zero baseline leaves the comparison unavailable.")
    report.extend(["Theta* can connect nonadjacent cell centers with clear line of sight, allowing shorter "
                   "paths than the eight-neighbor grid optimum.",
                   "![Matched path quality](figures/path_quality_comparison.svg)", "## 5. Random vs. Maze"])
    map_rows = []
    for family, (case, _, _) in representatives.items():
        subset = [r for r in records if r["case_id"][0] == case["map_name"]]
        paired = len(matched_ids(subset, algorithms))
        n = len({r["case_id"] for r in subset})
        for a, r in comparison(subset, algorithms).items():
            map_rows.append([family.title(), NAMES[a], fmt(r["runtime_all_mean"]), fmt(r["success_rate"]),
                             f"{paired}/{n}", fmt(r["length_mean"])])
    report.append(table(["Map", "Algorithm", "Mean runtime (ms)", "Success (%)", "Matched/selected",
                         "Matched mean length"], map_rows))
    report.append("## 6. ACO Stability")
    if "aco" in algorithms:
        report.append("Each row spans the scheduled seeds for one instance. Runtime SD includes failures. "
                      "Length statistics use valid paths. SD needs at least two measurements.")
        report.append(table(["Scenario file / index", "Successes/attempts", "Success (%)", "Best length",
                             "Worst length", "Length SD", "Runtime SD (ms)"],
                            [[f"{r['case_id'][1]} / {r['case_id'][2]}", f"{r['found']}/{r['attempts']}",
                              *(fmt(r[k]) for k in ("success_rate","best","worst","length_sd","runtime_all_sd"))]
                             for r in records if r["algorithm"] == "aco"]))
    else:
        report.append("This selection contains no ACO trials.")
    report.append("## 7. Heuristic vs. Metaheuristic Comparison")
    report.append("The measurements describe the selected implementations and configurations. "
                  "Each algorithm retains its own scenario-weighted result within its search family.")
    if len(algorithms) > 1:
        report.append("![Individual algorithm measurements by search family](figures/algorithm_families.svg)")
    else:
        report.append("A cross-algorithm comparison requires more than one selected algorithm.")
    report.extend(["## 8. Representative Path Visualization",
                   "First selected record per map, first measured repetition for each algorithm "
                   "(seed 0 for ACO). A failed attempt displays its outcome on the same grid."])
    for family, (case, _, _) in representatives.items():
        report.extend([f"**{case['scenario_file']} · index {case['scenario_index']}**",
                       f"![Measured routes on the {family} map](figures/{family}_paths.svg)"])
    report.extend(["## 9. Experimental Limitations",
                   "Two fixed 32×32 layouts cannot establish scalability. Theta* uses any-angle movement. "
                   "ACO depends on its seed and construction budget. Unsolved scenarios reduce the matched "
                   "quality sample. Interpretation therefore also requires the full success and coverage table. "
                   "Runtime depends on hardware and system load. Only A* exposes node counters. Theta* "
                   "expansions and LOS checks and ACO operation counts are unavailable. Configured ants and "
                   "rounds describe budgets. Smoke runs verify the workflow and provide limited performance evidence."])
    path.write_text("\n\n".join(report) + "\n", encoding="utf-8")


def run_benchmark(profile="smoke", algorithms=ALGORITHMS, output_dir=None):
    """Run serially. An explicit output_dir is the exact destination."""
    algorithms = tuple(algorithms)
    if not algorithms or len(set(algorithms)) != len(algorithms) or not set(algorithms) <= set(ALGORITHMS):
        raise ValueError("Select distinct supported algorithms")
    selected = load_scenarios(profile)
    ordered = [a for a in ALGORITHMS if a in algorithms]
    suffix = "" if len(ordered) == len(ALGORITHMS) else "-" + "-".join(ordered)
    output = Path(output_dir) if output_dir is not None else RESULTS / (profile + suffix)
    meta_path = output / "metadata.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous.get("profile") != profile or set(previous.get("algorithms", [])) != set(algorithms):
            raise ValueError(f"{output} contains another benchmark selection. Choose another output directory.")
    counts = repetition_counts(profile, ordered)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows, representatives, files = [], {}, set()
    for case in selected:
        pair = paths(case)
        files.update(pair)
        parser = MovingAIParser(*pair)
        samples = []
        representative = case["family"] not in representatives
        instance = []
        for a in ordered:
            for rep in range(1, counts[a]+1):
                row, route = run_algorithm(a, case, parser, rep)
                instance.append(row)
                if representative and rep == 1:
                    samples.append((row, route))
        add_relative_lengths(instance)
        rows.extend(instance)
        if representative:
            grid, _, _ = parser.get_input(case["scenario_index"])
            representatives[case["family"]] = case, grid, samples
    config = dict(astar=dict(heuristic="octile", weight=1.0, tie_breaking="high_g"),
                  theta_star="Basic Theta*, defaults", aco=vars(ACOSolver()))
    sources = [p for folder in ("src", "experiments") for p in (ROOT/folder).rglob("*.py")]
    metadata = dict(profile=profile, algorithms=ordered, selected=selected, repetitions_per_instance=counts,
                    seeds=list(range(counts.get("aco", 0))), config={a: config[a] for a in ordered},
                    timestamp_utc=timestamp, python=platform.python_version(), os=platform.system(),
                    os_release=platform.release(), architecture=platform.machine(), cpu=platform.processor(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    timing="Serial perf_counter_ns around the public solver call, converted to milliseconds. "
                           "Parsing, copies, ACO construction, external validation and reporting occur outside the timer.",
                    manifest_sha256=_hash(MANIFEST),
                    dataset_sha256={p.relative_to(ROOT).as_posix(): _hash(p) for p in sorted(files)},
                    source_sha256={p.relative_to(ROOT).as_posix(): _hash(p) for p in sorted(sources)})
    records = aggregate_results(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "benchmark_results.csv", rows)
    generate_figures(output / "figures", comparison(records, ordered), representatives)
    generate_report(output / "benchmark_report.md", rows, records, metadata, representatives)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} attempts for {len(selected)} scenarios to {output}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    profile = cli.add_mutually_exclusive_group()
    profile.add_argument("--full", action="store_true")
    profile.add_argument("--profile", choices=("smoke", "full"))
    selection = cli.add_mutually_exclusive_group()
    selection.add_argument("--algorithm", choices=ALGORITHMS)
    selection.add_argument("--algorithms", nargs="+", choices=ALGORITHMS)
    cli.add_argument("--output-dir", type=Path)
    args = cli.parse_args()
    algorithms = [args.algorithm] if args.algorithm else args.algorithms or ALGORITHMS
    if len(set(algorithms)) != len(algorithms):
        cli.error("Select each algorithm only once")
    try:
        run_benchmark("full" if args.full else args.profile or "smoke", algorithms, args.output_dir)
    except ValueError as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
