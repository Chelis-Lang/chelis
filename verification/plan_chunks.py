"""Plan the separately compiled whole-core extraction.

    python verification/plan_chunks.py --work FRESH_DIR --out PLAN.json \
        [--chain default|lowering-trace] [--charon BIN]

For every workspace crate in dependency order, Charon translates all local items
(roots = the crate) and stops after translation (CHARON_CALLGRAPH_ONLY=1),
writing its item graph to FRESH_DIR/dumps/<crate>.callgraph.json. The planner
then

1. computes the strongly connected components of the local items over
   `deps` + `reaches` (a trait declaration reaches its provided methods);
2. merges into one unit every item with the items its canonical pattern also
   matches (nested closures, provided methods, drop glue), since an --opaque
   pattern in a later chunk would also hide them; the quotient graph is made
   acyclic again by merging its SCCs;
3. packs the units, dependencies first, into chunks under a cost budget
   (statements + item weight x items); a unit larger than the budget is its
   own chunk.

Each chunk records its package, features, root patterns (every owned item that
Charon can select), the opaque patterns of the workspace items it reaches that
earlier chunks own (so it carries their signatures, not their bodies; types stay
transparent and come from the owner's manifest), and the
owned local items the coverage audit expects. Items that cannot be roots
(synthetic items, impls that Charon's passes remove) are owned but not rooted.

The FCIS imperative shell (verification/native_shell.json) is never extracted:
the dumps and every chunk make its patterns opaque, no chunk owns its items,
and the plan's `native_shell` section lists, per shell pattern, its kind, its
local items and `core_dependents` (the non-shell local items with a deps edge
into it). A shell pattern that selects no local item rejects the plan.

Chains: `default` extracts all crates with default features. `lowering-trace`
enables the feature on chelis-ir and chelis-pipeline-core; their chunks get the
module prefix `Trace` so both chains' modules can share one Lean source root
(the feature-independent chunks are identical in both plans).

The plan is a deterministic function of the dumps (no timestamps); dumps and
their measurements are kept under FRESH_DIR. `--from-dumps DIR` replans from an
earlier FRESH_DIR without running Charon (its stamps must match).
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_whole import (  # noqa: E402
    NATIVE_SHELL, PLAN_FORMAT, ROOT, WORKSPACE_CRATES, ExtractionFailure, Runner, charon_command,
    clean_package, load_native_shell, parse_charon_log, prepare_target, sha256_file,
    source_tree_sha256, tool_identity,
)
from extract import CHARON_OPTIONS, CORE_FAMILIES, ITERATOR_ADAPTERS  # noqa: E402

PACKAGES = {crate: crate.replace("_", "-") for crate in WORKSPACE_CRATES}
CHAINS = {
    "default": {"features": {}, "prefix": {}},
    "lowering-trace": {
        "features": {"chelis_ir": ["lowering-trace"], "chelis_pipeline_core": ["lowering-trace"]},
        "prefix": {"chelis_ir": "Trace", "chelis_pipeline_core": "Trace"},
    },
}


def tarjan(n: int, succ) -> list[list[int]]:
    """SCCs, dependencies first (sinks of `succ` first)."""
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    comps: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            edges = succ[v]
            for j in range(i, len(edges)):
                w = edges[j]
                if index[w] == -1:
                    work.append((v, j + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(sorted(comp))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return comps


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def units_of(items: list[dict], local: list[int]) -> list[list[int]]:
    """Acyclic units of local item indices, dependencies first."""
    position = {item: k for k, item in enumerate(local)}
    succ = [[position[d] for d in sorted(set(items[i]["deps"]) | set(items[i]["reaches"]))
             if d in position and d != i] for i in local]
    groups = UnionFind(len(local))
    for comp in tarjan(len(local), succ):
        for member in comp[1:]:
            groups.union(comp[0], member)
    for k, i in enumerate(local):
        for j in items[i]["pattern_also_matches"]:
            if j in position:
                groups.union(k, position[j])
    # The merges can close cycles through other units: collapse them again.
    reps = sorted({groups.find(k) for k in range(len(local))})
    rep_index = {rep: n for n, rep in enumerate(reps)}
    quotient = [set() for _ in reps]
    for k in range(len(local)):
        source = rep_index[groups.find(k)]
        for w in succ[k]:
            target = rep_index[groups.find(w)]
            if target != source:
                quotient[source].add(target)
    members = collections.defaultdict(list)
    for k in range(len(local)):
        members[rep_index[groups.find(k)]].append(local[k])
    units = []
    for comp in tarjan(len(reps), [sorted(edges) for edges in quotient]):
        units.append(sorted(i for q in comp for i in members[q]))
    return units


def rootable(item: dict) -> bool:
    return bool(item["pattern"]) and item["source"] != "synthetic" and not item.get("removed_by_passes")


def camel(crate: str) -> str:
    return "".join(part.capitalize() for part in crate.split("_"))


def shell_index(items: list[dict], patterns) -> dict[int, str]:
    """Items of a dump that belong to the native shell, mapped to the shell
    pattern covering them: the item a pattern names and every item nested
    under it (closures, statics, impls declared in its body)."""
    covered = {i: item["pattern"] for i, item in enumerate(items) if item["pattern"] in patterns}
    for i, pattern in list(covered.items()):
        for j in items[i]["pattern_also_matches"]:
            covered.setdefault(j, pattern)
    return covered


def plan_crate(dump: dict, crate: str, first_chunk: int, budget: int, item_weight: int,
               owner_of: dict, shell: dict[int, str]) -> tuple[list[dict], list[str]]:
    """Chunks of one crate, and the reachable items of earlier crates that no
    chunk owns (they are then translated, hence owned, here).

    `owner_of` maps (crate, kind, name) of every owned item to the latest chunk
    owning an item of that name (names are not unique: closures print alike,
    and a dependency's pattern may be spelled differently in another dump);
    it is extended here. Native shell items (`shell`) are owned by no chunk:
    every chunk makes them opaque."""
    items = dump["items"]
    local = [i for i, item in enumerate(items) if item["crate"] == crate and i not in shell]
    units = units_of(items, local)

    def cost(unit):
        return sum(items[i]["statements"] for i in unit) + item_weight * len(unit)

    packed: list[list[list[int]]] = []
    current: list[list[int]] = []
    current_cost = 0
    for unit in units:
        unit_cost = cost(unit)
        if current and current_cost + unit_cost > budget:
            packed.append(current)
            current, current_cost = [], 0
        current.append(unit)
        current_cost += unit_cost
    if current:
        packed.append(current)

    chunks = []
    local_owner: dict[int, int] = {}
    for offset, group in enumerate(packed):
        number = first_chunk + offset
        owned = sorted(i for unit in group for i in unit)
        for i in owned:
            local_owner[i] = number
            key = (crate, items[i]["kind"], items[i]["name"])
            owner_of[key] = max(owner_of.get(key, number), number)
        chunks.append({"index": number, "owned_indices": owned, "units": len(group),
                       "largest_unit_cost": max(cost(unit) for unit in group),
                       "cost": sum(cost(unit) for unit in group)})

    # Opaque: the workspace items reachable from the chunk that earlier chunks own.
    unowned: set[str] = set()
    for chunk in chunks:
        owned = set(chunk["owned_indices"])
        seen = set(owned)
        frontier = list(owned)
        opaque = set()
        while frontier:
            v = frontier.pop()
            for w in list(items[v]["deps"]) + list(items[v]["reaches"]):
                if w in seen:
                    continue
                seen.add(w)
                if w in shell:
                    continue
                frontier.append(w)
                target = items[w]
                # Types stay transparent: later chunks need their definitions (e.g. to
                # read a discriminant); Aeneas takes them from the owner's manifest.
                if target["crate"] not in WORKSPACE_CRATES or not target["pattern"] or target["kind"] == "type":
                    continue
                if target["crate"] == crate:
                    if local_owner[w] > chunk["index"]:
                        raise ExtractionFailure(f"{crate}: chunk order violated by {target['name']}")
                    opaque.add(target["pattern"])
                elif (target["crate"], target["kind"], target["name"]) in owner_of:
                    opaque.add(target["pattern"])
                else:
                    # Generated for this use only (a provided method or drop glue of an
                    # earlier crate's type that no earlier dump contains): this chunk
                    # translates it, hence owns it, and later chunks keep it opaque.
                    unowned.add(f"{target['kind']} {target['name']}")
                    owner_of[(target["crate"], target["kind"], target["name"])] = chunk["index"]
                    chunk.setdefault("adopted", []).append(w)
        chunk["opaque"] = sorted(opaque)
    return chunks, sorted(unowned)


def dump_crate(crate: str, work: Path, args, runner: Runner, target: Path,
               chain: dict) -> tuple[dict, dict, list[dict]]:
    """The callgraph of every local item of `crate`. With --allow-charon-errors
    (diagnostic plans only) Charon continues past translation errors, which
    are returned; the items it could not translate are missing from the graph."""
    dumps = work / "dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    callgraph = dumps / f"{crate}.callgraph.json"
    package = PACKAGES[crate]
    shell = dumps / "native-shell.opaque"
    if not shell.is_file():
        shell.write_text("".join(f"{item['pattern']}\n" for item in load_native_shell()))
    clean_package(runner, args.charon, package, target, dumps / f"{crate}.clean.log")
    command = charon_command(args.charon, package=package, features=chain["features"].get(crate, []),
                             llbc=dumps / f"{crate}.llbc", target=target,
                             strict=not args.allow_charon_errors, roots=[crate], shell_file=shell)
    log = dumps / f"{crate}.charon.log"
    step = runner.run(command, log=log,
                      env={"CHARON_CALLGRAPH": str(callgraph), "CHARON_CALLGRAPH_ONLY": "1"})
    errors = parse_charon_log(log) if step["status"] != "ok" else []
    if step["status"] != "ok" and not args.allow_charon_errors:
        raise ExtractionFailure(f"Charon callgraph dump failed for {crate}; log: {log}")
    if not callgraph.is_file() or callgraph.stat().st_mtime < step["started"] or callgraph.stat().st_size == 0:
        raise ExtractionFailure(f"no fresh callgraph for {crate}; log: {log}")
    return json.loads(callgraph.read_text()), step, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="fresh directory for the dumps")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chain", choices=sorted(CHAINS), default="default")
    parser.add_argument("--charon", default="charon")
    parser.add_argument("--from-dumps", type=Path,
                        help="replan from an earlier --work directory (same Charon and sources)")
    parser.add_argument("--budget", type=int, default=40_000,
                        help="chunk cost budget: statements + item weight x items")
    parser.add_argument("--item-weight", type=int, default=10)
    parser.add_argument("--target-dir", default=str(ROOT / "target" / "agents" / "wholecore"))
    parser.add_argument("--min-free-percent", type=int, default=20)
    parser.add_argument("--rss-cap-mb", type=int, default=7000)
    parser.add_argument("--allow-charon-errors", action="store_true",
                        help="diagnostic plan: dump past Charon translation errors (the extractor "
                             "refuses such a plan unless --diagnostic)")
    parser.add_argument("--heavy", default=os.environ.get("HEAVY_WRAPPER"),
                        help="slot limiter prefixed to every heavy step (default: $HEAVY_WRAPPER)")
    parser.add_argument("--reuse-dumps", type=Path,
                        help="copy the dumps of crates whose features are the same in this chain "
                             "from an earlier --work directory of another chain (same Charon, "
                             "sources and shell); dump the others")
    args = parser.parse_args()
    work = args.work.resolve()
    if work.exists():
        raise ExtractionFailure(f"--work already exists: {work}")
    chain = CHAINS[args.chain]
    charon = tool_identity(args.charon, ["version"])
    sources = source_tree_sha256()
    shell = load_native_shell()
    work.mkdir(parents=True)
    stamp = {"charon_identity": charon["identity"], "sources_sha256": sources, "chain": args.chain,
             "charon_options": list(CHARON_OPTIONS), "features": chain["features"],
             "allow_charon_errors": args.allow_charon_errors,
             "native_shell_opaque": sorted(item["pattern"] for item in shell)}
    measurements = {}
    dumps = {}
    dump_errors: dict[str, list] = {}
    if args.from_dumps:
        earlier = json.loads((args.from_dumps / "dumps" / "stamp.json").read_text())
        reused = earlier.pop("reused_dumps", None)
        # Items opaque in the dumps have no body edges, so they must stay shell items;
        # items added to the shell since were dumped with their bodies (more edges).
        dumped_opaque = set(earlier.pop("native_shell_opaque", []))
        current = dict(stamp)
        current.pop("native_shell_opaque")
        if earlier != current:
            raise ExtractionFailure("--from-dumps was made by another Charon, source tree or chain")
        stamp["native_shell_opaque"] = sorted(dumped_opaque)
        if reused:
            stamp["reused_dumps"] = reused
    else:
        runner = Runner(min_free=args.min_free_percent, max_wait=3600, rss_cap=args.rss_cap_mb << 20, nice=5,
                        wrapper=[args.heavy] if args.heavy else None)
        target = Path(args.target_dir).resolve()
        prepare_target(target, charon)
    reusable: dict[str, Path] = {}
    if args.reuse_dumps and not args.from_dumps:
        earlier = json.loads((args.reuse_dumps / "dumps" / "stamp.json").read_text())
        same = ("charon_identity", "charon_options", "allow_charon_errors", "native_shell_opaque")
        if any(earlier.get(key) != stamp[key] for key in same):
            raise ExtractionFailure("--reuse-dumps was made by another Charon or shell")
        if earlier["sources_sha256"] != sources:
            # Diagnostic plans only: the reused crates' dumps may predate the edit.
            if not args.allow_charon_errors:
                raise ExtractionFailure("--reuse-dumps was made from another source tree")
            stamp["reused_dumps_sources_sha256"] = earlier["sources_sha256"]
        stamp["reused_dumps"] = {}
        for crate in WORKSPACE_CRATES:
            if earlier["features"].get(crate, []) == chain["features"].get(crate, []):
                reusable[crate] = args.reuse_dumps / "dumps" / f"{crate}.callgraph.json"
                stamp["reused_dumps"][crate] = hashlib.sha256(reusable[crate].read_bytes()).hexdigest()
        earlier_errors = json.loads((args.reuse_dumps / "dumps" / "errors.json").read_text())
    for crate in WORKSPACE_CRATES:
        if args.from_dumps or crate in reusable:
            path = reusable.get(crate) or args.from_dumps / "dumps" / f"{crate}.callgraph.json"
            (work / "dumps").mkdir(exist_ok=True)
            (work / "dumps" / path.name).write_bytes(path.read_bytes())
            dumps[crate] = json.loads(path.read_text())
            if crate in reusable:
                measurements[crate] = {"status": "reused", "from": str(path)}
                if crate in earlier_errors:
                    dump_errors[crate] = earlier_errors[crate]
        else:
            dumps[crate], step, errors = dump_crate(crate, work, args, runner, target, chain)
            measurements[crate] = {key: step[key] for key in ("status", "wall_seconds", "peak_rss_bytes",
                                                              "peak_tree_rss_bytes")}
            if errors:
                dump_errors[crate] = errors
    if args.from_dumps:
        for name in ("errors.json", "measurements.json"):
            if (args.from_dumps / "dumps" / name).is_file():
                (work / "dumps" / name).write_bytes((args.from_dumps / "dumps" / name).read_bytes())
        if (work / "dumps" / "errors.json").is_file():
            dump_errors = json.loads((work / "dumps" / "errors.json").read_text())
    else:
        (work / "dumps" / "measurements.json").write_text(json.dumps(measurements, indent=2))
        (work / "dumps" / "errors.json").write_text(json.dumps(dump_errors, indent=2))
    (work / "dumps" / "stamp.json").write_text(json.dumps(stamp, indent=2))
    if args.from_dumps:
        effective = {item["pattern"] for crate in WORKSPACE_CRATES for item in dumps[crate]["items"]
                     if item["pattern"] in dumped_opaque}
        dropped = sorted(effective - {item["pattern"] for item in shell})
        if dropped:
            raise ExtractionFailure(f"--from-dumps made items opaque that are no longer shell: {dropped}")

    shell_kinds = {item["pattern"]: item["kind"] for item in shell}
    shells = {crate: shell_index(dumps[crate]["items"], shell_kinds) for crate in WORKSPACE_CRATES}
    shell_report = {pattern: {"kind": kind, "crate": None, "members": [], "core_dependents": set()}
                    for pattern, kind in shell_kinds.items()}
    for crate in WORKSPACE_CRATES:
        items = dumps[crate]["items"]
        for i, pattern in shells[crate].items():
            if items[i]["crate"] == crate:
                entry = shell_report[pattern]
                entry["crate"] = crate
                entry["members"].append({"kind": items[i]["kind"], "name": items[i]["name"]})
        # Extracted (non-shell) local items with a deps edge into the shell.
        for i, item in enumerate(items):
            if item["crate"] != crate or i in shells[crate]:
                continue
            for d in item["deps"]:
                if d in shells[crate]:
                    shell_report[shells[crate][d]]["core_dependents"].add(item["name"])
    unmatched = sorted(pattern for pattern, entry in shell_report.items() if not entry["members"])
    if unmatched:
        raise ExtractionFailure(f"native_shell.json patterns select no local item: {unmatched}")
    for entry in shell_report.values():
        entry["members"].sort(key=lambda member: (member["kind"], member["name"]))
        entry["core_dependents"] = sorted(entry["core_dependents"])

    owner_of: dict = {}
    planned = []
    crate_stats = {}
    for crate in WORKSPACE_CRATES:
        dump = dumps[crate]
        if dump["crate"] != crate:
            raise ExtractionFailure(f"dump for {crate} names crate {dump['crate']}")
        chunks, unowned = plan_crate(dump, crate, len(planned), args.budget, args.item_weight, owner_of,
                                     shells[crate])
        items = dump["items"]
        local = [item for item in items if item["crate"] == crate]
        crate_stats[crate] = {
            "local_items": len(local),
            "statements": sum(item["statements"] for item in local),
            "by_kind": dict(sorted(collections.Counter(item["kind"] for item in local).items())),
            "not_rootable": sum(1 for item in local if not rootable(item)),
            "folded": sum(len(item.get("folded") or []) for item in local),
            "native_shell": sum(1 for i in shells[crate] if items[i]["crate"] == crate),
            "chunks": len(chunks),
            "unowned_dependency_items": unowned,
        }
        prefix = chain["prefix"].get(crate, "")
        for k, chunk in enumerate(chunks):
            owned = [items[i] for i in chunk["owned_indices"]]
            planned.append({
                "name": f"{prefix.lower() + '_' if prefix else ''}{crate}_{k:02d}",
                "module": f"{prefix}{camel(crate)}{k:02d}",
                "crate": crate,
                "package": PACKAGES[crate],
                "features": chain["features"].get(crate, []),
                "roots": [item["pattern"] for item in owned if rootable(item)],
                "opaque": chunk["opaque"],
                "externals": [earlier["name"] for earlier in planned],
                "cost": chunk["cost"],
                "units": chunk["units"],
                "largest_unit_cost": chunk["largest_unit_cost"],
                "statements": sum(item["statements"] for item in owned),
                "folded": sum(len(item.get("folded") or []) for item in owned),
                "owned": [{"kind": item["kind"], "name": item["name"], "pattern": item["pattern"],
                           "source": item["source"], "generic": item["generic"],
                           "removed_by_passes": bool(item.get("removed_by_passes"))}
                          for item in owned],
                # Items of earlier crates generated for this chunk's uses only.
                "adopted": [{"crate": items[i]["crate"], "kind": items[i]["kind"], "name": items[i]["name"],
                             "pattern": items[i]["pattern"], "source": items[i]["source"],
                             "generic": items[i]["generic"],
                             "removed_by_passes": bool(items[i].get("removed_by_passes"))}
                            for i in sorted(chunk.get("adopted", []))],
            })

    # Items of the workspace reachable from chelis_pipeline_core's local items.
    pipeline = dumps["chelis_pipeline_core"]["items"]
    seen = {i for i, item in enumerate(pipeline) if item["crate"] == "chelis_pipeline_core"}
    frontier = list(seen)
    while frontier:
        v = frontier.pop()
        for w in list(pipeline[v]["deps"]) + list(pipeline[v]["reaches"]):
            if w not in seen:
                seen.add(w)
                frontier.append(w)
    reachable = collections.Counter(pipeline[i]["crate"] for i in seen if pipeline[i]["crate"] in WORKSPACE_CRATES)

    plan = {
        "format": PLAN_FORMAT, "version": 1, "chain": args.chain,
        "inputs": {"charon_identity": charon["identity"], "sources_sha256": sources,
                   "dump_sha256": {crate: hashlib.sha256(
                       (work / "dumps" / f"{crate}.callgraph.json").read_bytes()).hexdigest()
                       for crate in WORKSPACE_CRATES},
                   "native_shell_sha256": sha256_file(NATIVE_SHELL),
                   "budget": args.budget, "item_weight": args.item_weight,
                   "charon_options": list(CHARON_OPTIONS),
                   "includes": list(ITERATOR_ADAPTERS) + list(CORE_FAMILIES)},
        "diagnostic_only": bool(args.allow_charon_errors),
        "dump_errors": dump_errors,
        "crates": crate_stats,
        "pipeline_reachable": {crate: reachable.get(crate, 0) for crate in WORKSPACE_CRATES},
        "native_shell": shell_report,
        "chunks": planned,
    }
    args.out.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(f"plan ({args.chain}): {len(planned)} chunks -> {args.out}")
    for chunk in planned:
        print(f"  {chunk['name']:<32} roots {len(chunk['roots']):>6}  opaque {len(chunk['opaque']):>6}  "
              f"cost {chunk['cost']:>7}  statements {chunk['statements']:>7}  "
              f"largest unit {chunk['largest_unit_cost']:>6}")


if __name__ == "__main__":
    try:
        main()
    except (ExtractionFailure, OSError) as error:
        print(f"PLAN REJECTED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
