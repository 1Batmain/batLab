# Shared guards for the optimiser benchmarks. Source, don't execute.
#
# THE STALE-BINARY TRAP: this workspace's root package is `batBuilder`, and the
# trained binary lives in the `main` MEMBER crate. `cargo build --release` at the
# root therefore does NOT rebuild `target/release/main` — it reports "Finished"
# having built only the root lib. A whole He-init arm was run against a binary
# predating the He commit before this was caught: `--weight-init he` was parsed
# by nothing and silently ignored, so the arm was a byte-for-byte copy of the
# uniform arm and looked like a perfectly clean "He changes nothing" result.
#
# Two defences, because the build alone is not proof the flags exist:
#   ensure_binary   — build with the -p that actually produces the binary
#   assert_flag     — grep the run's own banner for the flag echoed back

ensure_binary() {
  cargo build --release -p main >&2 || { echo "BUILD FAILED" >&2; exit 1; }
}

# assert_flag <logfile> <needle> — the headless banner echoes the parsed config
# ("optimizer=adam, init=he, ..."). If the needle is absent, the binary did not
# understand the flag and the run is meaningless: fail loudly rather than
# publish a silently-degenerate arm.
assert_flag() {
  local log=$1 needle=$2
  if ! grep -q "$needle" "$log"; then
    echo "FATAL: '$needle' missing from $log banner — stale binary, arm discarded" >&2
    return 1
  fi
}
