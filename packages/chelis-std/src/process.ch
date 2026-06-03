module Std.Process
export (run, run_chelis)
def run(cmd: string, args: List[string]) -> (int64, string, string) = process_run(cmd, args)
def run_chelis(args: List[string]) -> (int64, string, string) = process_run("chelis", args)
