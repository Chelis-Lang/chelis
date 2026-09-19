The informational `PR Package Expansion` summary now classifies what it
observed instead of restating that something went wrong. Every failing test is
differenced against the newest complete nightly default-branch run and reported
as introduced or inherited, every selected target with no execution evidence is
counted as unrun, and the three counts stay disjoint so coverage the lane could
not reach is never reported as inherited. The report is clean when nothing was
introduced. A missing baseline, or one the candidate's own merge base does not
contain, fails the report loudly rather than differencing against the wrong
tree. The soft-budget finding is gone: the per-shard estimate is a
longest-processing-time balancing weight rather than a predicted duration, and
across every dispatch since the lane existed the comparison reported no defect
that the deadline and coverage counts did not already carry.
