"""FIFO gate admission (#1622): real processes/locks, explicitly scheduled polls.

Run: .venv/bin/python -m unittest scripts.test_gate_queue
The child handshake controls poll timing; no Cargo or workstation lease is used.
"""

from __future__ import annotations

import importlib.util
import argparse
import io
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("gate.py")
spec = importlib.util.spec_from_file_location("gate_queue_tests", SCRIPT)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

WORKER = r'''
import importlib.util, io, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gate_worker", sys.argv[1])
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)
def sleep(seconds):
    emit("poll", position=getattr(lease, "queue_position", None))
    if sys.stdin.readline().strip() != "continue":
        raise KeyboardInterrupt
lease = gate.GateLease(Path(sys.argv[2]), mode="local", worktree=Path(sys.argv[3]),
    head="queue-test", wait=True, timeout=None, output=io.StringIO(), sleep=sleep)
try:
    lease.acquire()
    emit("acquired")
    sys.stdin.readline()
except KeyboardInterrupt:
    emit("cancelled")
finally:
    lease.release()
'''


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="chelis-gate-queue-test-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "gate.lock"

    def lease(self, **kwargs):
        settings = dict(mode="local", worktree=Path("/test"), head="test",
                        wait=False, timeout=None, output=io.StringIO())
        settings.update(kwargs)
        lease = gate.GateLease(self.path, **settings)
        self.addCleanup(lease.release)
        return lease

    def worker(self, name):
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", WORKER, str(SCRIPT), str(self.path), name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.addCleanup(self.stop, process)
        return process

    def stop(self, process):
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)

    def event(self, process, expected, position=None):
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(10), f"no {expected} event from {process.pid}")
        line = process.stdout.readline()
        self.assertTrue(line, f"worker exited {process.poll()}")
        result = json.loads(line)
        self.assertEqual(result["event"], expected, result)
        if position is not None:
            self.assertEqual(result["position"], position)
        return result

    def send(self, process, command="continue"):
        process.stdin.write((command + "\n").encode())
        process.stdin.flush()

    def finish(self, process):
        self.send(process, "release")
        self.assertEqual(process.wait(timeout=5), 0)

    def tickets(self):
        return list(self.path.with_name("gate.lock.queue").glob("*.ticket"))

    def test_older_waiter_wins_despite_younger_poll_and_prompt_retaker(self):
        holder = self.lease()
        holder.acquire()
        older = self.worker("/older")
        self.event(older, "poll")
        younger = self.worker("/younger")
        self.event(younger, "poll")
        holder.release()
        # A free main lock does not authorize overtaking queued callers.
        for _ in range(2):
            with self.assertRaises(gate.LeaseHeld):
                self.lease().acquire()
        self.send(younger)
        self.event(younger, "poll", 2)
        self.send(older)
        self.event(older, "acquired")
        self.send(younger)
        self.event(younger, "poll", 1)
        self.finish(older)
        self.send(younger)
        self.event(younger, "acquired")
        self.finish(younger)
        self.assertEqual(self.tickets(), [])

    def test_cancelled_front_waiter_does_not_block_successor(self):
        self.check_abandoned_front(kill=False)

    def test_killed_front_waiter_does_not_block_successor(self):
        self.check_abandoned_front(kill=True)

    def check_abandoned_front(self, *, kill):
        holder = self.lease()
        holder.acquire()
        older = self.worker("/older")
        self.event(older, "poll", 1)
        younger = self.worker("/younger")
        self.event(younger, "poll", 2)
        if kill:
            older.kill()
            older.wait(timeout=5)
        else:
            self.send(older, "cancel")
            self.event(older, "cancelled")
            self.assertEqual(older.wait(timeout=5), 0)
        self.send(younger)
        self.event(younger, "poll", 1)
        holder.release()
        self.send(younger)
        self.event(younger, "acquired")
        self.finish(younger)
        self.assertEqual(self.tickets(), [])

    def test_killed_holder_releases_lease_to_waiter(self):
        holder = self.worker("/holder")
        self.event(holder, "acquired")
        waiter = self.worker("/waiter")
        self.event(waiter, "poll", 1)
        holder.kill()
        holder.wait(timeout=5)
        self.send(waiter)
        self.event(waiter, "acquired")
        self.finish(waiter)
        self.assertEqual(self.tickets(), [])

    def test_timeout_removes_ticket_and_honors_deadline(self):
        holder = self.lease()
        holder.acquire()
        now = [0.0]
        def sleep(seconds):
            now[0] += seconds
            holder.release()
        waiter = self.lease(wait=True, timeout=3, clock=lambda: now[0], sleep=sleep)
        with self.assertRaises(gate.LeaseHeld):
            waiter.acquire()
        self.assertEqual(now[0], 3)
        self.assertEqual(self.tickets(), [])
        self.lease().acquire()

    def test_interrupt_during_acquire_closes_ticket_and_descriptors(self):
        holder = self.lease()
        holder.acquire()
        waiter = self.lease(wait=True, sleep=mock.Mock(side_effect=KeyboardInterrupt))
        with self.assertRaises(KeyboardInterrupt):
            waiter.acquire()
        self.assertIsNone(waiter._fd)
        self.assertEqual(self.tickets(), [])
        holder.release()
        self.lease().acquire()

    def test_heartbeat_names_queue_position(self):
        holder = self.lease()
        holder.acquire()
        now = [0.0]
        def sleep(seconds):
            now[0] += seconds
            if now[0] >= 30:
                holder.release()
        out = io.StringIO()
        waiter = self.lease(wait=True, output=out, sleep=sleep,
                            clock=lambda: now[0], heartbeat_seconds=10)
        waiter.acquire()
        self.assertGreaterEqual(out.getvalue().count("queue position 1"), 2)

    def test_busy_queue_mutex_honors_no_wait_and_timeout(self):
        guard = self.path.with_name("gate.lock.queue.lock")
        fd = os.open(guard, os.O_RDWR | os.O_CREAT, 0o644)
        self.addCleanup(os.close, fd)
        gate.fcntl.flock(fd, gate.fcntl.LOCK_EX)
        with self.assertRaises(gate.LeaseHeld):
            self.lease().acquire()
        now = [0.0]
        def sleep(seconds):
            now[0] += seconds
        waiter = self.lease(wait=True, timeout=0.25, clock=lambda: now[0], sleep=sleep)
        with self.assertRaises(gate.LeaseHeld):
            waiter.acquire()
        self.assertEqual(now[0], 0.25)
        self.assertEqual(self.tickets(), [])
        gate.fcntl.flock(fd, gate.fcntl.LOCK_UN)
        self.lease().acquire()

    def test_queue_read_error_stops_gate_instead_of_bypassing_waiters(self):
        original = Path.iterdir
        def denied(path):
            if path.name == "gate.lock.queue":
                raise PermissionError("queue unreadable")
            return original(path)
        with mock.patch.object(Path, "iterdir", denied):
            report = gate.GateReport(mode="local", started_at="test")
            errors = io.StringIO()
            code, lease = gate.take_lease(
                mode="local", args=argparse.Namespace(), report=report,
                environ={gate.LEASE_DIR_ENV: self.tmp.name}, error_stream=errors,
            )
        self.assertEqual(code, gate.EXIT_ENVIRONMENT)
        self.assertIsNone(lease)
        self.assertEqual(report.lease["mode"], "error")
        self.assertIn("queue unreadable", errors.getvalue())
        self.assertIsNone(gate.GateLease.peek(self.path))
        self.lease().acquire()

    def test_ticket_creation_failure_does_not_leave_descriptors_or_block_guard(self):
        original = os.open
        def denied(path, *args, **kwargs):
            if str(path).endswith(".ticket"):
                raise PermissionError("ticket denied")
            return original(path, *args, **kwargs)
        waiter = self.lease()
        with mock.patch.object(os, "open", denied), self.assertRaises(gate.LeaseQueueError):
            waiter.acquire()
        self.assertIsNone(waiter._fd)
        self.assertIsNone(waiter._queue_guard_fd)
        self.assertIsNone(waiter._ticket_fd)
        self.lease().acquire()

    def test_simultaneous_registration_assigns_distinct_ordered_tickets(self):
        holder = self.lease()
        holder.acquire()
        processes = [self.worker(f"/waiter-{i}") for i in range(4)]
        order = {}
        for process in processes:
            event = self.event(process, "poll")
            for _ in range(10):
                if event["position"] is not None:
                    break
                self.send(process)
                event = self.event(process, "poll")
            self.assertIsNotNone(event["position"])
            self.assertNotIn(event["position"], order)
            order[event["position"]] = process
        self.assertEqual(sorted(order), [1, 2, 3, 4])
        self.assertEqual(len(self.tickets()), 4)
        holder.release()
        for position in sorted(order):
            process = order[position]
            self.send(process)
            self.event(process, "acquired")
            self.finish(process)
        self.assertEqual(self.tickets(), [])


if __name__ == "__main__":
    unittest.main()
