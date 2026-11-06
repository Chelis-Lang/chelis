"""Unit tests for `ci_apt_get.py`.

Run via: `python3 -m unittest scripts.test_ci_apt_get` from repo root,
or `python3 scripts/test_ci_apt_get.py`.

The tests verify transfer integrity, source replacement, retry failures, and
process containment without the real package manager. A task-local APT fixture
supplies subprocess behavior for the signal tests. Other tests replace subprocess
and network operations.
"""

import fcntl
import hashlib
import importlib.util
import io
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit
from urllib.request import HTTPSHandler, build_opener
from urllib.response import addinfourl


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ci_apt_get", here / "ci_apt_get.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cag = _load_module()


def _completed(returncode=0):
    return mock.Mock(returncode=returncode, stdout="", stderr="")


class CommandShapeTests(unittest.TestCase):
    def test_empty_package_list_is_a_noop(self):
        with mock.patch.object(cag.subprocess, "run") as run:
            self.assertEqual(cag.apt_get([], sleep=lambda _s: None), 0)
        run.assert_not_called()


class HostedUbuntuMirrorTests(unittest.TestCase):
    def test_hosted_mirrors_replace_azure_with_official_https_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            apt_root = Path(tmp)
            mirrors = apt_root / "apt-mirrors.txt"
            mirrors.write_text(
                "http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n"
                "https://archive.ubuntu.com/ubuntu/\tpriority:2\n"
                "https://security.ubuntu.com/ubuntu/\tpriority:3\n"
            )
            sources = apt_root / "sources.list.d"
            sources.mkdir()
            ubuntu = sources / "ubuntu.sources"
            ubuntu.write_text("URIs: mirror+file:/etc/apt/apt-mirrors.txt\n")
            third_party = sources / "third-party.list"
            third_party.write_text("deb https://example.org/packages stable main\n")

            cag.configure_hosted_ubuntu_mirrors(apt_root)

            self.assertEqual(
                mirrors.read_text(),
                "https://archive.ubuntu.com/ubuntu/\tpriority:1\n"
                "https://security.ubuntu.com/ubuntu/\tpriority:2\n",
            )
            self.assertEqual(ubuntu.read_text(), "URIs: mirror+file:/etc/apt/apt-mirrors.txt\n")
            self.assertEqual(
                third_party.read_text(), "deb https://example.org/packages stable main\n"
            )
            cag.configure_hosted_ubuntu_mirrors(apt_root)
            self.assertNotIn("azure", mirrors.read_text())

    def test_missing_hosted_mirror_list_fails_without_creating_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            apt_root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                cag.configure_hosted_ubuntu_mirrors(apt_root)
            self.assertEqual(list(apt_root.iterdir()), [])


class BullseyeSnapshotTests(unittest.TestCase):
    def test_configure_snapshot_replaces_every_moving_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            apt_root = Path(tmp)
            fragments = apt_root / "sources.list.d"
            fragments.mkdir()
            (apt_root / "sources.list").write_text(
                "deb http://deb.debian.org/debian bullseye main\n",
                encoding="utf-8",
            )
            (fragments / "debian.list").write_text(
                "deb http://security.debian.org bullseye-security main\n",
                encoding="utf-8",
            )
            (fragments / "debian.sources").write_text(
                "URIs: http://deb.debian.org/debian\n",
                encoding="utf-8",
            )
            (fragments / "README").write_text("keep me\n", encoding="utf-8")

            cag.configure_bullseye_snapshot(apt_root)

            entries = [
                line.split()
                for line in (apt_root / "sources.list")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                {entry[3] for entry in entries},
                {"bullseye", "bullseye-updates", "bullseye-security"},
            )
            self.assertEqual(len(entries), 3)
            for source_type, options, address, suite, component in entries:
                with self.subTest(suite=suite):
                    url = urlsplit(address)
                    self.assertEqual((source_type, component), ("deb", "main"))
                    self.assertEqual(options, "[check-valid-until=no]")
                    self.assertEqual(
                        (url.scheme, url.netloc),
                        ("https", "snapshot-cloudflare.debian.org"),
                    )
                    archive = (
                        "debian-security"
                        if suite == "bullseye-security"
                        else "debian"
                    )
                    self.assertEqual(
                        url.path, f"/archive/{archive}/20260901T000000Z/"
                    )
                    self.assertFalse(url.query)
                    self.assertFalse(url.fragment)
            self.assertFalse((fragments / "debian.list").exists())
            self.assertFalse((fragments / "debian.sources").exists())
            self.assertEqual(
                (fragments / "README").read_text(encoding="utf-8"), "keep me\n"
            )

    def test_configure_snapshot_requires_an_apt_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "missing-apt"
            with self.assertRaises(FileNotFoundError):
                cag.configure_bullseye_snapshot(absent)
            self.assertFalse(absent.exists())


class BullseyeCarrierTests(unittest.TestCase):
    payload = b"exact package bytes"
    main_origin = (
        "https://snapshot-cloudflare.debian.org/archive/debian/"
        "20260901T000000Z/"
    )
    security_origin = (
        "https://snapshot-cloudflare.debian.org/archive/debian-security/"
        "20260901T000000Z/"
    )
    cache_name = "libdemo_1%3a2.0_arm64.deb"

    def record(self, *, origin=None, path="pool/main/d/demo/demo_2.0_arm64.deb",
               name=None, size=None, digest=None):
        return (
            f"'{origin or self.main_origin}{path}' "
            f"{self.cache_name if name is None else name} "
            f"{len(self.payload) if size is None else size} "
            f"{digest or 'SHA256:' + hashlib.sha256(self.payload).hexdigest()}\n"
        )

    def exercise(self, root, plan, *, payload=None, plan_code=0, install_code=0,
                 transfer_error=None, opener=None):
        installed = []
        requested = []
        calls = []

        def apt(cmd, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                return mock.Mock(returncode=plan_code, stdout=plan)
            installed.append({
                path.name: path.read_bytes()
                for path in root.glob("*.deb")
            })
            return mock.Mock(returncode=install_code)

        def transfer(uri):
            requested.append(uri)
            if transfer_error:
                raise transfer_error
            if hasattr(payload, "read"):
                return payload
            return io.BytesIO(self.payload if payload is None else payload)

        if opener is None:
            opener = mock.Mock()
            opener.open.side_effect = transfer
        with (
            mock.patch.object(cag.subprocess, "run", side_effect=apt),
            mock.patch.object(cag, "build_opener", return_value=opener),
        ):
            rc = cag._bullseye_install(
                ["libdemo"], no_install_recommends=True, archives=root
            )
        return rc, installed, requested

    def test_exact_bytes_reach_apt_with_its_selected_cache_name(self):
        for origin, archive in (
            (self.main_origin, "debian"),
            (self.security_origin, "debian-security"),
        ):
            with self.subTest(archive=archive), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rc, installed, requested = self.exercise(
                    root, "APT status text\n" + self.record(origin=origin)
                )
                self.assertEqual(rc, 0)
                self.assertEqual(installed, [{self.cache_name: self.payload}])
                self.assertEqual(requested, [
                    f"https://archive.debian.org/{archive}/"
                    "pool/main/d/demo/demo_2.0_arm64.deb"
                ])
                self.assertEqual(list(root.glob(".ci-apt-*")), [])

    def test_corrupt_truncated_and_excess_bytes_never_reach_apt(self):
        for payload in (
            b"x" * len(self.payload), self.payload[:-1], self.payload + b"x"
        ):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rc, installed, _ = self.exercise(root, self.record(), payload=payload)
                self.assertNotEqual(rc, 0)
                self.assertEqual(installed, [])
                self.assertEqual(list(root.iterdir()), [])

    def test_unsafe_names_fail_before_any_transfer_or_cache_write(self):
        for name in (
            "../escape.deb", "/tmp/escape.deb", "sub/package.deb",
            r"sub\package.deb", "%2e%2e%2fescape.deb", "-option.deb",
            "package.txt",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rc, installed, requested = self.exercise(
                    root, self.record() + self.record(name=name)
                )
                self.assertNotEqual(rc, 0)
                self.assertEqual((installed, requested), ([], []))
                self.assertEqual(list(root.iterdir()), [])

    def test_weak_absent_and_malformed_hashes_fail_before_transfer(self):
        hashes = (
            "MD5Sum:" + "a" * 32, "SHA1:" + "a" * 40,
            "SHA256:" + "a" * 63, "SHA256:" + "g" * 64,
            "SHA512:" + "a" * 128,
        )
        plans = [self.record(digest=digest) for digest in hashes]
        plans.append(self.record().rsplit(" ", 1)[0] + "\n")
        for plan in plans:
            with self.subTest(plan=plan), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rc, installed, requested = self.exercise(root, plan)
                self.assertNotEqual(rc, 0)
                self.assertEqual((installed, requested), ([], []))
                self.assertEqual(list(root.iterdir()), [])

    def test_unrecognized_origins_and_unsafe_paths_fail_before_transfer(self):
        records = [
            self.record(origin="https://archive.debian.org/debian/"),
            self.record(origin=self.main_origin.replace("https:", "http:")),
            self.record(origin=self.main_origin.replace(
                "snapshot-cloudflare.debian.org",
                "snapshot-cloudflare.debian.org.evil.test",
            )),
            self.record(origin=self.main_origin.replace(
                "20260901T000000Z", "20260801T000000Z"
            )),
        ]
        records += [self.record(path=path) for path in (
            "pool/../escape.deb", "pool/%2e%2e/escape.deb",
            "pool/%252e%252e/escape.deb", "pool/main//demo.deb",
            "pool/main/demo.deb?redirect=1", "pool/main/demo.deb#fragment",
            "pool/main/%2fescape.deb", "pool/main/%5cescape.deb",
            "dists/bullseye/Release",
        )]
        for plan in records:
            with self.subTest(plan=plan), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rc, installed, requested = self.exercise(root, plan)
                self.assertNotEqual(rc, 0)
                self.assertEqual((installed, requested), ([], []))
                self.assertEqual(list(root.iterdir()), [])

    def test_invalid_sizes_and_duplicate_destinations_fail_before_transfer(self):
        plans = [
            self.record(size=0), self.record(size=-1), self.record(size="NaN"),
            self.record() + self.record(path="pool/main/d/demo/other.deb"),
        ]
        for plan in plans:
            with self.subTest(plan=plan), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rc, installed, requested = self.exercise(root, plan)
                self.assertNotEqual(rc, 0)
                self.assertEqual((installed, requested), ([], []))
                self.assertEqual(list(root.iterdir()), [])

    def test_plan_failure_never_transfers_or_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, installed, requested = self.exercise(
                root, self.record(), plan_code=100
            )
            self.assertEqual(rc, 100)
            self.assertEqual((installed, requested), ([], []))
            self.assertEqual(list(root.iterdir()), [])

    def test_transfer_error_leaves_no_package_and_never_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, installed, _ = self.exercise(
                root, self.record(), transfer_error=OSError("connection reset")
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(installed, [])
            self.assertEqual(list(root.iterdir()), [])

    def test_interrupted_body_leaves_no_partial_package(self):
        class InterruptedBody(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("connection reset")
                return super().read(4)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, installed, _ = self.exercise(
                root, self.record(), payload=InterruptedBody(self.payload)
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(installed, [])
            self.assertEqual(list(root.iterdir()), [])

    def test_publication_occurs_only_after_complete_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / self.cache_name
            previous = b"previous complete package"
            destination.write_bytes(previous)
            observed = []

            class ObservedBody(io.BytesIO):
                def read(self, size=-1):
                    observed.append(destination.read_bytes())
                    return super().read(4)

            rc, installed, _ = self.exercise(
                root, self.record(), payload=ObservedBody(self.payload)
            )
            self.assertEqual(rc, 0)
            self.assertTrue(observed)
            self.assertEqual(set(observed), {previous})
            self.assertEqual(destination.read_bytes(), self.payload)
            self.assertEqual(installed, [{self.cache_name: self.payload}])

    def test_failed_transfer_preserves_a_previous_complete_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / self.cache_name
            previous = b"previous complete package"
            destination.write_bytes(previous)
            rc, installed, _ = self.exercise(
                root, self.record(), payload=self.payload[:-1]
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(installed, [])
            self.assertEqual(destination.read_bytes(), previous)
            self.assertEqual(list(root.iterdir()), [destination])

    def test_symlink_destination_never_changes_its_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.write_bytes(b"do not change")
            (root / self.cache_name).symlink_to(outside)
            rc, installed, requested = self.exercise(root, self.record())
            self.assertNotEqual(rc, 0)
            self.assertEqual((installed, requested), ([], []))
            self.assertEqual(outside.read_bytes(), b"do not change")

    def test_install_failure_remains_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, installed, _ = self.exercise(
                Path(tmp), self.record(), install_code=100
            )
            self.assertEqual(rc, 100)
            self.assertEqual(installed, [{self.cache_name: self.payload}])

    def test_redirects_never_reach_another_origin_or_the_install(self):
        for location in (
            "http://archive.debian.org/debian/pool/demo.deb",
            "https://unexpected.test/demo.deb",
            "https://archive.debian.org/debian/pool/other.deb",
        ):
            requested = []

            class RedirectCarrier(HTTPSHandler):
                def https_open(self, request):
                    requested.append(request.full_url)
                    response = addinfourl(
                        io.BytesIO(b""), {"location": location},
                        request.full_url, code=302,
                    )
                    response.msg = "Found"
                    return response

            opener = build_opener(RedirectCarrier(), cag._RejectArchiveRedirects())
            with self.subTest(location=location), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rc, installed, _ = self.exercise(
                    root, self.record(), opener=opener
                )
                self.assertNotEqual(rc, 0)
                self.assertEqual(installed, [])
                self.assertEqual(requested, [
                    "https://archive.debian.org/debian/"
                    "pool/main/d/demo/demo_2.0_arm64.deb"
                ])
                self.assertEqual(list(root.iterdir()), [])

    def test_encoded_remote_path_preserves_the_apt_cache_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "libclang-dev_1%3a11.0-51+nmu5_arm64.deb"
            path = "pool/main/l/llvm-defaults/libclang-dev_11.0-51%2bnmu5_arm64.deb"
            rc, installed, requested = self.exercise(
                root, self.record(path=path, name=name)
            )
            self.assertEqual(rc, 0)
            self.assertEqual(installed, [{name: self.payload}])
            self.assertEqual(requested, [
                f"https://archive.debian.org/debian/{path}"
            ])

    def test_later_package_failure_prevents_the_entire_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.record() + self.record(
                name="other.deb", path="pool/main/d/demo/other.deb",
                digest="SHA256:" + "0" * 64,
            )
            rc, installed, _ = self.exercise(root, plan)
            self.assertNotEqual(rc, 0)
            self.assertEqual(installed, [])
            self.assertEqual(
                {path.name: path.read_bytes() for path in root.iterdir()},
                {self.cache_name: self.payload},
            )

    def test_empty_plan_can_install_without_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, installed, requested = self.exercise(
                Path(tmp), "All packages are already installed.\n"
            )
            self.assertEqual(rc, 0)
            self.assertEqual(installed, [{}])
            self.assertEqual(requested, [])


class BullseyeUpdateGateTests(unittest.TestCase):
    def test_refused_update_never_enters_the_carrier(self):
        outcomes = iter([_completed(returncode=100)])
        with (
            mock.patch.object(
                cag.subprocess, "run", side_effect=lambda *args, **kwargs: next(outcomes)
            ),
            mock.patch.object(cag, "build_opener") as opener,
        ):
            rc = cag.apt_get(
                ["libdemo"], debian_bullseye_snapshot=True,
                sudo=False, attempts=1, sleep=lambda _seconds: None,
            )
        self.assertEqual(rc, 100)
        opener.assert_not_called()


class BullseyePrivateCacheTests(unittest.TestCase):
    def test_verified_files_survive_retries_then_the_cache_is_removed(self):
        roots = []
        payload = b"verified carrier bytes"

        def attempt(packages, *, archives, **kwargs):
            roots.append(archives)
            self.assertIsNotNone(archives)
            destination = archives / "demo.deb"
            if len(roots) == 1:
                self.assertEqual(list(archives.iterdir()), [])
                destination.write_bytes(payload)
                return 100
            self.assertEqual(destination.read_bytes(), payload)
            return 0

        with mock.patch.object(cag, "_attempt", side_effect=attempt):
            rc = cag.apt_get(
                ["libdemo"], debian_bullseye_snapshot=True,
                attempts=2, sleep=lambda _seconds: None,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(roots), 2)
        self.assertEqual(roots[0], roots[1])
        self.assertFalse(roots[0].exists())

    def test_failed_invocations_remove_their_private_cache(self):
        roots = []

        def attempt(packages, *, archives, **kwargs):
            roots.append(archives)
            self.assertEqual(list(archives.iterdir()), [])
            (archives / "demo.deb").write_bytes(b"verified carrier bytes")
            return 100

        with mock.patch.object(cag, "_attempt", side_effect=attempt):
            rc = cag.apt_get(
                ["libdemo"], debian_bullseye_snapshot=True,
                attempts=1, sleep=lambda _seconds: None,
            )
        self.assertEqual(rc, 100)
        self.assertEqual(len(roots), 1)
        self.assertFalse(roots[0].exists())


class BullseyeWorkerSignalTests(unittest.TestCase):
    def test_worker_preserves_group_kill_and_child_termination(self):
        for ignore_term, timeout in ((False, 1), (True, 1), (False, 0)):
            with (
                self.subTest(ignore_term=ignore_term, timeout=timeout),
                tempfile.TemporaryDirectory() as tmp,
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server,
            ):
                root = Path(tmp)
                endpoint = root / "ready.sock"
                server.bind(str(endpoint))
                server.listen(1)
                server.settimeout(5)
                fixture = root / "apt-get"
                fixture.write_text(
                    "#!/usr/bin/env python3\n"
                    "import fcntl\n"
                    "from pathlib import Path\n"
                    "import signal\n"
                    "import socket\n"
                    "import sys\n"
                    "import time\n"
                    "if '--print-uris' in sys.argv:\n"
                    "    raise SystemExit(0)\n"
                    "root = Path(__file__).resolve().parent\n"
                    "with (root / 'install.lock').open('w') as lock:\n"
                    "    fcntl.flock(lock, fcntl.LOCK_EX)\n"
                    f"    if {ignore_term!r}:\n"
                    "        signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:\n"
                    "        client.connect(str(root / 'ready.sock'))\n"
                    "        client.sendall(b'R')\n"
                    "        time.sleep(60)\n",
                    encoding="utf-8",
                )
                fixture.chmod(0o755)
                archives = root / "archives"
                archives.mkdir()
                environment = dict(os.environ)
                environment["PATH"] = os.pathsep.join(
                    (str(root), str(Path(sys.executable).parent), os.environ["PATH"])
                )
                worker = subprocess.Popen(
                    [
                        sys.executable, cag.__file__, cag._BULLSEYE_WORKER_FLAG,
                        str(archives), "--timeout", str(timeout), "--", "probe-package",
                    ],
                    env=environment, start_new_session=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                try:
                    channel, _ = server.accept()
                    with channel, (root / "install.lock").open("a") as lock:
                        channel.settimeout(5)
                        self.assertEqual(channel.recv(1), b"R")
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        os.killpg(worker.pid, signal.SIGTERM)
                        if timeout:
                            with self.assertRaises(subprocess.TimeoutExpired):
                                worker.wait(timeout=0.2)
                        else:
                            self.assertEqual(worker.wait(timeout=5), -signal.SIGTERM)
                        if ignore_term:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        else:
                            self.assertEqual(channel.recv(1), b"")
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    worker.communicate(timeout=5)


class RetryTests(unittest.TestCase):
    def test_transient_update_failure_then_success_retries_and_passes(self):
        # First attempt: `apt-get update` resets (rc 100), so install is not
        # even reached; second attempt: both succeed. End result is success.
        results = [
            _completed(returncode=100),  # attempt 1 update: reset
            _completed(returncode=0),  # attempt 2 update
            _completed(returncode=0),  # attempt 2 install
        ]
        sleeps: list[float] = []
        with mock.patch.object(cag.subprocess, "run", side_effect=results) as run:
            rc = cag.apt_get(["gcc"], sleep=sleeps.append)
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 3)
        # Exactly one backoff between the two attempts.
        self.assertEqual(sleeps, [cag.BACKOFF_SECONDS])

    def test_transient_install_failure_then_success(self):
        # update ok but install resets on attempt 1; both ok on attempt 2.
        results = [
            _completed(returncode=0),  # attempt 1 update
            _completed(returncode=100),  # attempt 1 install: reset
            _completed(returncode=0),  # attempt 2 update
            _completed(returncode=0),  # attempt 2 install
        ]
        with mock.patch.object(cag.subprocess, "run", side_effect=results) as run:
            rc = cag.apt_get(["gcc"], sleep=lambda _s: None)
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 4)

    def test_persistent_failure_exhausts_attempts_and_returns_nonzero(self):
        # A genuinely-broken install (e.g. unknown package) fails every
        # attempt. The retries must NOT mask it: the helper returns the last
        # non-zero exit so the job still goes red. This is the "cannot turn
        # a real packaging error green" invariant.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=100)
        ) as run:
            rc = cag.apt_get(["nope-not-a-package"], attempts=3, sleep=lambda _s: None)
        self.assertEqual(rc, 100)
        # 3 attempts; each stops at the failing `update` (rc!=0 short-circuits
        # install), so 3 subprocess calls, not 6.
        self.assertEqual(run.call_count, 3)

    def test_no_sleep_after_the_final_attempt(self):
        # Backoff happens BETWEEN attempts only, never after the last one
        # (no point sleeping when we are about to give up).
        sleeps: list[float] = []
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=1)
        ):
            cag.apt_get(["gcc"], attempts=3, sleep=sleeps.append)
        self.assertEqual(len(sleeps), 2)  # between 1->2 and 2->3, not after 3

    def test_single_attempt_does_not_sleep(self):
        sleeps: list[float] = []
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=1)
        ):
            rc = cag.apt_get(["gcc"], attempts=1, sleep=sleeps.append)
        self.assertEqual(rc, 1)
        self.assertEqual(sleeps, [])

    def test_spawn_failure_is_treated_as_nonzero_and_retried(self):
        # If the OS cannot spawn apt-get at all (OSError), _run returns 1 and
        # the loop retries it like any other transient failure.
        with mock.patch.object(
            cag.subprocess, "run", side_effect=OSError("boom")
        ) as run:
            rc = cag.apt_get(["gcc"], attempts=2, sleep=lambda _s: None)
        self.assertEqual(rc, 1)
        # 2 attempts, each failing to spawn `update` (install never reached).
        self.assertEqual(run.call_count, 2)


class TimeoutTests(unittest.TestCase):
    # The ceiling is enforced by the coreutils `timeout` binary baked into the
    # command (NOT subprocess.run(timeout=)), because SIGKILLing our direct
    # child `sudo` would orphan the apt-get it spawned and leave the apt lock
    # held -- exactly the failure these tests pin down. So a "timed out"
    # attempt manifests as `timeout` exiting non-zero (124/137), which the mock
    # returns as an ordinary returncode.

    def test_timed_out_command_is_retried_and_self_heals(self):
        # attempt 1 `update` is killed by the wrapper (coreutils `timeout`
        # returns 124); the loop must treat it as a failed attempt and retry,
        # NOT poison the retry the way an orphaned apt-get would. attempt 2:
        # both succeed.
        results = [
            _completed(returncode=124),  # attempt 1 update: timed out + killed
            _completed(returncode=0),  # attempt 2 update
            _completed(returncode=0),  # attempt 2 install
        ]
        sleeps: list[float] = []
        with mock.patch.object(cag.subprocess, "run", side_effect=results) as run:
            rc = cag.apt_get(["gcc"], sleep=sleeps.append)
        self.assertEqual(rc, 0)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleeps, [cag.BACKOFF_SECONDS])

    def test_persistent_timeout_exhausts_attempts_and_returns_the_code(self):
        # A mirror that stalls on every attempt must not wedge forever: each
        # attempt is killed by the wrapper (124), and the helper gives up with
        # that code so the job goes red instead of burning the 6-hour cap.
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=124)
        ) as run:
            rc = cag.apt_get(["gcc"], attempts=3, sleep=lambda _s: None)
        self.assertEqual(rc, 124)
        self.assertIn(rc, cag.TIMEOUT_EXIT_CODES)
        self.assertEqual(run.call_count, 3)

    def test_kill_after_exit_code_is_recognised_as_a_timeout(self):
        # 137 (128+9) is what `timeout` returns when apt-get ignored SIGTERM
        # and had to be SIGKILLed at --kill-after; it is still a timeout kill.
        self.assertIn(137, cag.TIMEOUT_EXIT_CODES)
        with mock.patch.object(
            cag.subprocess, "run", return_value=_completed(returncode=137)
        ):
            rc = cag.apt_get(["gcc"], attempts=2, sleep=lambda _s: None)
        self.assertEqual(rc, 137)


class CliTests(unittest.TestCase):
    def test_hosted_and_snapshot_sources_are_mutually_exclusive(self):
        with (
            mock.patch.object(cag, "configure_hosted_ubuntu_mirrors") as hosted,
            mock.patch.object(cag, "configure_bullseye_snapshot") as snapshot,
            mock.patch.object(cag, "apt_get") as apt,
        ):
            with self.assertRaises(SystemExit) as raised:
                cag.main(["--github-hosted-ubuntu-mirrors", "--debian-bullseye-snapshot", "gcc"])
        self.assertEqual(raised.exception.code, 2)
        hosted.assert_not_called()
        snapshot.assert_not_called()
        apt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
