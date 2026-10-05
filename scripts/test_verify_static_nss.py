"""The static host-lookup check fails closed; no Docker required."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_static_nss as nss  # noqa: E402

FEDORA = "hosts:      files myhostname resolve [!UNAVAIL=return] dns\n"
ARCH = "# Name Service Switch\nhosts: mymachines resolve [!UNAVAIL=return] files myhostname dns\n"


def container(stdout: str, returncode: int = 0) -> nss.Docker:
    def run(argv):
        return subprocess.CompletedProcess(["docker", *argv], returncode, stdout, "")

    return run


class WitnessTests(unittest.TestCase):
    def test_plugins_are_the_unbuilt_services_a_missing_name_reaches(self) -> None:
        self.assertEqual(nss.plugin_services(FEDORA), ["myhostname", "resolve"])
        self.assertEqual(nss.plugin_services(ARCH), ["mymachines", "resolve"])
        self.assertEqual(nss.plugin_services("hosts: files dns\n"), [])
        self.assertEqual(nss.plugin_services("#hosts: myhostname\n"), [])
        # glibc may stop at an action item, so nothing after one counts.
        self.assertEqual(
            nss.plugin_services("hosts: files [NOTFOUND=return] myhostname\n"), []
        )

    def test_an_image_proves_nothing_unless_it_carries_a_named_plugin(self) -> None:
        carried = f"{FEDORA}{nss.MARKER}\n/usr/lib64/libnss_myhostname.so.2\n"
        self.assertEqual(nss.witness("fedora", container(carried)), "myhostname")
        for stdout in (
            f"{FEDORA}{nss.MARKER}\n/usr/lib64/libnss_files.so.2\n",
            f"hosts: files dns\n{nss.MARKER}\n/usr/lib/libnss_myhostname.so.2\n",
            FEDORA,
        ):
            with self.subTest(stdout=stdout), self.assertRaises(nss.CheckFailed):
                nss.witness("image", container(stdout))


class JudgeTests(unittest.TestCase):
    def test_only_an_ordinary_dns_error_naming_the_host_passes(self) -> None:
        line = (
            f"chelisup: install failed: network error fetching http://{nss.PROBE_HOST}/repos: "
            "error trying to connect: dns error: failed to lookup address information"
        )
        self.assertEqual(nss.judge(1, f"{line}\n"), line)
        refused = (
            f"chelisup: install failed: http://{nss.PROBE_HOST} is not an https URL"
        )
        for returncode, output, reason in [
            (136, "", "SIGFPE"),
            (139, line, "SIGSEGV"),
            (1, "chelisup: install failed: no GitHub token", "before its host lookup"),
            (1, refused, "without a lookup error"),
            (0, line, "succeeded"),
            (125, "docker: error response from daemon", "exited 125"),
        ]:
            with (
                self.subTest(returncode=returncode, output=output),
                self.assertRaisesRegex(nss.CheckFailed, reason),
            ):
                nss.judge(returncode, output)


if __name__ == "__main__":
    unittest.main()
