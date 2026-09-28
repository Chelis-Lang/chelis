"""Spec/11 ABI 2 obligations in the independent wire execution corpus.

These are corpus/selection controls. Actual codecs must execute the selected
cases in the wire factory before they provide authority.
"""

import json
import unittest

from capacity_census_wire_artifact import artifact_cases
from capacity_census_wire_structural import structural_evidence


class ArtifactAbi2Cases(unittest.TestCase):
    def test_only_version_two_has_positive_discriminator_execution(self):
        for codec in ("json", "construct"):
            cases = [c for c in artifact_cases()
                     if c.carrier == "ArtifactAbiVersion" and c.codec == codec]
            self.assertEqual([(c.input, c.expected) for c in cases
                              if c.expected is not None], [("2", 2)])
            for version in ("0", "1", "3", "4294967295"):
                rejected = [c for c in cases if c.input == version]
                self.assertEqual(len(rejected), 1)
                self.assertIsNone(rejected[0].expected)
                self.assertEqual(rejected[0].rejection_contains, "artifact ABI version")

    def test_old_producer_manifest_rejects_even_with_otherwise_valid_metadata(self):
        cases = artifact_cases()
        current = [c for c in cases if c.carrier == "CompiledArtifactManifest"
                   and c.expected is not None]
        self.assertTrue(current)
        self.assertEqual({c.expected["abi_version"] for c in current}, {2})
        legacy = [c for c in cases if c.identity ==
                  "CompiledArtifactManifest/json/legacy-valid-metadata"]
        self.assertEqual(len(legacy), 1)
        manifest = json.loads(legacy[0].input)
        self.assertEqual(manifest["abi_version"], 1)
        self.assertIn("inputs", manifest)
        self.assertIn("host_entry_name", manifest)
        self.assertIsNone(legacy[0].expected)
        self.assertEqual(legacy[0].rejection_contains, "artifact ABI version")

    def test_duplicate_and_malformed_version_headers_precede_invalid_metadata(self):
        cases = [c for c in artifact_cases()
                 if "/version-before-metadata-" in c.identity]
        inputs = {c.input for c in cases}
        for header in ("", ',"abi_version":1', ',"abi_version":3',
                       ',"abi_version":2.0', ',"abi_version":"2"',
                       ',"abi_version":2,"abi_version":2',
                       ',"abi_version":1,"abi_version":2',
                       ',"abi_version":2,"abi_version":1'):
            self.assertIn('{"inputs":"invalid"' + header + "}", inputs)
        self.assertTrue(all(c.expected is None and
                            c.rejection_contains == "artifact ABI version" for c in cases))

    def test_field_authority_requires_current_acceptance_and_both_version_boundaries(self):
        pairs = structural_evidence()["artifact-abi-version"]
        for codec in ("json", "construct"):
            for rejected in (0, 1, 3):
                self.assertIn((f"ArtifactAbiVersion/{codec}/2",
                               f"ArtifactAbiVersion/{codec}/{rejected}"), pairs)


if __name__ == "__main__":
    unittest.main()
