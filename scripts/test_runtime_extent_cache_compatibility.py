"""Independent negative controls for the two-binary cache execution oracle."""

import unittest

from runtime_extent_cache_compatibility import assert_result


class ResultContractTests(unittest.TestCase):
    def test_matching_reshape_requires_exact_shape_and_values(self):
        expected = "main = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])"
        good = {"exit": 0, "stdout": expected, "stderr": ""}
        assert_result(good, "reshape", True)
        for wrong in (
            "",
            expected.replace("[2, 2]", "[4]"),
            expected.replace("4.0", "0.0"),
        ):
            with self.subTest(output=wrong), self.assertRaises(AssertionError):
                assert_result({**good, "stdout": wrong}, "reshape", True)

    def test_mismatch_requires_the_correct_operation_source_axis_and_number(self):
        message = "error: extent `2`: claimed = 2, reshape axis 0 = 3\nnumeric trap: domain in reshape at int64"
        good = {"exit": 1, "stdout": "", "stderr": message}
        assert_result(good, "reshape", False)
        for wrong in (
            message.replace("axis 0 = 3", "axis 0 = 30"),
            message.replace("claimed = 2", "claimed = 20"),
            message.replace("reshape axis", "unrelated axis"),
            message.replace("axis 0", "axis 1"),
            message.replace("domain in reshape", "domain in load"),
            message.replace("numeric trap:", "unsupported numeric trap:"),
        ):
            with self.subTest(message=wrong), self.assertRaises(AssertionError):
                assert_result({**good, "stderr": wrong}, "reshape", False)
        with self.assertRaises(AssertionError):
            assert_result({**good, "exit": 0}, "reshape", False)


if __name__ == "__main__":
    unittest.main()
