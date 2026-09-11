"""Final authority controls for the four registered native Python bindings."""

import unittest


class NativeAuthorityContract(unittest.TestCase):
    def test_native_classification_contract_is_exact_and_complete(self):
        from capacity_census_native_authority import NATIVE_CLASSIFICATIONS

        self.assertEqual(
            NATIVE_CLASSIFICATIONS,
            {
                "chelis_python::CompiledModel::__call__": (
                    "TaggedTransport",
                    "native/compiled-tensor-call",
                    ("float-carrier", "numeric-param", "numeric-return"),
                ),
                "chelis_python::NativeTensor::__dlpack__": (
                    "TaggedTransport",
                    "native/dlpack-capsule",
                    ("float-carrier", "numeric-param", "numeric-return"),
                ),
                "chelis_python::NativeTensor::__dlpack_device__": (
                    "TaggedTransport",
                    "native/dlpack-device",
                    ("numeric-return",),
                ),
                "chelis_python::NativeTensor::shape": (
                    "NumericOperation",
                    "[05-OP-45]",
                    ("numeric-return",),
                ),
            },
        )

    def test_authority_witness_cannot_be_constructed_from_descriptors(self):
        from capacity_census_native_authority import VerifiedNativeBindings

        with self.assertRaises(TypeError):
            VerifiedNativeBindings(report={"passed": True})


if __name__ == "__main__":
    unittest.main()
