"""Exact test selections owned by the C6 wire acceptance command.

These literal identities are reviewed alongside their requirements. Framework
receipts reject absent, ignored, skipped, failed and zero-match selections.
The selection is never discovered from whatever tests happen to remain.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from capacity_census_wire_runner import build_and_run_rust_test, run_python_tests

ROOT = Path(__file__).resolve().parent.parent

MUTATION_CONTROLS = (
    "test_capacity_census_wire_adapters.CodecIdentityControls.test_codec_line_and_column_movement_preserves_structural_identity",
    "test_capacity_census_wire_adapters.CodecIdentityControls.test_codec_provenance_and_implementation_mode_still_reject",
    "test_capacity_census_wire_adapters.CodecIdentityControls.test_changed_method_or_numeric_shape_cannot_reuse_identity",
    "test_capacity_census_wire_schema.SchemaCases.test_required_span_decoder_is_confined_to_its_exact_optional_text_field",
    "test_capacity_census_wire_invocation_owners.CodecSpecializations.test_specializations_bind_owner_payload_and_serializer_shapes",
    "test_capacity_census_wire_invocation_owners.CodecSpecializations.test_specializations_reject_missing_duplicate_or_replaced_templates",
    "test_capacity_census_wire_invocation_owners.CodecSpecializations.test_specializations_reject_rebound_parameters_and_unknown_shapes",
    "test_capacity_census_wire_invocation_owners.CodecSpecializations.test_specializations_cannot_admit_bare_numbers_or_binary_owners",
    "test_capacity_census_wire_invocation_owners.CodecSpecializations.test_independent_publication_cannot_borrow_a_surrounding_codec",
    "test_capacity_census_wire_publication.PublishedRoots.test_check_report_protocol_binds_existing_dto_producer_and_consumer",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_check_report_publisher_requires_its_compiled_inherent_receiver",
    "test_capacity_census_wire_invocation_owners.InvocationOwnership.test_check_report_call_replays_exact_compiler_publisher_and_payload",
    "test_capacity_census_wire_runner.LibtestReceipts.test_failed_native_execution_reports_its_actual_output",
    "test_capacity_census_wire_invocation_owners.InvocationOwnership.test_cache_load_and_save_bind_each_exact_owner_and_payload",
    "test_capacity_census_wire_calls.DriverBuildControls.test_failed_cargo_reports_rendered_diagnostics_from_json_stdout",
    "test_capacity_census_cache_publication.CachePublicationSelection.test_fixture_inventory_accepts_exact_owned_sources_and_rejects_drift",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_shape_helper_joins_schema_calls_with_its_required_return_observation",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_shape_helper_validates_every_return_observation",
    "test_capacity_census_wire_calls.DriverBuildControls.test_actual_dep_info_and_policy_are_bound_and_missing_dep_info_rebuilds",
    "test_capacity_census_wire_calls.DriverBuildControls.test_source_dependencies_and_policy_changes_invalidate_cached_build",
    "test_capacity_census_wire_calls.DriverBuildControls.test_actual_clippy_rejects_hash_types_and_undeclared_configuration",
    "test_capacity_census_wire_calls.InvocationControls.test_aliased_macro_calls_bind_primitive_and_container_substitutions",
    "test_capacity_census_wire_calls.InvocationControls.test_private_generic_cache_instances_are_derived_and_open_generic_fails",
    "test_capacity_census_wire_calls.InvocationControls.test_custom_entrypoint_and_direct_serializer_do_not_need_known_names",
    "test_capacity_census_wire_calls.InvocationControls.test_indirect_encoder_and_erased_function_value_fail_closed",
    "test_capacity_census_wire_calls.InvocationControls.test_dynamic_container_and_opaque_returns_are_explicit_obligations",
    "test_capacity_census_wire_calls.InvocationControls.test_configuration_changes_change_compiler_evidence",
    "test_capacity_census_wire_calls.InvocationControls.test_codec_impl_does_not_hide_an_independent_publication",
    "test_capacity_census_wire_calls.InvocationControls.test_real_generic_derive_machinery_is_a_codec_obligation",
    "test_capacity_census_wire_calls.InvocationControls.test_trait_dispatched_and_generic_callback_encoders_fail_if_unresolved",
    "test_capacity_census_wire_calls.InvocationControls.test_text_conversion_entry_is_closed_by_an_explicit_string_parameter",
    "test_capacity_census_wire_calls.InvocationControls.test_schema_metadata_role_requires_the_actual_trait_identity",
    "test_capacity_census_wire_calls.InvocationControls.test_payload_shapes_resolve_aliases_arrays_slices_and_nominal_arguments",
    "test_capacity_census_wire_calls.InvocationControls.test_inherent_method_identity_retains_self_type_and_exact_item_name",
    "test_capacity_census_wire_calls.InvocationControls.test_returned_nominal_wrapper_cannot_hide_dynamic_json",
    "test_capacity_census_wire_calls.InvocationControls.test_supertrait_bound_and_external_opaque_value_are_resolved",
    "test_capacity_census_wire_calls.InvocationControls.test_missing_or_replaced_compiler_evidence_is_not_admission",
    "test_capacity_census_wire_calls.InvocationControls.test_defining_codec_package_requires_the_exact_locked_registry_origin",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_closure_joins_its_exact_compiler_parent_without_parsing_impl_names",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_wrong_trait_and_missing_graph_owner_are_rejected",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_missing_conflicting_or_cyclic_ancestry_is_rejected",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_generated_local_marker_inherits_only_a_verified_containing_owner",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_schema_calls_and_scoped_serde_calls_keep_their_numeric_payload",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_binary_calls_are_left_for_the_binary_owner_discharge",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_shape_helper_requires_exact_path_schema_type_and_tensor_owner",
    "test_capacity_census_wire_schema_publication.SchemaPublicationControls.test_empty_or_unresolved_evidence_cannot_be_an_owner_proof",
    "test_capacity_census_wire_invocation_owners.InvocationOwnership.test_known_carrier_and_container_do_not_authorize_arbitrary_numeric_payload",
    "test_capacity_census_wire_invocation_owners.InvocationOwnership.test_payload_shape_binds_width_container_and_defining_crate",
    "test_capacity_census_wire_invocation_owners.InvocationOwnership.test_binary_owner_is_an_exact_use_contract_and_never_numeric_authority",
    "test_capacity_census_wire_invocation_owners.InvocationOwnership.test_invocation_receipt_cannot_be_constructed_from_a_saved_report",
    "test_capacity_census_cache_publication.CachePublicationSelection.test_exact_owners_have_companion_unowned_compile_rejections",
    "test_capacity_census_cache_publication.CachePublicationSelection.test_runtime_receipt_requires_exact_framework_selection",
    "test_capacity_census_cache_publication.CachePublicationSelection.test_actual_trait_inventory_rejects_missing_extra_or_generic_owners",
    "test_capacity_census_wire_adapters.SourceIdentityControls.test_oracle_selection_and_consumer_source_changes_invalidate_evidence",
    "test_capacity_census_wire_adapters.SourceIdentityControls.test_actual_source_and_authority_bytes_ignore_git_visibility_flags",
    "test_capacity_census_graph.GraphClosure.test_private_nominal_container_generic_and_nonnumeric_companion",
    "test_capacity_census_graph.GraphClosure.test_missing_definition_unknown_primitive_and_open_generic_fail",
    "test_capacity_census_graph.GraphClosure.test_imported_definition_requires_actual_artifact",
    "test_capacity_census_graph.GraphClosure.test_container_lookalike_is_not_std_adapter",
    "test_capacity_census_graph.GraphClosure.test_recursive_and_growing_generic_equations_reach_fixed_point",
    "test_capacity_census_graph.GraphClosure.test_mutual_alias_cycles_and_generic_argument_errors_fail",
    "test_capacity_census_graph.GraphClosure.test_public_reexport_discovers_private_definition",
    "test_capacity_census_graph.GraphClosure.test_stripped_fields_unknown_format_and_unsupported_generics_reject",
    "test_capacity_census_graph.GraphClosure.test_generic_multiple_instantiations_and_private_relocation_remain_visible",
    "test_capacity_census_graph.GraphClosure.test_alias_array_tuple_reference_are_traversed_and_pointer_rejected",
    "test_capacity_census_graph.CarrierRecognition.test_source_roles_require_exact_shape_and_closed_role",
    "test_capacity_census_graph.CarrierRecognition.test_tag_width_relocation_and_new_numeric_field_invalidate_admission",
    "test_capacity_census_graph.CarrierRecognition.test_arbitrary_tagged_f64_and_integer_dtype_lookalikes_never_qualify",
    "test_capacity_census_graph.CarrierRecognition.test_mixed_container_does_not_authorize_numeric_sibling",
    "test_capacity_census_graph.CarrierRecognition.test_unknown_serde_and_custom_serializer_fail_closed",
    "test_capacity_census_graph.CarrierRecognition.test_input_reference_keeps_axis_separate_and_rejects_role_swaps",
    "test_capacity_census_graph.CarrierRecognition.test_impostor_serialize_trait_and_missing_trait_identity_do_not_prove_codec",
    "test_capacity_census_graph.CarrierRecognition.test_one_sided_derived_codec_identity_is_preserved",
    "test_capacity_census_graph.CarrierRecognition.test_missing_serde_proof_and_stale_leaf_are_rejected",
    "test_capacity_census_graph.SerializedCandidates.test_outside_schema_export_and_unknown_custom_codec_remain_candidates",
    "test_capacity_census_graph.SerializedCandidates.test_non_type_import_is_not_a_root_but_unresolved_type_import_rejects",
    "test_capacity_census_graph.ActualRustdoc.test_actual_serde_attributes_and_custom_impl_are_distinguished",
    "test_capacity_census_graph.ActualRustdoc.test_actual_private_reexport_generic_and_recursive_artifact",
    "test_capacity_census_wire_publication.PublishedRoots.test_unreferenced_private_protocol_is_a_root_and_cannot_hide_a_number",
    "test_capacity_census_wire_publication.PublishedRoots.test_private_binary_owner_is_recorded_and_does_not_authorize_a_new_protocol",
    "test_capacity_census_wire_publication.PublishedRoots.test_private_binding_serde_definition_requires_a_shared_wire_owner",
    "test_capacity_census_wire_publication.PublishedRoots.test_outside_schema_and_private_reachable_numeric_fields_are_discovered",
    "test_capacity_census_wire_publication.PublishedRoots.test_binary_ownership_is_exact_and_does_not_spread_to_new_candidates",
    "test_capacity_census_wire_publication.PublishedRoots.test_binary_owner_cannot_hide_a_cache_reachable_from_a_wire_root",
    "test_capacity_census_wire_publication.PublishedRoots.test_generic_exports_require_concrete_reachable_instantiations",
    "test_capacity_census_wire_publication.PublishedRoots.test_reexport_alias_is_accounted_without_duplicate_declared_leaves",
    "test_capacity_census_wire_publication.ActualPublicationArtifacts.test_rust_source_mutations_cannot_hide_new_or_relocated_numeric_capacity",
    "test_capacity_census_wire_authority.WireAuthorityPartition.test_exact_bijection_rejects_missing_duplicate_and_surplus_authority",
    "test_capacity_census_wire_authority.WireAuthorityPartition.test_tagging_arbitrary_numeric_data_does_not_make_it_source_metadata",
    "test_capacity_census_wire_authority.WireAuthorityPartition.test_legacy_or_nonnumeric_dispositions_cannot_admit_a_numeric_leaf",
    "test_capacity_census_wire_fixed_roles.FixedFieldRoles.test_field_contract_accepts_its_carrier_and_rejects_unrelated_tagged_data",
    "test_capacity_census_wire_fixed_roles.FixedFieldRoles.test_aliases_are_transparent_but_generic_wrappers_cannot_hide_a_new_use",
    "test_capacity_census_wire_fixed_roles.FixedFieldRoles.test_duplicate_roles_or_changed_carrier_do_not_supply_authority",
    "test_capacity_census_wire_operations.OperationFields.test_each_registered_parameter_has_an_existing_exact_operation_atom",
    "test_capacity_census_wire_operations.OperationFields.test_descriptor_type_and_selected_atom_are_both_checked",
    "test_capacity_census_wire_structural.StructuralRoles.test_source_carriers_keep_their_closed_point_and_measured_range_shapes",
    "test_capacity_census_wire_structural.StructuralRoles.test_all_decided_roles_require_the_exact_field_width_and_container",
    "test_capacity_census_wire_structural.StructuralRoles.test_a_tag_or_reference_name_does_not_grant_a_structural_role",
    "test_capacity_census_wire_structural.StructuralRoles.test_private_encoder_decoder_slots_share_their_public_contract",
    "test_capacity_census_wire_structural.StructuralRoles.test_a_duplicate_or_missing_edge_is_not_a_second_admission_path",
    "test_capacity_census_wire_structural.StructuralRoles.test_every_role_requires_its_selected_acceptance_and_rejection_executions",
    "test_capacity_census_wire_materialization.MaterializationCases.test_each_source_field_has_its_own_transport_and_admission_evidence",
    "test_capacity_census_wire_materialization.MaterializationCases.test_selection_is_nonempty_unique_and_pairs_every_route",
    "test_capacity_census_wire_materialization.MaterializationCases.test_independent_expectations_keep_wide_integer_and_negative_zero",
    "test_capacity_census_wire_materialization.MaterializationCases.test_rejections_need_actual_matching_diagnostics",
    "test_capacity_census_wire_local.LocalPublicationControls.test_module_dtos_allow_serde_internals_but_local_dtos_fail",
    "test_capacity_census_wire_local.LocalPublicationControls.test_aliases_renames_and_macros_do_not_create_a_local_exception",
    "test_capacity_census_wire_local.LocalPublicationControls.test_active_configuration_is_expanded_and_inactive_code_is_not_a_receipt",
    "test_capacity_census_wire_local.LocalPublicationControls.test_block_local_alias_module_and_impl_are_rejected",
    "test_capacity_census_wire_local.LocalPublicationControls.test_const_and_closure_declarations_cannot_hide_from_rustdoc",
    "test_capacity_census_wire_local.LocalPublicationControls.test_automatically_derived_is_not_authority_for_manual_local_helpers",
    "test_capacity_census_wire_local.LocalPublicationControls.test_generated_serde_custom_field_helpers_match_the_actual_derive",
    "test_capacity_census_wire_local.LocalPublicationControls.test_missing_or_malformed_compiler_output_fails_closed",
    "test_capacity_census_wire_local.LocalPublicationControls.test_provenance_requires_exact_definition_and_complete_parent_linkage",
    "test_capacity_census_wire_local.LocalPublicationControls.test_custom_derive_named_serialize_and_source_comments_have_no_authority",
    "test_capacity_census_wire_local.LocalPublicationControls.test_format_interpolation_is_visited",
    "test_capacity_census_wire_local.LocalPublicationControls.test_schema_derive_admits_only_proven_impls_and_unit_markers",
    "test_capacity_census_wire_runner.LibtestReceipts.test_cargo_artifact_must_be_the_unique_requested_test_in_this_target",
    "test_capacity_census_wire_runner.LibtestReceipts.test_library_artifact_requires_exact_library_kind_and_test_profile",
    "test_capacity_census_wire_runner.LibtestReceipts.test_actual_rust_framework_does_not_accept_ignored_or_zero_match",
    "test_capacity_census_wire_runner.LibtestReceipts.test_exact_positive_selection_and_execution_are_required",
    "test_capacity_census_wire_runner.SupervisedUnittest.test_framework_records_both_real_test_bodies",
    "test_capacity_census_wire_runner.SupervisedUnittest.test_skips_missing_tests_errors_and_early_exit_cannot_pass",
    "test_capacity_census_wire_runner.SupervisedUnittest.test_duplicate_empty_and_modified_selection_are_rejected",
    "test_capacity_census_wire_verifier.FinalWireCensus.test_descriptor_and_saved_receipt_cannot_construct_a_wire_witness",
    "test_capacity_census_wire_verifier.FinalWireCensus.test_final_rows_preserve_derived_capacity_and_exact_identity",
    "test_capacity_census_wire_verifier.FinalWireCensus.test_baseline_comparison_cannot_grant_exception_or_new_leaf_authority",
    "test_capacity_census_wire_verifier.FinalWireCensus.test_wire_cli_rejects_supplied_artifact_before_attempting_discovery",
)

HULL_CONSUMER_CONTROLS = tuple(
    "tests.conformance.hull.test_run_conformance.ExecutionV3ConsumerTests." + method
    for method in (
        "test_floats_decode_at_their_declared_storage_width",
        "test_integer_width_limits_are_exact_and_out_of_range_is_rejected",
        "test_legacy_and_malformed_scalar_codecs_never_produce_agreement",
        "test_tensor_shape_and_every_payload_element_are_validated",
        "test_execution_version_and_json_grammar_are_required",
        "test_boolean_value_and_storage_require_json_booleans",
        "test_nonfinite_agreement_requires_matching_class_and_infinity_sign",
    )
)

RUST_CONSUMER_CONTROLS = (
    (
        "chelis-compiler-api",
        "wire_extent_witness",
        (
            "witness_roundtrip_retains_claim_dependency_and_provenance",
            "wire_dag_v8_rejects_before_witness_body_decode",
            "malformed_claims_and_invocation_edges_are_not_decoded_or_encoded",
        ),
    ),
    (
        "chelis-compiler-api",
        "wire_random_domains",
        (
            "each_random_seed_preserves_all_bits_and_rejects_alternate_or_missing_encodings",
            "each_random_parameter_requires_its_exact_active_float_dtype",
            "each_random_parameter_rejects_nonfinite_values_at_object_admission",
            "each_random_parameter_rejects_malformed_tags_and_payloads",
            "dropout_has_closed_zero_and_open_one_bounds_in_every_float_dtype",
            "uniform_accepts_equal_bounds_and_rejects_independently_reversed_low_or_high",
            "random_operations_preserve_the_data_inputs_exact_shape_and_dtype",
            "uniform_accepts_only_an_optional_scalar_bool_activation",
            "uniform_activation_references_must_resolve_to_earlier_nodes",
            "uniform_activation_does_not_bypass_template_or_parameter_domains",
            "dropout_rejects_an_activation_input",
            "gradient_random_lowering_preserves_scalar_bool_activation",
        ),
    ),
    (
        "chelis-compiler-api",
        "wire_dag_domains",
        (
            "random_parameter_carriers_have_the_inputs_exact_dtype_and_domain",
            "every_float_dtype_checks_random_domains_at_its_governing_arithmetic_width",
            "live_lowering_emits_exact_parameter_carriers_and_rejects_bad_domains",
            "every_registered_single_axis_operation_checks_its_own_input_rank",
            "forward_and_adjoint_window_fields_reject_bad_domains_at_both_edges",
            "shared_expand_ir_accepts_broadcast_and_insert_axis_domains",
            "shared_expand_ir_rejects_axes_outside_its_selected_layout",
            "window_and_onehot_extents_are_positive_int64",
            "standalone_dimension_carriers_reject_negative_extents_before_dag_admission",
        ),
    ),
    (
        "chelis-compiler-api",
        "wire_dag_v6_count",
        (
            "current_wire_dag_count_round_trips_canonical_axes",
            "current_wire_dag_rejects_noncanonical_count_axes_on_encode_and_decode",
            "current_wire_dag_rejects_count_semantic_dtype_and_shape_corruption",
            "current_wire_dag_rejects_count_without_one_resolvable_input",
            "current_wire_dag_requires_accumulator_fields_in_current_ops",
        ),
    ),
    (
        "chelis-cli",
        "cli",
        (
            "eval_json_emits_int64_scalar",
            "eval_json_emits_tensor_shape_and_data",
            "eval_json_emits_tuple_of_int64",
            "eval_json_def_only_emits_empty_roots_json",
            "eval_json_unbound_name_errors_with_empty_stdout",
            "check_does_not_report_perfect_score_with_errors",
        ),
    ),
    (
        "chelis-tide",
        "api",
        (
            "lower_and_grad_responses_carry_validated_schema_version",
            "eval_endpoint_uses_named_bindings_and_rejects_missing_inputs",
            "eval_endpoint_returns_host_values_and_transcript",
        ),
    ),
)


def descriptor_controls():
    return tuple(
        name
        for name in MUTATION_CONTROLS
        if ".GraphClosure." in name or ".PublishedRoots." in name
    )


def authority_controls():
    return tuple(
        name
        for name in MUTATION_CONTROLS
        if ".FinalWireCensus." in name or ".WireAuthorityPartition." in name
    )


def execute_acceptance_controls(root: Path, target: Path):
    executions = [
        run_python_tests(root, MUTATION_CONTROLS),
        run_python_tests(root, HULL_CONSUMER_CONTROLS),
    ]
    for package, name, selected in RUST_CONSUMER_CONTROLS:
        executions.append(
            build_and_run_rust_test(root, target, package, name, selected)
        )
    executions.append(build_and_run_rust_test(
        root, target, "chelis-compiler-api", "chelis_compiler_api",
        (
            "compiler::tests::native_wire_witness_projection_preserves_exact_claims_and_provenance",
            "compiler::tests::native_wire_witness_projection_rejects_invalid_requirements_and_edges",
        ), kind="lib",
    ))
    return tuple(executions)


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["descriptor-controls"]:
            selection = descriptor_controls()
        elif sys.argv[1:] == ["authority-controls"]:
            selection = authority_controls()
        else:
            raise ValueError("expected descriptor-controls or authority-controls")
        print(json.dumps(asdict(run_python_tests(ROOT, selection)), sort_keys=True))
    except (RuntimeError, ValueError, OSError) as error:
        print(f"wire acceptance controls failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
