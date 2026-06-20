# VNN-LIB Front-End: Placeholder Specification

Intended location: `spec/design/vnnlib_frontend_placeholder.md` (chelis repo).
Companion documents: the master plan (`spec/design/verification_stack_master_plan.md`), the whole-stack sketch (`spec/design/verification_stack_sketch.md`), the dependency map (`spec/design/verification_stack_dependency_map.md`), and the Beacon engine plan (`spec/design/beacon_plan.md`).
Status: placeholder, backlogged. This records the intended shape for continuity; it is not a build spec.

## Intent

A front-end that ingests VNN-LIB property specifications and ONNX networks and turns them into chelis IR verification goals dischargeable by Beacon's standalone path, making that path a VNN-COMP-capable entry. VNN-COMP uses standardized formats (ONNX for networks, VNN-LIB for specifications) and equal-cost evaluation, so a competition entry requires speaking those formats.

## High-level shape

- Parse a VNN-LIB specification into an input box plus an output-constraint goal, expressed in the box-and-output-range goal schema (master WI-5).
- Lower the ONNX network to the RISC IR via Hydronnx, which already performs ONNX-to-IR translation into the same DAG type Beacon consumes.
- Dispatch the resulting goal to Beacon's standalone path.
- Emit results in the VNN-COMP result format.

## Dependencies

- Beacon's standalone and Hydronnx path (Beacon plan WI-B9).
- The box and output-range goal schema (master WI-5).
- Hydronnx ONNX-to-IR translation, which exists but has no VNN-LIB ingestion today and lacks coverage for some ONNX ops (recon noted Round, ScatterElements, ScatterND, and MultiHeadAttention as unsupported); op-coverage gaps are part of this front-end's scope when it is built.

## Why deferred

The finance line is the wedge and carries the near-term value. The neural-network verification entry is the second front: it reuses Beacon and the IR convergence rather than adding new core capability, so it follows once Beacon is built. This document exists so the intended path is recorded in repo source and is not re-derived later.
