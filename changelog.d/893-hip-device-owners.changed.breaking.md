HIP generated programs use opaque device owners, immutable checked metadata,
and 64-bit kernel geometry specialized to the program's checked ranks. Reshape
and BLAS operand preparation materialize logical input order into storage
accounted by the shared planner. Direct verified HIP codegen callers run
`prepare_dag_for_codegen` before ownership lowering; unprepared BLAS borrows
fail the storage-plan precondition.
HIP builds require matching official hipBLAS 3+ headers and libraries.
