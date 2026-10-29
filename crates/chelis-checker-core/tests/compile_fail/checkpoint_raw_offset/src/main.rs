use chelis_checker_core::checkpoint_iter_compile_probe;
use chelis_checker_core::errors::DiagnosticSink;

fn raw_offset_is_rejected(sink: &DiagnosticSink<'_>) {
    checkpoint_iter_compile_probe(sink, 0usize);
}

fn main() {}
