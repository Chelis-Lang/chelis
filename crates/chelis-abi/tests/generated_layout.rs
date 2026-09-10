//! Executed C/Rust layout agreement, not a text-only descriptor assertion.
//! The host views are exactly [05-OP-31]; the device packet follows C3.

use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::process::Command;

fn rendered_header() -> String {
    format!(
        "#include \"chelis_runtime_dtype.h\"\n{}\n{}",
        chelis_abi::render::host_views_header(),
        chelis_abi::render::device_descriptor_header()
    )
}

mod owner {
    use super::*;

    chelis_abi::define_read_view!(ReadView);
    chelis_abi::define_write_view!(WriteView);
    chelis_abi::define_device_descriptor!(DeviceDescriptor);

    pub fn rust_layout() -> Vec<usize> {
        // These assignments enforce exact types, including pointer-based rank
        // metadata. An int32 field or fixed array fails Rust compilation.
        fn read_fields(view: &ReadView) {
            let _: *const c_void = view.data;
            let _: i64 = view.count;
            let _: u8 = view.dtype;
            let _: [u8; 7] = view.reserved;
        }
        fn write_fields(view: &WriteView) {
            let _: *mut c_void = view.data;
            let _: i64 = view.count;
            let _: u8 = view.dtype;
            let _: [u8; 7] = view.reserved;
        }
        fn device_fields(packet: &DeviceDescriptor) {
            let _: *mut c_void = packet.data;
            let _: *const i64 = packet.shape;
            let _: *const i64 = packet.strides;
            let _: i32 = packet.rank;
            let _: i64 = packet.count;
            let _: i64 = packet.byte_capacity;
            let _: u8 = packet.dtype;
            let _: u8 = packet.ownership;
            let _: [u8; 2] = packet.reserved;
        }
        let _ = (read_fields, write_fields, device_fields);
        vec![
            size_of::<ReadView>(),
            align_of::<ReadView>(),
            offset_of!(ReadView, data),
            offset_of!(ReadView, count),
            offset_of!(ReadView, dtype),
            offset_of!(ReadView, reserved),
            size_of::<WriteView>(),
            align_of::<WriteView>(),
            offset_of!(WriteView, data),
            offset_of!(WriteView, count),
            offset_of!(WriteView, dtype),
            offset_of!(WriteView, reserved),
            size_of::<DeviceDescriptor>(),
            align_of::<DeviceDescriptor>(),
            offset_of!(DeviceDescriptor, data),
            offset_of!(DeviceDescriptor, shape),
            offset_of!(DeviceDescriptor, strides),
            offset_of!(DeviceDescriptor, rank),
            offset_of!(DeviceDescriptor, count),
            offset_of!(DeviceDescriptor, byte_capacity),
            offset_of!(DeviceDescriptor, dtype),
            offset_of!(DeviceDescriptor, ownership),
            offset_of!(DeviceDescriptor, reserved),
        ]
    }
}

struct ProbeDirectory(std::path::PathBuf);

impl Drop for ProbeDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn compile_and_run(header: &str, name: &str) -> std::process::Output {
    let directory = std::env::temp_dir().join(format!("chelis-abi-{}-{name}", std::process::id()));
    std::fs::create_dir(&directory).expect("unique probe directory");
    let directory = ProbeDirectory(directory);
    let source = directory.0.join("probe.c");
    let binary = directory.0.join("probe");
    let body = r#"
#include <stdio.h>
#include <stddef.h>
#define P(value) printf("%zu ", (size_t)(value))
int main(void) {
    P(sizeof(chelis_read_view)); P(_Alignof(chelis_read_view));
    P(offsetof(chelis_read_view,data)); P(offsetof(chelis_read_view,count));
    P(offsetof(chelis_read_view,dtype)); P(offsetof(chelis_read_view,reserved));
    P(sizeof(chelis_write_view)); P(_Alignof(chelis_write_view));
    P(offsetof(chelis_write_view,data)); P(offsetof(chelis_write_view,count));
    P(offsetof(chelis_write_view,dtype)); P(offsetof(chelis_write_view,reserved));
    P(sizeof(chelis_gpu_tensor)); P(_Alignof(chelis_gpu_tensor));
    P(offsetof(chelis_gpu_tensor,data)); P(offsetof(chelis_gpu_tensor,shape));
    P(offsetof(chelis_gpu_tensor,strides)); P(offsetof(chelis_gpu_tensor,rank));
    P(offsetof(chelis_gpu_tensor,count)); P(offsetof(chelis_gpu_tensor,byte_capacity));
    P(offsetof(chelis_gpu_tensor,dtype)); P(offsetof(chelis_gpu_tensor,ownership));
    P(offsetof(chelis_gpu_tensor,reserved));
    return 0;
}
"#;
    std::fs::write(&source, format!("{header}\n{body}")).unwrap();
    let runtime_include =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include");
    let compile = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg("-I")
        .arg(runtime_include)
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("C compiler");
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let output = Command::new(&binary)
        .output()
        .expect("execute layout probe");
    output
}

#[test]
fn generated_c_and_rust_layouts_agree_for_views_and_device_packet() {
    let header = rendered_header();
    let output = compile_and_run(&header, "positive");
    assert!(output.status.success());
    let c_layout: Vec<usize> = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|word| word.parse().unwrap())
        .collect();
    assert_eq!(c_layout, owner::rust_layout());
}

#[test]
fn narrowing_generated_count_is_observed_by_layout_probe() {
    let header = rendered_header();
    let mutation = header.replacen("int64_t count;", "int32_t count;", 1);
    assert_ne!(mutation, header, "mutation must alter a real declaration");
    let output = compile_and_run(&mutation, "narrow-count");
    assert!(output.status.success());
    let c_layout: Vec<usize> = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|word| word.parse().unwrap())
        .collect();
    assert_ne!(c_layout, owner::rust_layout());
}

#[test]
fn committed_view_and_device_fragments_are_exactly_generated() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for (path, rendered) in [
        (
            "../chelis-runtime/include/chelis_runtime_views.h",
            chelis_abi::render::host_views_header(),
        ),
        (
            "../chelis-backend-hip/runtime/chelis_device_descriptor.h",
            chelis_abi::render::device_descriptor_header(),
        ),
    ] {
        let actual = std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|error| panic!("missing generated fragment {path}: {error}"));
        assert_eq!(actual, rendered, "stale generated fragment: {path}");
    }
}

#[test]
fn generated_device_metadata_aliases_bind_exact_widths() {
    let header = chelis_abi::render::device_descriptor_header();
    assert!(header.contains("typedef int64_t chelis_device_metadata;"));
    assert!(header.contains("typedef int32_t chelis_device_rank;"));
}
