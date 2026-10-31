use chelis_abi::metadata::{ByteCount, ShapeMetadata};
use chelis_abi::{mapped, render};
use chelis_vocab::RuntimeDType;
use serde_json::{Value, json};
fn main() {
    let mut descriptors = Vec::new();
    for dtype in [RuntimeDType::F32, RuntimeDType::F64, RuntimeDType::I32, RuntimeDType::Bool] {
        for shape in [vec![], vec![2, 3], vec![0, 3], vec![0, -1],
            vec![0, i64::MAX, 2], vec![i64::MAX, 2], vec![i64::MAX / 4]] {
            let result = match ShapeMetadata::contiguous(&shape, dtype) {
                Ok(value) => json!({"status": "accepted", "rank": value.rank(),
                    "shape": value.shape(), "strides": value.strides(),
                    "elements": value.elements().get(), "bytes": value.bytes().get(),
                    "allocation": format!("{:?}", value.bytes().allocation()),
                    "target_32": format!("{:?}", value.bytes().project_limit(i32::MAX as u64))}),
                Err(error) => json!({"status": "rejected", "debug": format!("{error:?}"),
                    "message": error.to_string()}),
            };
            descriptors.push(json!({"dtype": format!("{dtype:?}"), "shape": shape, "result": result}));
        }
    }
    let ranges = [(-1, 1, 8), (0, -1, 8), (8, 0, 8), (8, 1, 8), (i64::MAX, 1, 8)]
        .map(|(offset, length, capacity)| format!("{:?}", mapped::mapped_range("oracle", offset, length, capacity)));
    let tensor_ranges = [-1, 0, 1, i64::MAX].map(|count|
        format!("{:?}", mapped::mapped_tensor_range(0, count, RuntimeDType::F32, 8)));
    let payloads: Vec<Value> = [vec![], vec![0, 1], vec![0, 2]].iter().map(|bytes|
        json!({"bytes": bytes, "result": format!("{:?}", mapped::check_bool_payload(bytes))})).collect();
    println!("CHECKER_ABI_SNAPSHOT {}", json!({"descriptors": descriptors,
        "ranges": ranges, "tensor_ranges": tensor_ranges, "payloads": payloads,
        "text": [format!("{:?}", mapped::mapped_text(4, b"hello")),
            format!("{:?}", mapped::mapped_text(4, &[0xff]))],
        "digests": [mapped::sha256_hex(b""), mapped::sha256_hex(b"abc")],
        "byte_counts": ([-1, 0, i64::MAX].map(|value| format!("{:?}", ByteCount::from_declared(value)))),
        "host_header": render::host_views_header(), "device_header": render::device_descriptor_header()}));
}
