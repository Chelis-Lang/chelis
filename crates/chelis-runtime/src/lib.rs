#![allow(
    clippy::missing_safety_doc,
    non_camel_case_types,
    private_interfaces,
    dangerous_implicit_autorefs
)]

use libc::{c_char, c_int};
use std::ffi::{CStr, CString};
use std::ptr;

const CHELIS_F32: c_int = 0;
const CHELIS_F64: c_int = 1;
const CHELIS_I32: c_int = 2;
const CHELIS_BOOL: c_int = 3;
const CHELIS_MAX_DIM: usize = 8;

macro_rules! runtime_fail {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
        std::process::exit(1);
    }};
}

#[repr(C)]
pub struct chelis_tensor {
    pub data: *mut f32,
    pub shape: [c_int; CHELIS_MAX_DIM],
    pub strides: [c_int; CHELIS_MAX_DIM],
    pub ndim: c_int,
    pub dtype: c_int,
    pub size: c_int,
    pub owns_data: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_string {
    pub handle: *mut RuntimeString,
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum chelis_value_tag {
    CHELIS_VALUE_INT64,
    CHELIS_VALUE_FLOAT64,
    CHELIS_VALUE_BOOL,
    CHELIS_VALUE_STRING,
    CHELIS_VALUE_TENSOR,
    CHELIS_VALUE_LIST,
    CHELIS_VALUE_TUPLE,
    CHELIS_VALUE_DICT,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union chelis_value_union {
    pub i64_: i64,
    pub f64_: f64,
    pub boolean: bool,
    pub string: chelis_string,
    pub tensor: *mut chelis_tensor,
    pub list: *mut chelis_list,
    pub tuple: *mut chelis_tuple,
    pub dict: *mut chelis_dict,
}

unsafe fn chelis_flat_to_indices(flat: c_int, shape: *const c_int, ndim: c_int, out: *mut c_int) {
    let mut flat = flat;
    for d in (0..ndim as isize).rev() {
        *out.offset(d) = flat % *shape.offset(d);
        flat /= *shape.offset(d);
    }
}

unsafe fn chelis_indices_to_flat(
    indices: *const c_int,
    strides: *const c_int,
    ndim: c_int,
) -> c_int {
    let mut flat = 0;
    for d in 0..ndim as isize {
        flat += *indices.offset(d) * *strides.offset(d);
    }
    flat
}

unsafe fn chelis_is_contiguous(t: *const chelis_tensor) -> c_int {
    let mut expected = 1;
    for d in (0..(*t).ndim as isize).rev() {
        if (*t).strides[d as usize] != expected {
            return 0;
        }
        expected *= (*t).shape[d as usize];
    }
    1
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_value {
    pub tag: chelis_value_tag,
    pub as_: chelis_value_union,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_dict_entry {
    pub key: chelis_value,
    pub value: chelis_value,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_option_i64 {
    pub is_some: bool,
    pub value: i64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_option_f64 {
    pub is_some: bool,
    pub value: f64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct chelis_option_value {
    pub is_some: bool,
    pub value: chelis_value,
}

#[repr(C)]
pub struct chelis_list {
    refcount: usize,
    items: Vec<chelis_value>,
}

#[repr(C)]
pub struct chelis_tuple {
    refcount: usize,
    items: Vec<chelis_value>,
}

#[repr(C)]
pub struct chelis_dict {
    refcount: usize,
    entries: Vec<chelis_dict_entry>,
}

struct RuntimeString {
    refcount: usize,
    value: String,
    cstring: CString,
}

unsafe fn retain_string_handle(handle: *mut RuntimeString) {
    if !handle.is_null() {
        (*handle).refcount += 1;
    }
}

unsafe fn release_string_handle(handle: *mut RuntimeString) {
    if !handle.is_null() {
        let inner = &mut *handle;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            drop(Box::from_raw(handle));
        }
    }
}

unsafe fn retain_list_ptr(list: *mut chelis_list) {
    if !list.is_null() {
        (*list).refcount += 1;
    }
}

unsafe fn retain_tuple_ptr(tuple: *mut chelis_tuple) {
    if !tuple.is_null() {
        (*tuple).refcount += 1;
    }
}

unsafe fn retain_dict_ptr(dict: *mut chelis_dict) {
    if !dict.is_null() {
        (*dict).refcount += 1;
    }
}

unsafe fn release_list_ptr(list: *mut chelis_list) {
    if !list.is_null() {
        let inner = &mut *list;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            for value in &inner.items {
                chelis_value_release(*value);
            }
            drop(Box::from_raw(list));
        }
    }
}

unsafe fn release_tuple_ptr(tuple: *mut chelis_tuple) {
    if !tuple.is_null() {
        let inner = &mut *tuple;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            for value in &inner.items {
                chelis_value_release(*value);
            }
            drop(Box::from_raw(tuple));
        }
    }
}

unsafe fn release_dict_ptr(dict: *mut chelis_dict) {
    if !dict.is_null() {
        let inner = &mut *dict;
        inner.refcount -= 1;
        if inner.refcount == 0 {
            for entry in &inner.entries {
                chelis_value_release(entry.key);
                chelis_value_release(entry.value);
            }
            drop(Box::from_raw(dict));
        }
    }
}

fn cstr_to_string(ptr_: *const c_char) -> String {
    if ptr_.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr_) }
            .to_string_lossy()
            .into_owned()
    }
}

fn new_runtime_string(value: String) -> chelis_string {
    let cstring = CString::new(value.clone()).unwrap_or_else(|_| CString::new("").unwrap());
    let inner = Box::new(RuntimeString {
        refcount: 1,
        value,
        cstring,
    });
    chelis_string {
        handle: Box::into_raw(inner),
    }
}

unsafe fn string_value(value: chelis_string) -> &'static RuntimeString {
    if value.handle.is_null() {
        runtime_fail!("null string handle");
    }
    &*value.handle
}

unsafe fn clone_items(items: &[chelis_value]) -> Vec<chelis_value> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        chelis_value_retain(*item);
        out.push(*item);
    }
    out
}

unsafe fn value_key_eq(lhs: chelis_value, rhs: chelis_value) -> bool {
    if lhs.tag != rhs.tag {
        return false;
    }
    match lhs.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => lhs.as_.i64_ == rhs.as_.i64_,
        chelis_value_tag::CHELIS_VALUE_STRING => chelis_string_eq(lhs.as_.string, rhs.as_.string),
        _ => false,
    }
}

unsafe fn dict_find(dict: *const chelis_dict, key: chelis_value) -> Option<usize> {
    if dict.is_null() {
        return None;
    }
    (*dict)
        .entries
        .iter()
        .position(|entry| value_key_eq(entry.key, key))
}

unsafe fn tensor_normalize_axis(tensor: *const chelis_tensor, axis: i64, op: &str) -> usize {
    if tensor.is_null() || axis < 0 || axis >= (*tensor).ndim as i64 {
        runtime_fail!("{op} axis {axis} out of bounds");
    }
    axis as usize
}

unsafe fn tensor_clone(tensor: *const chelis_tensor) -> *mut chelis_tensor {
    let out = chelis_alloc((*tensor).ndim, (*tensor).shape.as_ptr(), (*tensor).dtype);
    ptr::copy_nonoverlapping((*tensor).data, (*out).data, (*tensor).size as usize);
    out
}

unsafe fn require_same_tensor_shape(
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
    op: &str,
) {
    if (*lhs).ndim != (*rhs).ndim {
        runtime_fail!("{op} expects matching tensor rank");
    }
    for axis in 0..(*lhs).ndim as usize {
        if (*lhs).shape[axis] != (*rhs).shape[axis] {
            runtime_fail!("{op} expects matching tensor shape");
        }
    }
}

unsafe fn tensor_scalar_or_same_shape(
    bound: *const chelis_tensor,
    tensor: *const chelis_tensor,
) -> bool {
    if (*bound).ndim == 0 {
        return true;
    }
    if (*bound).ndim != (*tensor).ndim {
        return false;
    }
    for axis in 0..(*tensor).ndim as usize {
        if (*bound).shape[axis] != (*tensor).shape[axis] {
            return false;
        }
    }
    true
}

unsafe fn int_list_value(list: *const chelis_list, index: i64, op: &str) -> i64 {
    if list.is_null()
        || index < 0
        || index >= (*list).items.len() as i64
        || (*list).items[index as usize].tag != chelis_value_tag::CHELIS_VALUE_INT64
    {
        runtime_fail!("{op} expects a list of int64 values");
    }
    (*list).items[index as usize].as_.i64_
}

#[no_mangle]
pub unsafe extern "C" fn chelis_alloc(
    ndim: c_int,
    shape: *const c_int,
    dtype: c_int,
) -> *mut chelis_tensor {
    let mut tensor = Box::new(chelis_tensor {
        data: ptr::null_mut(),
        shape: [0; CHELIS_MAX_DIM],
        strides: [0; CHELIS_MAX_DIM],
        ndim,
        dtype,
        size: 1,
        owns_data: 1,
    });
    if ndim > 0 {
        for d in 0..ndim as usize {
            tensor.shape[d] = *shape.add(d);
            tensor.size *= tensor.shape[d];
        }
        for d in (0..ndim as usize).rev() {
            tensor.strides[d] = if d + 1 == ndim as usize {
                1
            } else {
                tensor.strides[d + 1] * tensor.shape[d + 1]
            };
        }
    }
    if tensor.size == 0 {
        tensor.size = 1;
    }
    let bytes = tensor.size as usize * std::mem::size_of::<f32>();
    tensor.data = libc::calloc(tensor.size as usize, std::mem::size_of::<f32>()) as *mut f32;
    if tensor.data.is_null() && bytes != 0 {
        runtime_fail!("tensor allocation failed");
    }
    Box::into_raw(tensor)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_alloc_view(
    ndim: c_int,
    shape: *const c_int,
    dtype: c_int,
    data: *mut f32,
) -> *mut chelis_tensor {
    let tensor = chelis_alloc(ndim, shape, dtype);
    (*tensor).data = data;
    (*tensor).owns_data = 0;
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_free(t: *mut chelis_tensor) {
    if !t.is_null() {
        if (*t).owns_data != 0 && !(*t).data.is_null() {
            libc::free((*t).data.cast());
        }
        drop(Box::from_raw(t));
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_fill_f32(t: *mut chelis_tensor, val: f32) {
    for i in 0..(*t).size as isize {
        *(*t).data.offset(i) = val;
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_scalar_tensor_from_i64(value: i64) -> *mut chelis_tensor {
    let tensor = chelis_alloc(0, ptr::null(), CHELIS_I32);
    *(*tensor).data = value as f32;
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_scalar_tensor_from_f64(value: f64) -> *mut chelis_tensor {
    let tensor = chelis_alloc(0, ptr::null(), CHELIS_F64);
    *(*tensor).data = value as f32;
    tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_to_f64(t: *const chelis_tensor) -> f64 {
    if t.is_null() || (*t).ndim != 0 {
        runtime_fail!("chelis_tensor_to_f64 expects a rank-0 tensor");
    }
    *(*t).data as f64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_rank(t: *const chelis_tensor) -> i64 {
    if t.is_null() {
        0
    } else {
        (*t).ndim as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_shape(t: *const chelis_tensor, axis: i64) -> i64 {
    if t.is_null() || axis < 0 || axis >= (*t).ndim as i64 {
        runtime_fail!("chelis_tensor_shape axis out of bounds");
    }
    (*t).shape[axis as usize] as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_numel(t: *const chelis_tensor) -> i64 {
    if t.is_null() {
        0
    } else {
        (*t).size as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_cstr(value: *const c_char) -> chelis_string {
    new_runtime_string(cstr_to_string(value))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_data(value: chelis_string) -> *const c_char {
    string_value(value).cstring.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_retain(value: chelis_string) {
    retain_string_handle(value.handle);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_release(value: chelis_string) {
    release_string_handle(value.handle);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_concat(
    lhs: chelis_string,
    rhs: chelis_string,
) -> chelis_string {
    let mut out = string_value(lhs).value.clone();
    out.push_str(&string_value(rhs).value);
    new_runtime_string(out)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_trim(value: chelis_string) -> chelis_string {
    new_runtime_string(string_value(value).value.trim().to_owned())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_slice(
    value: chelis_string,
    start: i64,
    len: i64,
) -> chelis_string {
    if start < 0 || len < 0 {
        return new_runtime_string(String::new());
    }
    let chars: Vec<char> = string_value(value).value.chars().collect();
    if start as usize >= chars.len() {
        return new_runtime_string(String::new());
    }
    let end = ((start + len) as usize).min(chars.len());
    let out: String = chars[start as usize..end].iter().collect();
    new_runtime_string(out)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_eq(lhs: chelis_string, rhs: chelis_string) -> bool {
    string_value(lhs).value == string_value(rhs).value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_contains(
    haystack: chelis_string,
    needle: chelis_string,
) -> bool {
    string_value(haystack)
        .value
        .contains(&string_value(needle).value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_starts_with(
    value: chelis_string,
    prefix: chelis_string,
) -> bool {
    string_value(value)
        .value
        .starts_with(&string_value(prefix).value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_ends_with(
    value: chelis_string,
    suffix: chelis_string,
) -> bool {
    string_value(value)
        .value
        .ends_with(&string_value(suffix).value)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_len(value: chelis_string) -> i64 {
    string_value(value).value.chars().count() as i64
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_int64(value: i64) -> chelis_string {
    new_runtime_string(value.to_string())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_f64(value: f64) -> chelis_string {
    new_runtime_string(format!("{}", value))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_string_from_bool(value: bool) -> chelis_string {
    new_runtime_string(if value { "true" } else { "false" }.to_owned())
}

#[no_mangle]
pub unsafe extern "C" fn chelis_parse_int64(value: chelis_string) -> chelis_option_i64 {
    match string_value(value).value.trim().parse::<i64>() {
        Ok(parsed) => chelis_option_i64 {
            is_some: true,
            value: parsed,
        },
        Err(_) => chelis_option_i64 {
            is_some: false,
            value: 0,
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_parse_f64(value: chelis_string) -> chelis_option_f64 {
    match string_value(value).value.trim().parse::<f64>() {
        Ok(parsed) => chelis_option_f64 {
            is_some: true,
            value: parsed,
        },
        Err(_) => chelis_option_f64 {
            is_some: false,
            value: 0.0,
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_retain(list: *const chelis_list) {
    retain_list_ptr(list as *mut chelis_list);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_release(list: *const chelis_list) {
    release_list_ptr(list as *mut chelis_list);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_len(list: *const chelis_list) -> i64 {
    if list.is_null() {
        0
    } else {
        (*list).items.len() as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_retain(tuple: *const chelis_tuple) {
    retain_tuple_ptr(tuple as *mut chelis_tuple);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_release(tuple: *const chelis_tuple) {
    release_tuple_ptr(tuple as *mut chelis_tuple);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_len(tuple: *const chelis_tuple) -> i64 {
    if tuple.is_null() {
        0
    } else {
        (*tuple).items.len() as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_retain(dict: *const chelis_dict) {
    retain_dict_ptr(dict as *mut chelis_dict);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_release(dict: *const chelis_dict) {
    release_dict_ptr(dict as *mut chelis_dict);
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_len(dict: *const chelis_dict) -> i64 {
    if dict.is_null() {
        0
    } else {
        (*dict).entries.len() as i64
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_int64(value: i64) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_INT64,
        as_: chelis_value_union { i64_: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_f64(value: f64) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_FLOAT64,
        as_: chelis_value_union { f64_: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_bool(value: bool) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_BOOL,
        as_: chelis_value_union { boolean: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_string(value: chelis_string) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_STRING,
        as_: chelis_value_union { string: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_tensor(value: *mut chelis_tensor) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_TENSOR,
        as_: chelis_value_union { tensor: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_list(value: *mut chelis_list) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_LIST,
        as_: chelis_value_union { list: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_tuple(value: *mut chelis_tuple) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_TUPLE,
        as_: chelis_value_union { tuple: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_from_dict(value: *mut chelis_dict) -> chelis_value {
    chelis_value {
        tag: chelis_value_tag::CHELIS_VALUE_DICT,
        as_: chelis_value_union { dict: value },
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_retain(value: chelis_value) {
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_STRING => retain_string_handle(value.as_.string.handle),
        chelis_value_tag::CHELIS_VALUE_LIST => retain_list_ptr(value.as_.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => retain_tuple_ptr(value.as_.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => retain_dict_ptr(value.as_.dict),
        _ => {}
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_release(value: chelis_value) {
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_STRING => release_string_handle(value.as_.string.handle),
        chelis_value_tag::CHELIS_VALUE_LIST => release_list_ptr(value.as_.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => release_tuple_ptr(value.as_.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => release_dict_ptr(value.as_.dict),
        _ => {}
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_int64(value: chelis_value) -> i64 {
    if value.tag != chelis_value_tag::CHELIS_VALUE_INT64 {
        runtime_fail!("expected int64 value");
    }
    value.as_.i64_
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_f64(value: chelis_value) -> f64 {
    if value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        return value.as_.i64_ as f64;
    }
    if value.tag != chelis_value_tag::CHELIS_VALUE_FLOAT64 {
        runtime_fail!("expected float64 value");
    }
    value.as_.f64_
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_bool(value: chelis_value) -> bool {
    if value.tag != chelis_value_tag::CHELIS_VALUE_BOOL {
        runtime_fail!("expected bool value");
    }
    value.as_.boolean
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_string(value: chelis_value) -> chelis_string {
    if value.tag != chelis_value_tag::CHELIS_VALUE_STRING {
        runtime_fail!("expected string value");
    }
    value.as_.string
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_tensor(value: chelis_value) -> *mut chelis_tensor {
    if value.tag != chelis_value_tag::CHELIS_VALUE_TENSOR {
        runtime_fail!("expected tensor value");
    }
    value.as_.tensor
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_list(value: chelis_value) -> *mut chelis_list {
    if value.tag != chelis_value_tag::CHELIS_VALUE_LIST {
        runtime_fail!("expected list value");
    }
    value.as_.list
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_tuple(value: chelis_value) -> *mut chelis_tuple {
    if value.tag != chelis_value_tag::CHELIS_VALUE_TUPLE {
        runtime_fail!("expected tuple value");
    }
    value.as_.tuple
}

#[no_mangle]
pub unsafe extern "C" fn chelis_value_as_dict(value: chelis_value) -> *mut chelis_dict {
    if value.tag != chelis_value_tag::CHELIS_VALUE_DICT {
        runtime_fail!("expected dict value");
    }
    value.as_.dict
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_empty() -> *mut chelis_list {
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: Vec::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_from_values(
    items: *const chelis_value,
    len: i64,
) -> *mut chelis_list {
    let slice = if items.is_null() || len <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(items, len as usize)
    };
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: clone_items(slice),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_index(list: *const chelis_list, index: i64) -> chelis_value {
    if list.is_null() || index < 0 || index >= (*list).items.len() as i64 {
        runtime_fail!("list index out of bounds");
    }
    let value = (*list).items[index as usize];
    chelis_value_retain(value);
    value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_append(
    list: *const chelis_list,
    value: chelis_value,
) -> *mut chelis_list {
    let mut items = if list.is_null() {
        Vec::new()
    } else {
        clone_items(&(*list).items)
    };
    chelis_value_retain(value);
    items.push(value);
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_concat(
    lhs: *const chelis_list,
    rhs: *const chelis_list,
) -> *mut chelis_list {
    let mut items = if lhs.is_null() {
        Vec::new()
    } else {
        clone_items(&(*lhs).items)
    };
    if !rhs.is_null() {
        for item in &(*rhs).items {
            chelis_value_retain(*item);
            items.push(*item);
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_take(
    list: *const chelis_list,
    count: i64,
) -> *mut chelis_list {
    if count < 0 {
        runtime_fail!("take requires non-negative count");
    }
    let items = if list.is_null() {
        Vec::new()
    } else {
        clone_items(&(*list).items[..(*list).items.len().min(count as usize)])
    };
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_drop(
    list: *const chelis_list,
    count: i64,
) -> *mut chelis_list {
    if count < 0 {
        runtime_fail!("drop requires non-negative count");
    }
    if list.is_null() || count as usize >= (*list).items.len() {
        return chelis_list_empty();
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: clone_items(&(*list).items[count as usize..]),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_chunk(
    list: *const chelis_list,
    size: i64,
) -> *mut chelis_list {
    if size <= 0 {
        runtime_fail!("chunk requires positive size");
    }
    let mut outer = Vec::new();
    let items = if list.is_null() {
        &[][..]
    } else {
        &(*list).items[..]
    };
    for chunk in items.chunks(size as usize) {
        let inner = Box::into_raw(Box::new(chelis_list {
            refcount: 1,
            items: clone_items(chunk),
        }));
        outer.push(chelis_value_from_list(inner));
    }
    let out = Box::new(chelis_list {
        refcount: 1,
        items: outer,
    });
    Box::into_raw(out)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_flatten(list: *const chelis_list) -> *mut chelis_list {
    let mut out = Vec::new();
    if !list.is_null() {
        for item in &(*list).items {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("flatten expects nested list input");
            }
            let inner = item.as_.list;
            if !inner.is_null() {
                for value in &(*inner).items {
                    chelis_value_retain(*value);
                    out.push(*value);
                }
            }
        }
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: out,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_range_i64(start: i64, end: i64) -> *mut chelis_list {
    let mut items = Vec::new();
    for value in start..end {
        items.push(chelis_value_from_int64(value));
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_from_values(
    items: *const chelis_value,
    len: i64,
) -> *mut chelis_tuple {
    let slice = if items.is_null() || len <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(items, len as usize)
    };
    Box::into_raw(Box::new(chelis_tuple {
        refcount: 1,
        items: clone_items(slice),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tuple_get(tuple: *const chelis_tuple, index: i64) -> chelis_value {
    if tuple.is_null() || index < 0 || index >= (*tuple).items.len() as i64 {
        runtime_fail!("tuple index out of bounds");
    }
    let value = (*tuple).items[index as usize];
    chelis_value_retain(value);
    value
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_zip(
    lhs: *const chelis_list,
    rhs: *const chelis_list,
) -> *mut chelis_list {
    let lhs_items = if lhs.is_null() {
        &[][..]
    } else {
        &(*lhs).items[..]
    };
    let rhs_items = if rhs.is_null() {
        &[][..]
    } else {
        &(*rhs).items[..]
    };
    let mut out = Vec::new();
    for (left, right) in lhs_items.iter().zip(rhs_items.iter()) {
        let pair = [*left, *right];
        let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
        out.push(chelis_value_from_tuple(tuple));
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: out,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_enumerate(list: *const chelis_list) -> *mut chelis_list {
    let mut out = Vec::new();
    if !list.is_null() {
        for (index, item) in (*list).items.iter().enumerate() {
            let pair = [chelis_value_from_int64(index as i64), *item];
            let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
            out.push(chelis_value_from_tuple(tuple));
        }
    }
    Box::into_raw(Box::new(chelis_list {
        refcount: 1,
        items: out,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_from_pairs(pairs: *const chelis_list) -> *mut chelis_dict {
    let mut entries: Vec<chelis_dict_entry> = Vec::new();
    if !pairs.is_null() {
        for pair in &(*pairs).items {
            if pair.tag != chelis_value_tag::CHELIS_VALUE_TUPLE {
                runtime_fail!("dict_of expects list entries to be tuples");
            }
            let entry = pair.as_.tuple;
            if entry.is_null() || (*entry).items.len() != 2 {
                runtime_fail!("dict_of expects 2-tuples");
            }
            let key = (*entry).items[0];
            if !matches!(
                key.tag,
                chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
            ) {
                runtime_fail!("dict_of keys must be int64 or string");
            }
            let value = (*entry).items[1];
            if let Some(pos) = entries
                .iter()
                .position(|existing| value_key_eq(existing.key, key))
            {
                chelis_value_release(entries[pos].value);
                chelis_value_retain(value);
                entries[pos].value = value;
            } else {
                chelis_value_retain(key);
                chelis_value_retain(value);
                entries.push(chelis_dict_entry { key, value });
            }
        }
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_contains(dict: *const chelis_dict, key: chelis_value) -> bool {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    dict_find(dict, key).is_some()
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get(
    dict: *const chelis_dict,
    key: chelis_value,
) -> chelis_option_value {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    if let Some(index) = dict_find(dict, key) {
        let value = (*dict).entries[index].value;
        chelis_value_retain(value);
        chelis_option_value {
            is_some: true,
            value,
        }
    } else {
        chelis_option_value {
            is_some: false,
            value: chelis_value_from_int64(0),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get_i64(
    dict: *const chelis_dict,
    key: chelis_value,
) -> chelis_option_i64 {
    let value = chelis_dict_get(dict, key);
    if !value.is_some {
        return chelis_option_i64 {
            is_some: false,
            value: 0,
        };
    }
    let out = match value.value.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => chelis_option_i64 {
            is_some: true,
            value: value.value.as_.i64_,
        },
        _ => runtime_fail!("dict value is not int64"),
    };
    chelis_value_release(value.value);
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_get_f64(
    dict: *const chelis_dict,
    key: chelis_value,
) -> chelis_option_f64 {
    let value = chelis_dict_get(dict, key);
    if !value.is_some {
        return chelis_option_f64 {
            is_some: false,
            value: 0.0,
        };
    }
    let out = match value.value.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => chelis_option_f64 {
            is_some: true,
            value: value.value.as_.i64_ as f64,
        },
        chelis_value_tag::CHELIS_VALUE_FLOAT64 => chelis_option_f64 {
            is_some: true,
            value: value.value.as_.f64_,
        },
        _ => runtime_fail!("dict value is not float64"),
    };
    chelis_value_release(value.value);
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_remove(
    dict: *const chelis_dict,
    key: chelis_value,
) -> *mut chelis_dict {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    let mut entries = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            if !value_key_eq(entry.key, key) {
                chelis_value_retain(entry.key);
                chelis_value_retain(entry.value);
                entries.push(*entry);
            }
        }
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_insert(
    dict: *const chelis_dict,
    key: chelis_value,
    value: chelis_value,
) -> *mut chelis_dict {
    if !matches!(
        key.tag,
        chelis_value_tag::CHELIS_VALUE_INT64 | chelis_value_tag::CHELIS_VALUE_STRING
    ) {
        runtime_fail!("dict key must be int64 or string");
    }
    let mut entries = if dict.is_null() {
        Vec::new()
    } else {
        let mut out = Vec::with_capacity((*dict).entries.len() + 1);
        for entry in &(*dict).entries {
            chelis_value_retain(entry.key);
            chelis_value_retain(entry.value);
            out.push(*entry);
        }
        out
    };
    if let Some(index) = entries
        .iter()
        .position(|entry| value_key_eq(entry.key, key))
    {
        chelis_value_release(entries[index].value);
        chelis_value_retain(value);
        entries[index].value = value;
    } else {
        chelis_value_retain(key);
        chelis_value_retain(value);
        entries.push(chelis_dict_entry { key, value });
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_merge(
    lhs: *const chelis_dict,
    rhs: *const chelis_dict,
) -> *mut chelis_dict {
    let mut entries = Vec::new();
    if !lhs.is_null() {
        for entry in &(*lhs).entries {
            chelis_value_retain(entry.key);
            chelis_value_retain(entry.value);
            entries.push(*entry);
        }
    }
    if !rhs.is_null() {
        for entry in &(*rhs).entries {
            if let Some(index) = entries
                .iter()
                .position(|existing| value_key_eq(existing.key, entry.key))
            {
                chelis_value_release(entries[index].value);
                chelis_value_retain(entry.value);
                entries[index].value = entry.value;
            } else {
                chelis_value_retain(entry.key);
                chelis_value_retain(entry.value);
                entries.push(*entry);
            }
        }
    }
    Box::into_raw(Box::new(chelis_dict {
        refcount: 1,
        entries,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_keys(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            chelis_value_retain(entry.key);
            items.push(entry.key);
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_values(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            chelis_value_retain(entry.value);
            items.push(entry.value);
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_dict_entries(dict: *const chelis_dict) -> *mut chelis_list {
    let mut items = Vec::new();
    if !dict.is_null() {
        for entry in &(*dict).entries {
            let pair = [entry.key, entry.value];
            let tuple = chelis_tuple_from_values(pair.as_ptr(), 2);
            items.push(chelis_value_from_tuple(tuple));
        }
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_from_value_list(
    list: *const chelis_list,
) -> *mut chelis_tensor {
    let len = chelis_list_len(list) as c_int;
    let shape = [len];
    let dtype = if !list.is_null()
        && !(*list).items.is_empty()
        && (*list).items[0].tag == chelis_value_tag::CHELIS_VALUE_INT64
    {
        CHELIS_I32
    } else {
        CHELIS_F64
    };
    let out = chelis_alloc(1, shape.as_ptr(), dtype);
    if !list.is_null() {
        for (i, item) in (*list).items.iter().enumerate() {
            *(*out).data.add(i) = match item.tag {
                chelis_value_tag::CHELIS_VALUE_INT64 => item.as_.i64_ as f32,
                chelis_value_tag::CHELIS_VALUE_FLOAT64 => item.as_.f64_ as f32,
                _ => runtime_fail!("to_tensor expects numeric list elements"),
            };
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_list_from_tensor(tensor: *const chelis_tensor) -> *mut chelis_list {
    if tensor.is_null() {
        return chelis_list_empty();
    }
    if (*tensor).ndim != 1 {
        runtime_fail!("to_list expects a rank-1 tensor");
    }
    let mut items = Vec::with_capacity((*tensor).shape[0] as usize);
    for i in 0..(*tensor).shape[0] as usize {
        let raw = *(*tensor).data.add(i * (*tensor).strides[0] as usize);
        let value = match (*tensor).dtype {
            CHELIS_BOOL => chelis_value_from_bool(raw != 0.0),
            CHELIS_I32 => chelis_value_from_int64(raw as i64),
            CHELIS_F32 | CHELIS_F64 => chelis_value_from_f64(raw as f64),
            _ => runtime_fail!("to_list expects numeric or bool tensor input"),
        };
        items.push(value);
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_pad_sequences(
    sequences: *const chelis_list,
    pad_value: chelis_value,
) -> *mut chelis_tensor {
    let batch = chelis_list_len(sequences) as usize;
    let mut width = 0usize;
    if !sequences.is_null() {
        for item in &(*sequences).items {
            if item.tag != chelis_value_tag::CHELIS_VALUE_LIST {
                runtime_fail!("pad_sequences expects nested lists");
            }
            width = width.max((*item.as_.list).items.len());
        }
    }
    let shape = [batch as c_int, width as c_int];
    let dtype = if pad_value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        CHELIS_I32
    } else {
        CHELIS_F64
    };
    let out = chelis_alloc(2, shape.as_ptr(), dtype);
    let pad = if pad_value.tag == chelis_value_tag::CHELIS_VALUE_INT64 {
        pad_value.as_.i64_ as f64
    } else {
        pad_value.as_.f64_
    };
    if !sequences.is_null() {
        for (row, item) in (*sequences).items.iter().enumerate() {
            let seq = item.as_.list;
            for col in 0..width {
                let flat = row * width + col;
                let value = if col < (*seq).items.len() {
                    match (*seq).items[col].tag {
                        chelis_value_tag::CHELIS_VALUE_INT64 => (*seq).items[col].as_.i64_ as f64,
                        chelis_value_tag::CHELIS_VALUE_FLOAT64 => (*seq).items[col].as_.f64_,
                        _ => runtime_fail!("pad_sequences expects numeric nested lists"),
                    }
                } else {
                    pad
                };
                *(*out).data.add(flat) = value as f32;
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_concat(
    parts: *const chelis_list,
    axis: i64,
) -> *mut chelis_tensor {
    if parts.is_null() || (*parts).items.is_empty() {
        runtime_fail!("concat expects at least one tensor part");
    }
    let first = chelis_value_as_tensor((*parts).items[0]);
    let axis_i = tensor_normalize_axis(first, axis, "concat");
    let mut out_shape = (*first).shape;
    out_shape[axis_i] = 0;
    for item in &(*parts).items {
        let tensor = chelis_value_as_tensor(*item);
        if (*tensor).ndim != (*first).ndim || (*tensor).dtype != (*first).dtype {
            runtime_fail!("concat expects matching tensor rank and dtype");
        }
        for axis2 in 0..(*tensor).ndim as usize {
            if axis2 != axis_i && (*tensor).shape[axis2] != (*first).shape[axis2] {
                runtime_fail!("concat expects matching non-concatenated axes");
            }
        }
        out_shape[axis_i] += (*tensor).shape[axis_i];
    }
    let out = chelis_alloc((*first).ndim, out_shape.as_ptr(), (*first).dtype);
    let mut axis_offset = 0;
    let mut indices = [0; CHELIS_MAX_DIM];
    for item in &(*parts).items {
        let tensor = chelis_value_as_tensor(*item);
        for linear in 0..(*tensor).size {
            chelis_flat_to_indices(
                linear,
                (*tensor).shape.as_ptr(),
                (*tensor).ndim,
                indices.as_mut_ptr(),
            );
            indices[axis_i] += axis_offset;
            let out_linear =
                chelis_indices_to_flat(indices.as_ptr(), (*out).strides.as_ptr(), (*out).ndim);
            *(*out).data.add(out_linear as usize) = *(*tensor).data.add(linear as usize);
            indices[axis_i] -= axis_offset;
        }
        axis_offset += (*tensor).shape[axis_i];
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_split(
    tensor: *const chelis_tensor,
    axis: i64,
    sizes: *const chelis_list,
) -> *mut chelis_list {
    let axis_i = tensor_normalize_axis(tensor, axis, "split");
    let mut total = 0i64;
    for i in 0..chelis_list_len(sizes) {
        total += int_list_value(sizes, i, "split");
    }
    if total != (*tensor).shape[axis_i] as i64 {
        runtime_fail!("split sizes must sum to the selected axis extent");
    }
    let mut items = Vec::new();
    let mut axis_offset = 0;
    let mut indices = [0; CHELIS_MAX_DIM];
    for part_idx in 0..chelis_list_len(sizes) {
        let part_size = int_list_value(sizes, part_idx, "split") as c_int;
        let mut shape = (*tensor).shape;
        shape[axis_i] = part_size;
        let part = chelis_alloc((*tensor).ndim, shape.as_ptr(), (*tensor).dtype);
        for linear in 0..(*part).size {
            chelis_flat_to_indices(
                linear,
                (*part).shape.as_ptr(),
                (*part).ndim,
                indices.as_mut_ptr(),
            );
            indices[axis_i] += axis_offset;
            let src = chelis_indices_to_flat(
                indices.as_ptr(),
                (*tensor).strides.as_ptr(),
                (*tensor).ndim,
            );
            *(*part).data.add(linear as usize) = *(*tensor).data.add(src as usize);
            indices[axis_i] -= axis_offset;
        }
        axis_offset += part_size;
        items.push(chelis_value_from_tensor(part));
    }
    Box::into_raw(Box::new(chelis_list { refcount: 1, items }))
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_gather(
    tensor: *const chelis_tensor,
    indices: *const chelis_tensor,
    axis: i64,
) -> *mut chelis_tensor {
    let axis_i = tensor_normalize_axis(tensor, axis, "gather");
    let out_ndim = (*tensor).ndim as usize - 1 + (*indices).ndim as usize;
    let mut out_shape = [0; CHELIS_MAX_DIM];
    let mut pos = 0usize;
    for i in 0..axis_i {
        out_shape[pos] = (*tensor).shape[i];
        pos += 1;
    }
    for i in 0..(*indices).ndim as usize {
        out_shape[pos] = (*indices).shape[i];
        pos += 1;
    }
    for i in axis_i + 1..(*tensor).ndim as usize {
        out_shape[pos] = (*tensor).shape[i];
        pos += 1;
    }
    let out = chelis_alloc(out_ndim as c_int, out_shape.as_ptr(), (*tensor).dtype);
    let mut out_index = [0; CHELIS_MAX_DIM];
    let mut src_index = [0; CHELIS_MAX_DIM];
    let mut gather_index = [0; CHELIS_MAX_DIM];
    for linear in 0..(*out).size {
        chelis_flat_to_indices(
            linear,
            (*out).shape.as_ptr(),
            (*out).ndim,
            out_index.as_mut_ptr(),
        );
        let mut src_pos = 0usize;
        for &val in &out_index[..axis_i] {
            src_index[src_pos] = val;
            src_pos += 1;
        }
        gather_index[..(*indices).ndim as usize]
            .copy_from_slice(&out_index[axis_i..((*indices).ndim as usize + axis_i)]);
        let index_linear = chelis_indices_to_flat(
            gather_index.as_ptr(),
            (*indices).strides.as_ptr(),
            (*indices).ndim,
        );
        let gathered = *(*indices).data.add(index_linear as usize) as i64;
        if gathered < 0 || gathered >= (*tensor).shape[axis_i] as i64 {
            runtime_fail!("gather index {gathered} out of bounds");
        }
        src_index[src_pos] = gathered as c_int;
        src_pos += 1;
        for i in axis_i + 1..(*tensor).ndim as usize {
            src_index[src_pos] = out_index[axis_i + (*indices).ndim as usize + (i - axis_i - 1)];
            src_pos += 1;
        }
        let src_linear = chelis_indices_to_flat(
            src_index.as_ptr(),
            (*tensor).strides.as_ptr(),
            (*tensor).ndim,
        );
        *(*out).data.add(linear as usize) = *(*tensor).data.add(src_linear as usize);
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_cmplt(
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
) -> *mut chelis_tensor {
    require_same_tensor_shape(lhs, rhs, "cmplt");
    let out = chelis_alloc((*lhs).ndim, (*lhs).shape.as_ptr(), CHELIS_BOOL);
    for i in 0..(*out).size as usize {
        *(*out).data.add(i) = if *(*lhs).data.add(i) < *(*rhs).data.add(i) {
            1.0
        } else {
            0.0
        };
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_scatter(
    base: *const chelis_tensor,
    indices: *const chelis_tensor,
    updates: *const chelis_tensor,
    axis: i64,
    mode: chelis_string,
) -> *mut chelis_tensor {
    let axis_i = tensor_normalize_axis(base, axis, "scatter");
    let expected = chelis_tensor_gather(base, indices, axis);
    require_same_tensor_shape(expected, updates, "scatter");
    chelis_free(expected);
    let out = tensor_clone(base);
    let mode_str = string_value(mode).value.clone();
    let replace_mode = mode_str == "replace";
    let add_mode = mode_str == "add";
    if !replace_mode && !add_mode {
        runtime_fail!("scatter mode must be replace or add");
    }
    let mut seen = if replace_mode {
        vec![false; (*out).size as usize]
    } else {
        Vec::new()
    };
    let mut update_index = [0; CHELIS_MAX_DIM];
    let mut out_index = [0; CHELIS_MAX_DIM];
    let mut gather_index = [0; CHELIS_MAX_DIM];
    for linear in 0..(*updates).size {
        chelis_flat_to_indices(
            linear,
            (*updates).shape.as_ptr(),
            (*updates).ndim,
            update_index.as_mut_ptr(),
        );
        let mut out_pos = 0usize;
        for &val in &update_index[..axis_i] {
            out_index[out_pos] = val;
            out_pos += 1;
        }
        gather_index[..(*indices).ndim as usize]
            .copy_from_slice(&update_index[axis_i..((*indices).ndim as usize + axis_i)]);
        let index_linear = chelis_indices_to_flat(
            gather_index.as_ptr(),
            (*indices).strides.as_ptr(),
            (*indices).ndim,
        );
        let gathered = *(*indices).data.add(index_linear as usize) as i64;
        if gathered < 0 || gathered >= (*base).shape[axis_i] as i64 {
            runtime_fail!("scatter index {gathered} out of bounds");
        }
        out_index[out_pos] = gathered as c_int;
        out_pos += 1;
        for i in axis_i + 1..(*base).ndim as usize {
            out_index[out_pos] = update_index[axis_i + (*indices).ndim as usize + (i - axis_i - 1)];
            out_pos += 1;
        }
        let out_linear =
            chelis_indices_to_flat(out_index.as_ptr(), (*out).strides.as_ptr(), (*out).ndim)
                as usize;
        if replace_mode {
            if seen[out_linear] {
                runtime_fail!("scatter replace mode rejects duplicate target index {out_linear}");
            }
            seen[out_linear] = true;
            *(*out).data.add(out_linear) = *(*updates).data.add(linear as usize);
        } else {
            *(*out).data.add(out_linear) += *(*updates).data.add(linear as usize);
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_where(
    cond: *const chelis_tensor,
    then_tensor: *const chelis_tensor,
    else_tensor: *const chelis_tensor,
) -> *mut chelis_tensor {
    require_same_tensor_shape(cond, then_tensor, "where");
    require_same_tensor_shape(then_tensor, else_tensor, "where");
    let out = chelis_alloc(
        (*then_tensor).ndim,
        (*then_tensor).shape.as_ptr(),
        (*then_tensor).dtype,
    );
    for i in 0..(*out).size as usize {
        *(*out).data.add(i) = if *(*cond).data.add(i) != 0.0 {
            *(*then_tensor).data.add(i)
        } else {
            *(*else_tensor).data.add(i)
        };
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_cumsum(
    tensor: *const chelis_tensor,
    axis: i64,
) -> *mut chelis_tensor {
    let axis_i = tensor_normalize_axis(tensor, axis, "cumsum");
    let out = tensor_clone(tensor);
    let axis_size = (*tensor).shape[axis_i] as usize;
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in axis_i + 1..(*tensor).ndim as usize {
        inner *= (*tensor).shape[i] as usize;
    }
    for i in 0..axis_i {
        outer *= (*tensor).shape[i] as usize;
    }
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut running = 0.0f32;
            for axis_idx in 0..axis_size {
                let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                running += *(*out).data.add(linear);
                *(*out).data.add(linear) = running;
            }
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_sort(
    tensor: *const chelis_tensor,
    axis: i64,
) -> *mut chelis_tuple {
    let axis_i = tensor_normalize_axis(tensor, axis, "sort");
    let values = tensor_clone(tensor);
    let indices = chelis_alloc((*tensor).ndim, (*tensor).shape.as_ptr(), CHELIS_I32);
    let axis_size = (*tensor).shape[axis_i] as usize;
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in axis_i + 1..(*tensor).ndim as usize {
        inner *= (*tensor).shape[i] as usize;
    }
    for i in 0..axis_i {
        outer *= (*tensor).shape[i] as usize;
    }
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            for i in 0..axis_size {
                let linear = (outer_idx * axis_size + i) * inner + inner_idx;
                *(*indices).data.add(linear) = i as f32;
            }
            for i in 1..axis_size {
                let mut j = i;
                while j > 0 {
                    let left = (outer_idx * axis_size + (j - 1)) * inner + inner_idx;
                    let right = (outer_idx * axis_size + j) * inner + inner_idx;
                    if *(*values).data.add(left) <= *(*values).data.add(right) {
                        break;
                    }
                    let tmpv = *(*values).data.add(left);
                    *(*values).data.add(left) = *(*values).data.add(right);
                    *(*values).data.add(right) = tmpv;
                    let tmpi = *(*indices).data.add(left);
                    *(*indices).data.add(left) = *(*indices).data.add(right);
                    *(*indices).data.add(right) = tmpi;
                    j -= 1;
                }
            }
        }
    }
    let items = [
        chelis_value_from_tensor(values),
        chelis_value_from_tensor(indices),
    ];
    chelis_tuple_from_values(items.as_ptr(), 2)
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_diagonal(
    tensor: *const chelis_tensor,
    axis1: i64,
    axis2: i64,
) -> *mut chelis_tensor {
    let axis1_i = tensor_normalize_axis(tensor, axis1, "diagonal");
    let axis2_i = tensor_normalize_axis(tensor, axis2, "diagonal");
    if axis1_i == axis2_i {
        runtime_fail!("diagonal expects distinct axes");
    }
    let diag = (*tensor).shape[axis1_i].min((*tensor).shape[axis2_i]);
    let mut out_shape = [0; CHELIS_MAX_DIM];
    let mut pos = 0usize;
    for i in 0..(*tensor).ndim as usize {
        if i == axis1_i {
            out_shape[pos] = diag;
            pos += 1;
        } else if i != axis2_i {
            out_shape[pos] = (*tensor).shape[i];
            pos += 1;
        }
    }
    let out = chelis_alloc((*tensor).ndim - 1, out_shape.as_ptr(), (*tensor).dtype);
    let mut out_index = [0; CHELIS_MAX_DIM];
    let mut src_index = [0; CHELIS_MAX_DIM];
    for linear in 0..(*out).size {
        chelis_flat_to_indices(
            linear,
            (*out).shape.as_ptr(),
            (*out).ndim,
            out_index.as_mut_ptr(),
        );
        let diag_idx = out_index[axis1_i];
        let mut out_pos = 0usize;
        for (i, src_slot) in src_index
            .iter_mut()
            .enumerate()
            .take((*tensor).ndim as usize)
        {
            if i == axis1_i || i == axis2_i {
                *src_slot = diag_idx;
            } else {
                *src_slot = out_index[out_pos];
                out_pos += 1;
            }
        }
        let src = chelis_indices_to_flat(
            src_index.as_ptr(),
            (*tensor).strides.as_ptr(),
            (*tensor).ndim,
        );
        *(*out).data.add(linear as usize) = *(*tensor).data.add(src as usize);
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_trace(
    tensor: *const chelis_tensor,
    axis1: i64,
    axis2: i64,
) -> *mut chelis_tensor {
    let diag = chelis_tensor_diagonal(tensor, axis1, axis2);
    let reduce_axis = axis1.min(axis2) as usize;
    let axis_size = (*diag).shape[reduce_axis] as usize;
    let mut out_shape = [0; CHELIS_MAX_DIM];
    let mut pos = 0usize;
    for i in 0..(*diag).ndim as usize {
        if i != reduce_axis {
            out_shape[pos] = (*diag).shape[i];
            pos += 1;
        }
    }
    let out = chelis_alloc((*diag).ndim - 1, out_shape.as_ptr(), (*diag).dtype);
    let mut inner = 1usize;
    let mut outer = 1usize;
    for i in reduce_axis + 1..(*diag).ndim as usize {
        inner *= (*diag).shape[i] as usize;
    }
    for i in 0..reduce_axis {
        outer *= (*diag).shape[i] as usize;
    }
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut sum = 0.0f32;
            for axis_idx in 0..axis_size {
                let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                sum += *(*diag).data.add(linear);
            }
            *(*out).data.add(outer_idx * inner + inner_idx) = sum;
        }
    }
    chelis_free(diag);
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_clamp(
    tensor: *const chelis_tensor,
    lo: *const chelis_tensor,
    hi: *const chelis_tensor,
) -> *mut chelis_tensor {
    if !tensor_scalar_or_same_shape(lo, tensor) || !tensor_scalar_or_same_shape(hi, tensor) {
        runtime_fail!("clamp expects scalar bounds or matching-shape tensor bounds");
    }
    let out = chelis_alloc((*tensor).ndim, (*tensor).shape.as_ptr(), (*tensor).dtype);
    for i in 0..(*out).size as usize {
        let low = if (*lo).ndim == 0 {
            *(*lo).data
        } else {
            *(*lo).data.add(i)
        };
        let high = if (*hi).ndim == 0 {
            *(*hi).data
        } else {
            *(*hi).data.add(i)
        };
        let mut value = *(*tensor).data.add(i);
        value = value.max(low).min(high);
        *(*out).data.add(i) = value;
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_einsum(
    equation: chelis_string,
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
) -> *mut chelis_tensor {
    let equation = &string_value(equation).value;
    if equation.contains("...") {
        runtime_fail!("einsum ellipsis support is deferred in 3h");
    }
    if (*lhs).dtype != (*rhs).dtype {
        runtime_fail!("einsum expects matching tensor dtype");
    }
    let (input, output) = equation
        .split_once("->")
        .unwrap_or_else(|| runtime_fail!("einsum equation must contain explicit output"));
    let mut ops = input.split(',');
    let lhs_labels = ops.next().unwrap_or("");
    let rhs_labels = ops.next().unwrap_or("");
    if ops.next().is_some() || lhs_labels.is_empty() || rhs_labels.is_empty() {
        runtime_fail!("einsum 3h currently supports exactly two operands");
    }
    let out_labels: Vec<char> = output.chars().collect();
    let lhs_chars: Vec<char> = lhs_labels.chars().collect();
    let rhs_chars: Vec<char> = rhs_labels.chars().collect();
    if lhs_chars.len() != (*lhs).ndim as usize || rhs_chars.len() != (*rhs).ndim as usize {
        runtime_fail!("einsum label count must match operand rank");
    }
    let mut label_dims = [-1i32; 256];
    let mut label_values = [0i32; 256];
    let mut out_contains = [false; 256];
    let mut reduction_seen = [false; 256];
    for &label in &out_labels {
        out_contains[label as usize] = true;
    }
    for (i, &label) in lhs_chars.iter().enumerate() {
        let idx = label as usize;
        if label_dims[idx] >= 0 && label_dims[idx] != (*lhs).shape[i] {
            runtime_fail!("einsum label `{label}` has inconsistent extents");
        }
        label_dims[idx] = (*lhs).shape[i];
    }
    for (i, &label) in rhs_chars.iter().enumerate() {
        let idx = label as usize;
        if label_dims[idx] >= 0 && label_dims[idx] != (*rhs).shape[i] {
            runtime_fail!("einsum label `{label}` has inconsistent extents");
        }
        label_dims[idx] = (*rhs).shape[i];
    }
    let mut out_shape = [0; CHELIS_MAX_DIM];
    for (i, &label) in out_labels.iter().enumerate() {
        let idx = label as usize;
        if label_dims[idx] < 0 {
            runtime_fail!("einsum output label `{label}` missing from inputs");
        }
        out_shape[i] = label_dims[idx];
    }
    let mut reduction_labels: Vec<char> = Vec::new();
    let mut reduction_shape: Vec<i32> = Vec::new();
    for &label in lhs_chars.iter().chain(rhs_chars.iter()) {
        let idx = label as usize;
        if !out_contains[idx] && !reduction_seen[idx] {
            reduction_seen[idx] = true;
            reduction_labels.push(label);
            reduction_shape.push(label_dims[idx]);
        }
    }
    let out = chelis_alloc(out_labels.len() as c_int, out_shape.as_ptr(), (*lhs).dtype);
    let reduction_total = reduction_shape
        .iter()
        .fold(1usize, |acc, dim| acc * (*dim as usize));
    let mut out_index = [0; CHELIS_MAX_DIM];
    let mut reduction_index = [0; CHELIS_MAX_DIM];
    let mut lhs_index = [0; CHELIS_MAX_DIM];
    let mut rhs_index = [0; CHELIS_MAX_DIM];
    for out_linear in 0..(*out).size as usize {
        if !out_labels.is_empty() {
            chelis_flat_to_indices(
                out_linear as c_int,
                out_shape.as_ptr(),
                out_labels.len() as c_int,
                out_index.as_mut_ptr(),
            );
        }
        for (i, &label) in out_labels.iter().enumerate() {
            label_values[label as usize] = out_index[i];
        }
        let mut acc = 0.0f32;
        for reduction_linear in 0..reduction_total {
            if !reduction_labels.is_empty() {
                chelis_flat_to_indices(
                    reduction_linear as c_int,
                    reduction_shape.as_ptr(),
                    reduction_shape.len() as c_int,
                    reduction_index.as_mut_ptr(),
                );
            }
            for (i, &label) in reduction_labels.iter().enumerate() {
                label_values[label as usize] = reduction_index[i];
            }
            for (i, &label) in lhs_chars.iter().enumerate() {
                lhs_index[i] = label_values[label as usize];
            }
            for (i, &label) in rhs_chars.iter().enumerate() {
                rhs_index[i] = label_values[label as usize];
            }
            acc += *(*lhs).data.add(chelis_indices_to_flat(
                lhs_index.as_ptr(),
                (*lhs).strides.as_ptr(),
                lhs_chars.len() as c_int,
            ) as usize)
                * *(*rhs).data.add(chelis_indices_to_flat(
                    rhs_index.as_ptr(),
                    (*rhs).strides.as_ptr(),
                    rhs_chars.len() as c_int,
                ) as usize);
        }
        *(*out).data.add(out_linear) = acc;
    }
    out
}

unsafe fn write_stdout(text: &str) {
    let c_text = CString::new(text).expect("runtime print text must not contain NUL");
    libc::printf(c"%s".as_ptr(), c_text.as_ptr());
}

unsafe fn value_to_string_inline(value: chelis_value) -> String {
    match value.tag {
        chelis_value_tag::CHELIS_VALUE_INT64 => value.as_.i64_.to_string(),
        chelis_value_tag::CHELIS_VALUE_FLOAT64 => value.as_.f64_.to_string(),
        chelis_value_tag::CHELIS_VALUE_BOOL => {
            if value.as_.boolean { "true" } else { "false" }.to_string()
        }
        chelis_value_tag::CHELIS_VALUE_STRING => string_value(value.as_.string).value.clone(),
        chelis_value_tag::CHELIS_VALUE_TENSOR => tensor_to_string(value.as_.tensor),
        chelis_value_tag::CHELIS_VALUE_LIST => list_to_string(value.as_.list),
        chelis_value_tag::CHELIS_VALUE_TUPLE => tuple_to_string(value.as_.tuple),
        chelis_value_tag::CHELIS_VALUE_DICT => dict_to_string(value.as_.dict),
    }
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_list(list: *const chelis_list) {
    write_stdout(&list_to_string(list));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_tuple(tuple: *const chelis_tuple) {
    write_stdout(&tuple_to_string(tuple));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_dict(dict: *const chelis_dict) {
    write_stdout(&dict_to_string(dict));
}

#[no_mangle]
pub unsafe extern "C" fn chelis_contiguous(t: *const chelis_tensor) -> *mut chelis_tensor {
    if chelis_is_contiguous(t) != 0 {
        let out = chelis_alloc((*t).ndim, (*t).shape.as_ptr(), (*t).dtype);
        ptr::copy_nonoverlapping((*t).data, (*out).data, (*t).size as usize);
        return out;
    }
    let out = chelis_alloc((*t).ndim, (*t).shape.as_ptr(), (*t).dtype);
    let mut indices = [0; CHELIS_MAX_DIM];
    for i in 0..(*out).size {
        chelis_flat_to_indices(i, (*out).shape.as_ptr(), (*out).ndim, indices.as_mut_ptr());
        let src = chelis_indices_to_flat(indices.as_ptr(), (*t).strides.as_ptr(), (*t).ndim);
        *(*out).data.add(i as usize) = *(*t).data.add(src as usize);
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn chelis_print_f32(t: *const chelis_tensor) {
    write_stdout(&tensor_to_string(t));
}

unsafe fn list_to_string(list: *const chelis_list) -> String {
    let mut out = String::from("[");
    if !list.is_null() {
        for (i, value) in (*list).items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&value_to_string_inline(*value));
        }
    }
    out.push(']');
    out
}

unsafe fn tuple_to_string(tuple: *const chelis_tuple) -> String {
    let mut out = String::from("(");
    if !tuple.is_null() {
        for (i, value) in (*tuple).items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&value_to_string_inline(*value));
        }
    }
    out.push(')');
    out
}

unsafe fn dict_to_string(dict: *const chelis_dict) -> String {
    let mut out = String::from("dict(");
    if !dict.is_null() {
        for (i, entry) in (*dict).entries.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&value_to_string_inline(entry.key));
            out.push_str(": ");
            out.push_str(&value_to_string_inline(entry.value));
        }
    }
    out.push(')');
    out
}

unsafe fn tensor_to_string(t: *const chelis_tensor) -> String {
    let mut out = String::from("tensor(shape=[");
    for d in 0..(*t).ndim as usize {
        if d > 0 {
            out.push_str(", ");
        }
        out.push_str(&(*t).shape[d].to_string());
    }
    out.push_str("], data=[");
    let n = ((*t).size as usize).min(10);
    for i in 0..n {
        let value = *(*t).data.add(i) as f64;
        if i > 0 {
            out.push_str(", ");
        }
        if (value - value.round()).abs() < 1e-9 {
            out.push_str(&format!("{value:.1}"));
        } else {
            out.push_str(&value.to_string());
        }
    }
    if (*t).size > 10 {
        out.push_str(", ...");
    }
    out.push_str("])");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    unsafe fn runtime_str(value: &str) -> chelis_string {
        let c = CString::new(value).expect("cstring");
        chelis_string_from_cstr(c.as_ptr())
    }

    unsafe fn string_text(value: chelis_string) -> String {
        CStr::from_ptr(chelis_string_data(value))
            .to_str()
            .expect("utf8")
            .to_owned()
    }

    #[test]
    fn string_len_counts_characters() {
        unsafe {
            let value = runtime_str("naïve");
            assert_eq!(chelis_string_len(value), 5);
            chelis_string_release(value);
        }
    }

    #[test]
    fn string_slice_uses_character_offsets() {
        unsafe {
            let value = runtime_str("héllo");
            let slice = chelis_string_slice(value, 1, 3);
            assert_eq!(string_text(slice), "éll");
            chelis_string_release(slice);
            chelis_string_release(value);
        }
    }

    #[test]
    fn list_append_and_index_round_trip() {
        unsafe {
            let base = chelis_list_empty();
            let list = chelis_list_append(base, chelis_value_from_int64(41));
            let list = chelis_list_append(list, chelis_value_from_int64(42));
            let item = chelis_list_index(list, 1);
            assert_eq!(chelis_value_as_int64(item), 42);
            chelis_value_release(item);
            chelis_list_release(list);
            chelis_list_release(base);
        }
    }

    #[test]
    fn dict_insert_replaces_without_reordering() {
        unsafe {
            let key_a = chelis_value_from_string(runtime_str("a"));
            let key_b = chelis_value_from_string(runtime_str("b"));
            let dict = chelis_dict_insert(std::ptr::null(), key_a, chelis_value_from_int64(1));
            let dict = chelis_dict_insert(dict, key_b, chelis_value_from_int64(2));
            let dict = chelis_dict_insert(dict, key_a, chelis_value_from_int64(3));
            let entries = chelis_dict_entries(dict);
            assert_eq!(chelis_list_len(entries), 2);
            let first = chelis_list_index(entries, 0);
            let pair = chelis_value_as_tuple(first);
            let first_key = chelis_tuple_get(pair, 0);
            let first_value = chelis_tuple_get(pair, 1);
            assert_eq!(string_text(chelis_value_as_string(first_key)), "a");
            assert_eq!(chelis_value_as_int64(first_value), 3);
            chelis_value_release(first_value);
            chelis_value_release(first_key);
            chelis_value_release(first);
            chelis_list_release(entries);
            chelis_dict_release(dict);
            chelis_value_release(key_b);
            chelis_value_release(key_a);
        }
    }

    #[test]
    fn tensor_rank_shape_and_numel_match_allocated_layout() {
        unsafe {
            let shape = [2, 3];
            let tensor = chelis_alloc(2, shape.as_ptr(), CHELIS_F32);
            assert_eq!(chelis_tensor_rank(tensor), 2);
            assert_eq!(chelis_tensor_shape(tensor, 0), 2);
            assert_eq!(chelis_tensor_shape(tensor, 1), 3);
            assert_eq!(chelis_tensor_numel(tensor), 6);
            chelis_free(tensor);
        }
    }

    #[test]
    fn tensor_layout_stays_stable() {
        assert_eq!(std::mem::size_of::<chelis_tensor>(), 88);
        assert_eq!(std::mem::align_of::<chelis_tensor>(), 8);
    }
}
