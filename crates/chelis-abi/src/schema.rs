//! One field list for each exact transport; Rust and C consume these lists.

#[doc(hidden)]
#[macro_export]
macro_rules! __abi_schema {
    (read, $callback:ident, [$($context:tt)*]) => {
        $crate::$callback! { [$($context)*] {
            data: ConstPointer,
            count: Int64,
            dtype: DType,
            reserved: Reserved7,
        } }
    };
    (write, $callback:ident, [$($context:tt)*]) => {
        $crate::$callback! { [$($context)*] {
            data: MutPointer,
            count: Int64,
            dtype: DType,
            reserved: Reserved7,
        } }
    };
    (device, $callback:ident, [$($context:tt)*]) => {
        $crate::$callback! { [$($context)*] {
            data: MutPointer,
            shape: ExtentPointer,
            strides: ExtentPointer,
            count: Int64,
            byte_capacity: Int64,
            rank: Rank,
            dtype: DType,
            ownership: Ownership,
            reserved: Reserved2,
        } }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __abi_field_type {
    (ConstPointer) => {
        *const ::core::ffi::c_void
    };
    (MutPointer) => {
        *mut ::core::ffi::c_void
    };
    (ExtentPointer) => {
        *const i64
    };
    (Int64) => {
        i64
    };
    (Rank) => {
        i32
    };
    (DType) => {
        u8
    };
    (Ownership) => {
        u8
    };
    (Reserved7) => {
        [u8; 7]
    };
    (Reserved2) => {
        [u8; 2]
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __abi_rust_struct {
    ([$visibility:vis $name:ident; $field_visibility:vis,] {
        $($field:ident: $kind:ident,)*
    }) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        $visibility struct $name {
            $($field_visibility $field: $crate::__abi_field_type!($kind),)*
        }
    };
}

/// Define an exact [05-OP-31] read-view transport inside its owning module.
/// `public_fields` preserves the runtime's published Rust view surface.
#[macro_export]
macro_rules! define_read_view {
    ($visibility:vis $name:ident) => {
        $crate::__abi_schema!(read, __abi_rust_struct, [$visibility $name; ,]);
    };
    ($visibility:vis $name:ident, public_fields) => {
        $crate::__abi_schema!(read, __abi_rust_struct, [$visibility $name; pub,]);
    };
}

/// Define an exact [05-OP-31] write-view transport inside its owning module.
#[macro_export]
macro_rules! define_write_view {
    ($visibility:vis $name:ident) => {
        $crate::__abi_schema!(write, __abi_rust_struct, [$visibility $name; ,]);
    };
    ($visibility:vis $name:ident, public_fields) => {
        $crate::__abi_schema!(write, __abi_rust_struct, [$visibility $name; pub,]);
    };
}

/// Define a raw device packet with fields private to the invoking owner module.
///
/// This packet is unvalidated transport, not a tensor owner or typed reference.
#[macro_export]
macro_rules! define_device_descriptor {
    ($visibility:vis $name:ident) => {
        $crate::__abi_schema!(device, __abi_rust_struct, [$visibility $name; ,]);
    };
}
