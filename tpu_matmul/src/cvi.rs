use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::slice;

// --- Low-Level FFI (Raw C Bindings) ---
mod ffi {
    use super::*;

    pub type CviModelHandle = *mut c_void;

    #[repr(C)]
    pub struct CviTensor {
        _opaque: [u8; 0],
    }

    unsafe extern "C" {
        pub fn rust_cvi_nn_register_model(model_file: *const c_char, model: *mut CviModelHandle) -> c_int;
        pub fn rust_cvi_nn_get_input_output_tensors(
            model: CviModelHandle,
            inputs: *mut *mut CviTensor,
            input_num: *mut i32,
            outputs: *mut *mut CviTensor,
            output_num: *mut i32,
        ) -> c_int;
        pub fn rust_cvi_nn_forward(
            model: CviModelHandle,
            inputs: *mut CviTensor,
            input_num: i32,
            outputs: *mut CviTensor,
            output_num: i32,
        ) -> c_int;
        pub fn rust_cvi_nn_cleanup_model(model: CviModelHandle) -> c_int;

        pub fn rust_cvi_tensor_ptr(tensor: *mut CviTensor, idx: i32) -> *mut c_void;
        pub fn rust_cvi_tensor_size(tensor: *mut CviTensor, idx: i32) -> usize;
        pub fn rust_cvi_tensor_count(tensor: *mut CviTensor, idx: i32) -> usize;
        pub fn rust_cvi_tensor_name(tensor: *mut CviTensor, idx: i32) -> *const c_char;
    }
}

// --- High-Level Safe Abstraction ---

use std::fmt;

/// TPU モデルのエラー型
#[derive(Debug)]
pub enum CviError {
    RegisterFailed(i32),
    GetTensorsFailed(i32),
    ForwardFailed(i32),
    InvalidPath,
}

// 1. エラーを人間が読める文字列に変換するための Display トレイト実装
impl fmt::Display for CviError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegisterFailed(code) => {
                write!(f, "Failed to register model from path (code: {:#x})", code)
            }
            Self::GetTensorsFailed(code) => {
                write!(f, "Failed to query input/output tensors (code: {:#x})", code)
            }
            Self::ForwardFailed(code) => {
                write!(f, "TPU Forward inference failed (code: {:#x})", code)
            }
            Self::InvalidPath => write!(f, "Invalid path string"),
        }
    }
}

impl std::error::Error for CviError {}

/// ION メモリ上の TPU モデルリソースを管理する RAII ラッパー
pub struct Model {
    handle: ffi::CviModelHandle,
    inputs_raw: *mut ffi::CviTensor,
    input_num: i32,
    outputs_raw: *mut ffi::CviTensor,
    output_num: i32,
}

impl Model {
    /// `.cvimodel` をロードして ION メモリに初期化する
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CviError> {
        let path_str = path.as_ref().to_str().ok_or(CviError::InvalidPath)?;
        let c_path = std::ffi::CString::new(path_str).map_err(|_| CviError::InvalidPath)?;

        let mut handle: ffi::CviModelHandle = ptr::null_mut();
        let ret = unsafe { ffi::rust_cvi_nn_register_model(c_path.as_ptr(), &mut handle) };
        if ret != 0 {
            return Err(CviError::RegisterFailed(ret));
        }

        let mut inputs_raw = ptr::null_mut();
        let mut outputs_raw = ptr::null_mut();
        let mut input_num = 0;
        let mut output_num = 0;

        let ret = unsafe {
            ffi::rust_cvi_nn_get_input_output_tensors(
                handle,
                &mut inputs_raw,
                &mut input_num,
                &mut outputs_raw,
                &mut output_num,
            )
        };

        if ret != 0 {
            unsafe { ffi::rust_cvi_nn_cleanup_model(handle) };
            return Err(CviError::GetTensorsFailed(ret));
        }

        Ok(Self {
            handle,
            inputs_raw,
            input_num,
            outputs_raw,
            output_num,
        })
    }

    /// 入力テンソルのリストを取得（Model のライフタイムと紐付け）
    pub fn inputs(&self) -> Vec<Tensor<'_>> {
        (0..self.input_num)
            .map(|i| Tensor {
                base_ptr: self.inputs_raw,
                idx: i,
                _marker: std::marker::PhantomData,
            })
            .collect()
    }

    /// 出力テンソルのリストを取得（Model のライフタイムと紐付け）
    pub fn outputs(&self) -> Vec<Tensor<'_>> {
        (0..self.output_num)
            .map(|i| Tensor {
                base_ptr: self.outputs_raw,
                idx: i,
                _marker: std::marker::PhantomData,
            })
            .collect()
    }

    /// 推論を実行する
    pub fn forward(&mut self) -> Result<(), CviError> {
        let ret = unsafe {
            ffi::rust_cvi_nn_forward(
                self.handle,
                self.inputs_raw,
                self.input_num,
                self.outputs_raw,
                self.output_num,
            )
        };
        if ret != 0 {
            Err(CviError::ForwardFailed(ret))
        } else {
            Ok(())
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::rust_cvi_nn_cleanup_model(self.handle);
            }
        }
    }
}

/// 個別のテンソル参照（ION メモリへのゼロコピービュー）
pub struct Tensor<'a> {
    base_ptr: *mut ffi::CviTensor,
    idx: i32,
    _marker: std::marker::PhantomData<&'a Model>,
}

impl<'a> Tensor<'a> {
    pub fn name(&self) -> &str {
        unsafe {
            let c_str = ffi::rust_cvi_tensor_name(self.base_ptr, self.idx);
            CStr::from_ptr(c_str).to_str().unwrap_or("")
        }
    }

    pub fn count(&self) -> usize {
        unsafe { ffi::rust_cvi_tensor_count(self.base_ptr, self.idx) }
    }

    pub fn size_bytes(&self) -> usize {
        unsafe { ffi::rust_cvi_tensor_size(self.base_ptr, self.idx) }
    }

    /// 入力用: ION メモリへの書き込み用ミュータブルスライスを取得
    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        let ptr = unsafe { ffi::rust_cvi_tensor_ptr(self.base_ptr, self.idx) } as *mut T;
        assert!(!ptr.is_null(), "Tensor pointer is null");
        unsafe { slice::from_raw_parts_mut(ptr, self.count()) }
    }

    /// 出力用: ION メモリからの読み出し用スライスを取得
    pub fn as_slice<T>(&self) -> &[T] {
        let ptr = unsafe { ffi::rust_cvi_tensor_ptr(self.base_ptr, self.idx) } as *const T;
        assert!(!ptr.is_null(), "Tensor pointer is null");
        unsafe { slice::from_raw_parts(ptr, self.count()) }
    }
}
