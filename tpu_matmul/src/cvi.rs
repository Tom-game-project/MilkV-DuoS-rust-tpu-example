use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr::{self, NonNull};
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
        pub fn rust_cvi_nn_register_model(
            model_file: *const c_char,
            model: *mut CviModelHandle,
        ) -> c_int;
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

/// ION バッファへ安全に直接読み書きできるプリミティブ型を表す Sealed トレイト
mod private {
    pub trait Sealed {}
}

/// TPU テンソル要素として扱える POD (Plain Old Data) 型
///
/// # Safety
/// メモリレイアウトが C 互換で、任意のビットパターンが有効（未定義動作を起こさない）な数値型のみに実装する。
pub unsafe trait TensorElem: private::Sealed + Copy + 'static {}

macro_rules! impl_tensor_elem {
    ($($t:ty),* $(,)?) => {
        $(
            impl private::Sealed for $t {}
            unsafe impl TensorElem for $t {}
        )*
    };
}

// FP32, INT8, UINT8, BF16/FP16 (u16 bits), INT16, INT32 を許可
impl_tensor_elem!(f32, i8, u8, i16, u16, i32, u32);

/// TPU モデルのエラー型
#[derive(Debug)]
pub enum CviError {
    RegisterFailed(i32),
    GetTensorsFailed(i32),
    ForwardFailed(i32),
    InvalidPath,
    IndexOutOfBounds { index: usize, len: usize },
    TypeSizeMismatch {
        tensor_bytes: usize,
        elem_count: usize,
        elem_size: usize,
    },
}

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
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "Tensor index {} out of bounds (len: {})", index, len)
            }
            Self::TypeSizeMismatch {
                tensor_bytes,
                elem_count,
                elem_size,
            } => {
                write!(
                    f,
                    "Tensor byte size mismatch: buffer is {} bytes, but {} elements of {} bytes require {} bytes",
                    tensor_bytes,
                    elem_count,
                    elem_size,
                    elem_count * elem_size
                )
            }
        }
    }
}

impl std::error::Error for CviError {}

/// ION メモリ上の TPU モデルリソースを管理する RAII ラッパー
pub struct Model {
    handle: NonNull<c_void>,
    inputs_raw: *mut ffi::CviTensor,
    input_num: i32,
    outputs_raw: *mut ffi::CviTensor,
    output_num: i32,
}

impl Model {
    /// `.cvimodel` をロードして ION メモリに初期化する
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CviError> {
        #[cfg(unix)]
        let c_path = {
            use std::os::unix::ffi::OsStrExt;
            std::ffi::CString::new(path.as_ref().as_os_str().as_bytes())
                .map_err(|_| CviError::InvalidPath)?
        };
        #[cfg(not(unix))]
        let c_path = {
            let path_str = path.as_ref().to_str().ok_or(CviError::InvalidPath)?;
            std::ffi::CString::new(path_str).map_err(|_| CviError::InvalidPath)?
        };

        let mut raw_handle: ffi::CviModelHandle = ptr::null_mut();
        let ret = unsafe { ffi::rust_cvi_nn_register_model(c_path.as_ptr(), &mut raw_handle) };
        if ret != 0 {
            return Err(CviError::RegisterFailed(ret));
        }
        let handle = NonNull::new(raw_handle).ok_or(CviError::RegisterFailed(-1))?;

        let mut inputs_raw = ptr::null_mut();
        let mut outputs_raw = ptr::null_mut();
        let mut input_num = 0;
        let mut output_num = 0;

        let ret = unsafe {
            ffi::rust_cvi_nn_get_input_output_tensors(
                handle.as_ptr(),
                &mut inputs_raw,
                &mut input_num,
                &mut outputs_raw,
                &mut output_num,
            )
        };

        if ret != 0 {
            unsafe { ffi::rust_cvi_nn_cleanup_model(handle.as_ptr()) };
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

    /// 入力テンソルの数
    pub fn input_count(&self) -> usize {
        self.input_num.max(0) as usize
    }

    /// 出力テンソルの数
    pub fn output_count(&self) -> usize {
        self.output_num.max(0) as usize
    }

    /// 指定インデックスの入力テンソル（可変ビュー）を取得する
    /// `&mut self` を借用するため、スライス保持中に `forward()` や二重借用が起きるのをコンパイル時に防ぐ
    pub fn input_mut(&mut self, index: usize) -> Result<TensorMut<'_>, CviError> {
        let len = self.input_count();
        if index >= len {
            return Err(CviError::IndexOutOfBounds { index, len });
        }
        Ok(TensorMut {
            base_ptr: self.inputs_raw,
            idx: index as i32,
            _marker: PhantomData,
        })
    }

    /// 指定インデックスの出力テンソル（不変ビュー）を取得する
    /// `&self` を借用するため、出力スライスを参照している間は `forward()` による上書きをコンパイル時に防ぐ
    pub fn output(&self, index: usize) -> Result<TensorRef<'_>, CviError> {
        let len = self.output_count();
        if index >= len {
            return Err(CviError::IndexOutOfBounds { index, len });
        }
        Ok(TensorRef {
            base_ptr: self.outputs_raw,
            idx: index as i32,
            _marker: PhantomData,
        })
    }

    /// 全入力テンソルのリスト（可変ビュー）を取得する
    pub fn inputs_mut(&mut self) -> Vec<TensorMut<'_>> {
        (0..self.input_num)
            .map(|i| TensorMut {
                base_ptr: self.inputs_raw,
                idx: i,
                _marker: PhantomData,
            })
            .collect()
    }

    /// 全出力テンソルのリスト（不変ビュー）を取得する
    pub fn outputs(&self) -> Vec<TensorRef<'_>> {
        (0..self.output_num)
            .map(|i| TensorRef {
                base_ptr: self.outputs_raw,
                idx: i,
                _marker: PhantomData,
            })
            .collect()
    }

    /// 推論を実行する
    pub fn forward(&mut self) -> Result<(), CviError> {
        let ret = unsafe {
            ffi::rust_cvi_nn_forward(
                self.handle.as_ptr(),
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
        unsafe {
            ffi::rust_cvi_nn_cleanup_model(self.handle.as_ptr());
        }
    }
}

/// 出力テンソル等の読み取り専用ビュー（`&'a Model` に紐づく）
pub struct TensorRef<'a> {
    base_ptr: *mut ffi::CviTensor,
    idx: i32,
    _marker: PhantomData<&'a Model>,
}

impl<'a> TensorRef<'a> {
    pub fn name(&self) -> &'a str {
        unsafe {
            let c_str = ffi::rust_cvi_tensor_name(self.base_ptr, self.idx);
            if c_str.is_null() {
                return "";
            }
            CStr::from_ptr(c_str).to_str().unwrap_or("")
        }
    }

    pub fn count(&self) -> usize {
        unsafe { ffi::rust_cvi_tensor_count(self.base_ptr, self.idx) }
    }

    pub fn size_bytes(&self) -> usize {
        unsafe { ffi::rust_cvi_tensor_size(self.base_ptr, self.idx) }
    }

    /// 型サイズとバッファバイト数の整合性を検証してスライスを返す（ライフタイム `'a` で Model に紐づく）
    pub fn try_as_slice<T: TensorElem>(&self) -> Result<&'a [T], CviError> {
        let elem_count = self.count();
        let tensor_bytes = self.size_bytes();
        let elem_size = std::mem::size_of::<T>();
        if elem_count * elem_size != tensor_bytes {
            return Err(CviError::TypeSizeMismatch {
                tensor_bytes,
                elem_count,
                elem_size,
            });
        }
        let ptr = unsafe { ffi::rust_cvi_tensor_ptr(self.base_ptr, self.idx) } as *const T;
        assert!(!ptr.is_null(), "Tensor pointer is null");
        Ok(unsafe { slice::from_raw_parts(ptr, elem_count) })
    }

    /// ION メモリからの読み出し用スライスを取得（サイズが合わない場合は panic）
    pub fn as_slice<T: TensorElem>(&self) -> &'a [T] {
        self.try_as_slice().expect("Tensor element type size mismatch")
    }
}

/// 入力テンソルの読み書き可能ビュー（`&'a mut Model` に紐づく）
pub struct TensorMut<'a> {
    base_ptr: *mut ffi::CviTensor,
    idx: i32,
    _marker: PhantomData<&'a mut Model>,
}

impl<'a> TensorMut<'a> {
    pub fn name(&self) -> &str {
        unsafe {
            let c_str = ffi::rust_cvi_tensor_name(self.base_ptr, self.idx);
            if c_str.is_null() {
                return "";
            }
            CStr::from_ptr(c_str).to_str().unwrap_or("")
        }
    }

    pub fn count(&self) -> usize {
        unsafe { ffi::rust_cvi_tensor_count(self.base_ptr, self.idx) }
    }

    pub fn size_bytes(&self) -> usize {
        unsafe { ffi::rust_cvi_tensor_size(self.base_ptr, self.idx) }
    }

    /// 型サイズとバッファバイト数の整合性を検証して可変スライスを返す
    pub fn try_as_slice_mut<T: TensorElem>(&mut self) -> Result<&mut [T], CviError> {
        let elem_count = self.count();
        let tensor_bytes = self.size_bytes();
        let elem_size = std::mem::size_of::<T>();
        if elem_count * elem_size != tensor_bytes {
            return Err(CviError::TypeSizeMismatch {
                tensor_bytes,
                elem_count,
                elem_size,
            });
        }
        let ptr = unsafe { ffi::rust_cvi_tensor_ptr(self.base_ptr, self.idx) } as *mut T;
        assert!(!ptr.is_null(), "Tensor pointer is null");
        Ok(unsafe { slice::from_raw_parts_mut(ptr, elem_count) })
    }

    /// ION メモリへの書き込み用ミュータブルスライスを取得（サイズが合わない場合は panic）
    pub fn as_slice_mut<T: TensorElem>(&mut self) -> &mut [T] {
        self.try_as_slice_mut()
            .expect("Tensor element type size mismatch")
    }
}