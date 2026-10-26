use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::slice;

pub type CviModelHandle = *mut c_void;

#[repr(C)]
pub struct CviTensor {
    _opaque: [u8; 0],
}

// C ラッパー関数経由で呼び出す
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_file = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("matmul_add_bf16.cvimodel");

    let input_vec: [f32; 4] = if args.len() >= 6 {
        [
            args[2].parse()?,
            args[3].parse()?,
            args[4].parse()?,
            args[5].parse()?,
        ]
    } else {
        [1.0, 2.0, 3.0, 4.0]
    };

    let c_model_path = CString::new(model_file)?;
    let mut model: CviModelHandle = ptr::null_mut();

    unsafe {
        // 1. モデルのロード
        println!("[1] Loading model: {}", model_file);

        let ret = rust_cvi_nn_register_model(c_model_path.as_ptr(), &mut model);
        if ret != 0 {
            return Err(format!("rust_cvi_nn_register_model failed with code: {:#x}", ret).into());
        }

        // 2. 入出力テンソルの取得
        let mut inputs: *mut CviTensor = ptr::null_mut();
        let mut outputs: *mut CviTensor = ptr::null_mut();
        let mut input_num: i32 = 0;
        let mut output_num: i32 = 0;

        let ret = rust_cvi_nn_get_input_output_tensors(
            model,
            &mut inputs,
            &mut input_num,
            &mut outputs,
            &mut output_num,
        );
        if ret != 0 {
            rust_cvi_nn_cleanup_model(model);
            return Err(format!("rust_cvi_nn_get_input_output_tensors failed: {:#x}", ret).into());
        }

        let in_name = CStr::from_ptr(rust_cvi_tensor_name(inputs, 0)).to_string_lossy();
        let in_count = rust_cvi_tensor_count(inputs, 0);
        let in_bytes = rust_cvi_tensor_size(inputs, 0);
        println!(
            "[2] Input Tensor [0] : name='{}', elements={}, bytes={}",
            in_name, in_count, in_bytes
        );

        // 3. 入力データの書き込み
        let in_ptr = rust_cvi_tensor_ptr(inputs, 0) as *mut f32;
        assert!(!in_ptr.is_null(), "Input tensor pointer is null");
        assert!(in_count >= 4, "Expected at least 4 elements in input tensor");

        let in_slice = slice::from_raw_parts_mut(in_ptr, 4);
        in_slice.copy_from_slice(&input_vec);
        println!("[3] Input X written  : {:?}", in_slice);

        // 4. 推論実行
        println!("[4] Running CVI_NN_Forward on SG2000 TPU (/dev/cvi-tpu0)...");
        let ret = rust_cvi_nn_forward(model, inputs, input_num, outputs, output_num);
        if ret != 0 {
            rust_cvi_nn_cleanup_model(model);
            return Err(format!("rust_cvi_nn_forward failed: {:#x}", ret).into());
        }

        // 5. 結果の読み出し
        let out_name = CStr::from_ptr(rust_cvi_tensor_name(outputs, 0)).to_string_lossy();
        let out_count = rust_cvi_tensor_count(outputs, 0);
        let out_ptr = rust_cvi_tensor_ptr(outputs, 0) as *const f32;
        assert!(!out_ptr.is_null(), "Output tensor pointer is null");

        let out_slice = slice::from_raw_parts(out_ptr, out_count);
        println!(
            "[5] Output Tensor [0]: name='{}', result Y = {:?}",
            out_name, out_slice
        );

        let expected: [f32; 4] = [
            input_vec[0] * 1.0 + 10.0,
            input_vec[1] * 2.0 + 20.0,
            input_vec[2] * 3.0 + 30.0,
            input_vec[3] * 4.0 + 40.0,
        ];
        println!("    Expected Y       : {:?}", expected);

        // 6. クリーンアップ
        rust_cvi_nn_cleanup_model(model);
        println!("[6] Cleanup complete.");
    }

    Ok(())
}
