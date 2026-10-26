// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    let sdk_dir = PathBuf::from(
        env::var("TPU_SDK_DIR").expect("環境変数 TPU_SDK_DIR に tpu-sdk-sg200x のパスを指定してください"),
    );

    // cviruntime.h の static inline ヘルパーを C 関数としてラップ
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper_c = out_dir.join("cvi_wrapper.c");
    std::fs::write(
        &wrapper_c,
        r#"
        #include "cviruntime.h"

        // テンソル操作ヘルパー
        void* rust_cvi_tensor_ptr(CVI_TENSOR* t, int idx) { return CVI_NN_TensorPtr(&t[idx]); }
        size_t rust_cvi_tensor_size(CVI_TENSOR* t, int idx) { return CVI_NN_TensorSize(&t[idx]); }
        size_t rust_cvi_tensor_count(CVI_TENSOR* t, int idx) { return CVI_NN_TensorCount(&t[idx]); }
        const char* rust_cvi_tensor_name(CVI_TENSOR* t, int idx) { return CVI_NN_TensorName(&t[idx]); }

        // CVI_NN API ラッパー
        int rust_cvi_nn_register_model(const char* model_file, CVI_MODEL_HANDLE* model) {
            return CVI_NN_RegisterModel(model_file, model);
        }

        int rust_cvi_nn_get_input_output_tensors(
            CVI_MODEL_HANDLE model,
            CVI_TENSOR** inputs,
            int* input_num,
            CVI_TENSOR** outputs,
            int* output_num) {
            return CVI_NN_GetInputOutputTensors(model, inputs, input_num, outputs, output_num);
        }

        int rust_cvi_nn_forward(
            CVI_MODEL_HANDLE model,
            CVI_TENSOR* inputs,
            int input_num,
            CVI_TENSOR* outputs,
            int output_num) {
            return CVI_NN_Forward(model, inputs, input_num, outputs, output_num);
        }

        int rust_cvi_nn_cleanup_model(CVI_MODEL_HANDLE model) {
            return CVI_NN_CleanupModel(model);
        }
        "#,
    )
    .unwrap();

    cc::Build::new()
        .file(&wrapper_c)
        .include(sdk_dir.join("include"))
        .compile("cvi_wrapper");

    println!("cargo:rustc-link-search=native={}", sdk_dir.join("lib").display());
    // println!("cargo:rustc-link-lib=dylib=cviruntime");
    // .so を動的リンク指定
    println!("cargo:rustc-link-lib=dylib=cviruntime");
    println!("cargo:rustc-link-lib=dylib=cvikernel");
    println!("cargo:rustc-link-lib=dylib=cvimath");
    // .a ファイル群を静的リンク (static=...) として指定
    // println!("cargo:rustc-link-lib=static=cviruntime-static");
    // println!("cargo:rustc-link-lib=static=cvikernel-static");
    // println!("cargo:rustc-link-lib=static=cvimath-static");
    // println!("cargo:rustc-link-lib=stdc++");

}
