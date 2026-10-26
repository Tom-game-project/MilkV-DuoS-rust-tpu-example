mod cvi;
use cvi::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_file = args.get(1).map(|s| s.as_str()).unwrap_or("matmul_add_bf16.cvimodel");

    let input_vec: [f32; 4] = if args.len() >= 6 {
        [args[2].parse()?, args[3].parse()?, args[4].parse()?, args[5].parse()?]
    } else {
        [1.0, 2.0, 3.0, 4.0]
    };

    // 1. モデルのロード (RAII)
    println!("[1] Loading model: {}", model_file);
    let mut model = Model::load(model_file)?;

    // 2. 入力テンソルの設定 (ION メモリへゼロコピーコピー)
    let mut inputs = model.inputs();
    println!(
        "[2] Input Tensor [0] : name='{}', elements={}, bytes={}",
        inputs[0].name(), inputs[0].count(), inputs[0].size_bytes()
    );

    let in_slice = inputs[0].as_slice_mut::<f32>();
    in_slice[..4].copy_from_slice(&input_vec);
    println!("[3] Input X written  : {:?}", &in_slice[..4]);

    // 3. 推論実行
    println!("[4] Running CVI_NN_Forward on SG2000 TPU (/dev/cvi-tpu0)...");
    model.forward()?;

    // 4. 計算結果の参照 (ゼロコピー読み出し)
    let outputs = model.outputs();
    let out_slice = outputs[0].as_slice::<f32>();
    println!(
        "[5] Output Tensor [0]: name='{}', result Y = {:?}",
        outputs[0].name(), out_slice
    );

    let expected = [
        input_vec[0] * 1.0 + 10.0,
        input_vec[1] * 2.0 + 20.0,
        input_vec[2] * 3.0 + 30.0,
        input_vec[3] * 4.0 + 40.0,
    ];
    println!("    Expected Y       : {:?}", expected);

    // スコープ終了時に Model の Drop が走り、自動的に ION メモリがクリーンアップされる
    Ok(())
}
