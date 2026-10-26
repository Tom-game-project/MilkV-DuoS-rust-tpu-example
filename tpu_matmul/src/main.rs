mod cvi;
use cvi::Model;

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

    // 1. モデルのロード (RAII)
    println!("[1] Loading model: {}", model_file);
    let mut model = Model::load(model_file)?;

    // 2. 入力テンソルの設定 (ION メモリへゼロコピー書き込み)
    {
        let mut input0 = model.input_mut(0)?;
        println!(
            "[2] Input Tensor [0] : name='{}', elements={}, bytes={}",
            input0.name(),
            input0.count(),
            input0.size_bytes()
        );

        let in_slice = input0.try_as_slice_mut::<f32>()?;
        in_slice[..4].copy_from_slice(&input_vec);
        println!("[3] Input X written  : {:?}", &in_slice[..4]);
    } // ← ここで可変借用 (input0) が終了するため、安全に model.forward() を呼べる

    // 3. 推論実行
    println!("[4] Running CVI_NN_Forward on SG2000 TPU (/dev/cvi-tpu0)...");
    model.forward()?;

    // 4. 計算結果の参照 (ゼロコピー読み出し)
    let output0 = model.output(0)?;
    let out_slice = output0.try_as_slice::<f32>()?;
    println!(
        "[5] Output Tensor [0]: name='{}', result Y = {:?}",
        output0.name(),
        out_slice
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