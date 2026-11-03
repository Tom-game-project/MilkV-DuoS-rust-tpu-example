//! Milk-V Duo S (SG2000 / cv181x) の TPU で動作確認するための
//! 最小の行列演算モデル (Y = X @ W + B) を生成するプログラム。

use protobuf_shape_guard::Expr2D;

fn main() -> std::io::Result<()> {
    // 1. 目で見て暗算できる 4x4 の定数重み行列 W とバイアス B を定義
    //    W: 対角成分を 1, 2, 3, 4 としたスケーリング行列 (計算結果が一目でわかる構成)
    let w = Expr2D::<4, 4>::constant_nd(
        "W",
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
    );

    //    B: 各列に 10, 20, 30, 40 を加算するバイアス
    let b = Expr2D::<1, 4>::constant_nd("B", [[10.0, 20.0, 30.0, 40.0]]);

    // 2. ONNX の入出力テンソル型定義 (固定形状: [1, 4])
    let x = Expr2D::<1, 4>::input("X");

    // 4. 演算ノードの定義: Y = (X @ W) + B
    let y: Expr2D<1, 4> = x
        .matmul(&w)
        .with_output_name("XW")
        .add(&b)
        .with_output_name("Y");

    // 5. グラフとモデルの作成 (Opset 13)
    let onnx_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../matmul_add.onnx".to_string());
    y.save_onnx(&onnx_path)?;

    println!("[OK] Saved ONNX model : {}", onnx_path);
    Ok(())
}
