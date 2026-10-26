# /// script
# requires-python = ">=3.10"
# dependencies = [ "numpy",
#     "onnx",
# ]
# ///
"""
Milk-V Duo S (SG2000 / cv181x) の TPU で動作確認するための
最小の行列演算モデル (Y = X @ W + B) とテスト用入力 (.npz) を生成するスクリプト。

実行方法 (uv 単体実行):
    uv run generate_matmul_onnx.py
"""

import numpy as np
import onnx
from onnx import TensorProto, helper


def main() -> None:
    # 1. 目で見て暗算できる 4x4 の定数重み行列 W とバイアス B を定義
    #    W: 対角成分を 1, 2, 3, 4 としたスケーリング行列 (計算結果が一目でわかる構成)
    w_data = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )

    #    B: 各列に 10, 20, 30, 40 を加算するバイアス
    b_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    # 2. ONNX の入出力テンソル型定義 (固定形状: [1, 4])
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    # 3. 定数パラメータ (Initializer) の作成
    w_init = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=w_data.shape,
        vals=w_data.flatten().tolist(),
    )
    b_init = helper.make_tensor(
        name="B",
        data_type=TensorProto.FLOAT,
        dims=b_data.shape,
        vals=b_data.flatten().tolist(),
    )

    # 4. 演算ノードの定義: Y = (X @ W) + B
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["X", "W"],
        outputs=["XW"],
        name="MatMul_Node",
    )
    add_node = helper.make_node(
        "Add",
        inputs=["XW", "B"],
        outputs=["Y"],
        name="Add_Node",
    )

    # 5. グラフとモデルの作成 (Opset 13)
    graph = helper.make_graph(
        nodes=[matmul_node, add_node],
        name="SimpleMatMulAddGraph",
        inputs=[x_info],
        outputs=[y_info],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(
        graph,
        producer_name="milkv-duos-matmul-demo",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx_path = "matmul_add.onnx"
    onnx.save(model, onnx_path)

    # 6. TPU-MLIR の検証ゲートおよび実機テスト用の入力データ (.npz) を作成
    #    X = [[1.0, 2.0, 3.0, 4.0]]
    #    期待される出力 Y:
    #      Y[0] = 1.0 * 1.0 + 10.0 = 11.0
    #      Y[1] = 2.0 * 2.0 + 20.0 = 24.0
    #      Y[2] = 3.0 * 3.0 + 30.0 = 39.0
    #      Y[3] = 4.0 * 4.0 + 40.0 = 56.0
    x_test = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y_expected = x_test @ w_data + b_data

    npz_path = "matmul_in.npz"
    np.savez(npz_path, X=x_test)

    print(f"[OK] Saved ONNX model : {onnx_path}")
    print(f"[OK] Saved test input : {npz_path}")
    print(f"Input X               : {x_test.tolist()}")
    print(f"Expected Output Y     : {y_expected.tolist()}")


if __name__ == "__main__":
    main()
