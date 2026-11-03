use prost::Message;
use std::marker::PhantomData;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================================
// 0. prost による ONNX Protobuf スキーマ定義 (onnx.proto3 準拠)
//    `prost-build` や外部の `protoc` コンパイラを一切インストールしなくても、
//    `#[derive(prost::Message)]` とタグ番号だけで標準規格の `.onnx` バイナリを直接生成できる！
// ============================================================================
pub mod onnx_proto {
    use prost::Message;

    /// AttributeProto (フィールド番号: name=1, f=2, i=3, floats=7, ints=8, type=20)
    #[derive(Clone, PartialEq, Message)]
    pub struct AttributeProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(float, optional, tag = "2")]
        pub f: Option<f32>,
        #[prost(int64, optional, tag = "3")]
        pub i: Option<i64>,
        #[prost(float, repeated, packed = "false", tag = "7")]
        pub floats: Vec<f32>,
        #[prost(int64, repeated, packed = "false", tag = "8")]
        pub ints: Vec<i64>,
        /// AttributeType: FLOAT = 1, INT = 2, FLOATS = 6, INTS = 7
        #[prost(int32, tag = "20")]
        pub r#type: i32,
    }

    /// NodeProto (フィールド番号: input=1, output=2, name=3, op_type=4, attribute=5)
    #[derive(Clone, PartialEq, Message)]
    pub struct NodeProto {
        #[prost(string, repeated, tag = "1")]
        pub input: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        pub output: Vec<String>,
        #[prost(string, tag = "3")]
        pub name: String,
        #[prost(string, tag = "4")]
        pub op_type: String,
        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<AttributeProto>,
    }

    /// TensorProto (フィールド番号: dims=1, data_type=2, float_data=4, int64_data=7, name=8, raw_data=9)
    #[derive(Clone, PartialEq, Message)]
    pub struct TensorProto {
        #[prost(int64, repeated, tag = "1")]
        pub dims: Vec<i64>,
        /// DataType: FLOAT = 1, INT64 = 7
        #[prost(int32, tag = "2")]
        pub data_type: i32,
        #[prost(float, repeated, tag = "4")]
        pub float_data: Vec<f32>,
        #[prost(int64, repeated, tag = "7")]
        pub int64_data: Vec<i64>,
        #[prost(string, tag = "8")]
        pub name: String,
        #[prost(bytes = "vec", optional, tag = "9")]
        pub raw_data: Option<Vec<u8>>,
    }

    /// TensorShapeProto.Dimension (フィールド番号: dim_value=1)
    #[derive(Clone, PartialEq, Message)]
    pub struct Dimension {
        #[prost(int64, tag = "1")]
        pub dim_value: i64,
    }

    /// TensorShapeProto (フィールド番号: dim=1)
    #[derive(Clone, PartialEq, Message)]
    pub struct TensorShapeProto {
        #[prost(message, repeated, tag = "1")]
        pub dim: Vec<Dimension>,
    }

    /// TypeProto.Tensor (フィールド番号: elem_type=1, shape=2)
    #[derive(Clone, PartialEq, Message)]
    pub struct TypeProtoTensor {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<TensorShapeProto>,
    }

    /// TypeProto (フィールド番号: tensor_type=1)
    #[derive(Clone, PartialEq, Message)]
    pub struct TypeProto {
        #[prost(message, optional, tag = "1")]
        pub tensor_type: Option<TypeProtoTensor>,
    }

    /// ValueInfoProto (フィールド番号: name=1, type=2)
    #[derive(Clone, PartialEq, Message)]
    pub struct ValueInfoProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(message, optional, tag = "2")]
        pub r#type: Option<TypeProto>,
    }

    /// GraphProto (フィールド番号: node=1, name=2, initializer=5, input=11, output=12)
    #[derive(Clone, PartialEq, Message)]
    pub struct GraphProto {
        #[prost(message, repeated, tag = "1")]
        pub node: Vec<NodeProto>,
        #[prost(string, tag = "2")]
        pub name: String,
        #[prost(message, repeated, tag = "5")]
        pub initializer: Vec<TensorProto>,
        #[prost(message, repeated, tag = "11")]
        pub input: Vec<ValueInfoProto>,
        #[prost(message, repeated, tag = "12")]
        pub output: Vec<ValueInfoProto>,
    }

    /// OperatorSetIdProto (フィールド番号: domain=1, version=2)
    #[derive(Clone, PartialEq, Message)]
    pub struct OperatorSetIdProto {
        #[prost(string, tag = "1")]
        pub domain: String,
        #[prost(int64, tag = "2")]
        pub version: i64,
    }

    /// ModelProto (フィールド番号: ir_version=1, producer_name=2, graph=7, opset_import=8)
    #[derive(Clone, PartialEq, Message)]
    pub struct ModelProto {
        #[prost(int64, tag = "1")]
        pub ir_version: i64,
        #[prost(string, tag = "2")]
        pub producer_name: String,
        #[prost(message, optional, tag = "7")]
        pub graph: Option<GraphProto>,
        #[prost(message, repeated, tag = "8")]
        pub opset_import: Vec<OperatorSetIdProto>,
    }

    impl ValueInfoProto {
        pub fn new_tensor(name: &str, elem_type: i32, dims: &[usize]) -> Self {
            Self {
                name: name.to_string(),
                r#type: Some(TypeProto {
                    tensor_type: Some(TypeProtoTensor {
                        elem_type,
                        shape: Some(TensorShapeProto {
                            dim: dims
                                .iter()
                                .map(|&d| Dimension {
                                    dim_value: d as i64,
                                })
                                .collect(),
                        }),
                    }),
                }),
            }
        }
    }
}

// ============================================================================
// 1. 動的な AST ノード定義 (任意の次元数に対応)
// ============================================================================
#[derive(Clone, Debug)]
pub enum AttrValue {
    Floats(Vec<f32>),
    Ints(Vec<i64>),
}

#[derive(Clone, Debug)]
pub enum AstNode {
    Input {
        name: String,
        shape: Vec<usize>,
    },
    Constant {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    },
    ConstantInt64 {
        name: String,
        shape: Vec<usize>,
        data: Vec<i64>,
    },
    Op {
        op_type: String,
        inputs: Vec<Rc<AstNode>>,
        attrs: Vec<(String, AttrValue)>,
        output_name: String,
    },
}

fn next_id() -> u32 {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// 2. 次元 (Rank) を型として表現する Shape トレイト
// ============================================================================
pub trait Shape {
    type ArrayType;
    fn dims() -> Vec<usize>;
    fn numel() -> usize;
    fn flatten_array(arr: &Self::ArrayType) -> Vec<f32>;
}

/// 2階テンソル (行列: [Rows, Cols])
#[derive(Clone, Copy, Debug)]
pub struct Dim2<const R: usize, const C: usize>;

/// 4階テンソル (TPU ネイティブ形式: [Batch, Channel, Height, Width])
#[derive(Clone, Copy, Debug)]
pub struct Dim4<const N: usize, const C: usize, const H: usize, const W: usize>;

impl<const R: usize, const C: usize> Shape for Dim2<R, C> {
    type ArrayType = [[f32; C]; R];

    fn dims() -> Vec<usize> {
        vec![R, C]
    }

    fn numel() -> usize {
        R * C
    }

    fn flatten_array(arr: &Self::ArrayType) -> Vec<f32> {
        arr.as_flattened().to_vec()
    }
}

impl<const N: usize, const C: usize, const H: usize, const W: usize> Shape for Dim4<N, C, H, W> {
    type ArrayType = [[[[f32; W]; H]; C]; N];

    fn dims() -> Vec<usize> {
        vec![N, C, H, W]
    }

    fn numel() -> usize {
        N * C * H * W
    }

    fn flatten_array(arr: &Self::ArrayType) -> Vec<f32> {
        let mut out = Vec::with_capacity(Self::numel());
        for n in arr {
            for c in n {
                for h in c {
                    out.extend_from_slice(h);
                }
            }
        }
        out
    }
}

// ============================================================================
// 3. 型安全なテンソル式ラッパー `Expr<S: Shape>` と型エイリアス
// ============================================================================
#[derive(Clone, Debug)]
pub struct Expr<S: Shape> {
    pub node: Rc<AstNode>,
    _shape: PhantomData<S>,
}

/// 2階テンソル (行列) の型エイリアス: `Expr2D<R, C>`
pub type Expr2D<const R: usize, const C: usize> = Expr<Dim2<R, C>>;

/// 4階テンソル (NCHW) の型エイリアス: `Expr4D<N, C, H, W>`
pub type Expr4D<const N: usize, const C: usize, const H: usize, const W: usize> =
    Expr<Dim4<N, C, H, W>>;

// --- 任意の階数のテンソル (2D / 4D 共通) で使えるメソッド ---
impl<S: Shape> Expr<S> {
    fn from_node(node: AstNode) -> Self {
        Self {
            node: Rc::new(node),
            _shape: PhantomData,
        }
    }

    /// 入力テンソルの作成 (型パラメータの Shape がそのまま ONNX の入力形状になる)
    pub fn input(name: &str) -> Self {
        Self::from_node(AstNode::Input {
            name: name.to_string(),
            shape: S::dims(),
        })
    }

    /// 多次元固定長配列 (`[[f32; C]; R]` または `[[[[f32; W]; H]; C]; N]`) から定数を作成
    pub fn constant_nd(name: &str, data: S::ArrayType) -> Self {
        Self::from_node(AstNode::Constant {
            name: name.to_string(),
            shape: S::dims(),
            data: S::flatten_array(&data),
        })
    }

    /// 1次元 Vec<f32> から定数を作成 (実行時要素数チェック付き)
    pub fn constant_vec(name: &str, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            S::numel(),
            "Constant '{}' shape {:?} requires {} elements, got {}",
            name,
            S::dims(),
            S::numel(),
            data.len()
        );
        Self::from_node(AstNode::Constant {
            name: name.to_string(),
            shape: S::dims(),
            data,
        })
    }
    /// 出力テンソル名を明示的に指定したい場合 (Python の `outputs=["Y"]` と完全一致させる用)
    pub fn with_output_name(self, new_name: &str) -> Self {
        match (*self.node).clone() {
            AstNode::Op {
                op_type,
                inputs,
                attrs,
                ..
            } => Self::from_node(AstNode::Op {
                op_type,
                inputs,
                attrs,
                output_name: new_name.to_string(),
            }),
            other => Self::from_node(other),
        }
    }

    /// 同形状テンソル同士の要素ごと加算
    pub fn add(&self, rhs: &Expr<S>) -> Expr<S> {
        Self::from_node(AstNode::Op {
            op_type: "Add".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&rhs.node)],
            attrs: vec![],
            output_name: format!("add_out_{}", next_id()),
        })
    }

    /// 要素ごとの乗算 (Hadamard 積)
    pub fn mul(&self, rhs: &Expr<S>) -> Expr<S> {
        Self::from_node(AstNode::Op {
            op_type: "Mul".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&rhs.node)],
            attrs: vec![],
            output_name: format!("mul_out_{}", next_id()),
        })
    }

    /// ReLU 活性化関数
    pub fn relu(&self) -> Expr<S> {
        Self::from_node(AstNode::Op {
            op_type: "Relu".to_string(),
            inputs: vec![Rc::clone(&self.node)],
            attrs: vec![],
            output_name: format!("relu_out_{}", next_id()),
        })
    }

    /// LeakyReLU 活性化関数
    pub fn leaky_relu(&self, alpha: f32) -> Expr<S> {
        Self::from_node(AstNode::Op {
            op_type: "LeakyRelu".to_string(),
            inputs: vec![Rc::clone(&self.node)],
            attrs: vec![("alpha".to_string(), AttrValue::Floats(vec![alpha]))],
            output_name: format!("leaky_relu_out_{}", next_id()),
        })
    }

    /// Tanh 活性化関数
    pub fn tanh(&self) -> Expr<S> {
        Self::from_node(AstNode::Op {
            op_type: "Tanh".to_string(),
            inputs: vec![Rc::clone(&self.node)],
            attrs: vec![],
            output_name: format!("tanh_out_{}", next_id()),
        })
    }

    pub fn reshape<TargetShape: Shape>(&self) -> Expr<TargetShape> {
        assert_eq!(
            S::numel(),
            TargetShape::numel(),
            "Reshape error: source shape {:?} ({} elems) != target shape {:?} ({} elems)",
            S::dims(),
            S::numel(),
            TargetShape::dims(),
            TargetShape::numel()
        );
        let shape_i64: Vec<i64> = TargetShape::dims().into_iter().map(|d| d as i64).collect();
        let shape_node = Rc::new(AstNode::ConstantInt64 {
            name: format!("shape_const_{}", next_id()),
            shape: vec![shape_i64.len()],
            data: shape_i64,
        });

        Expr::from_node(AstNode::Op {
            op_type: "Reshape".to_string(),
            inputs: vec![Rc::clone(&self.node), shape_node],
            attrs: vec![],
            output_name: format!("reshape_out_{}", next_id()),
        })
    }

    /// `prost` でエンコードした `.onnx` バイナリを直接ファイルに保存する
    pub fn save_onnx<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let bytes = output_onnx_bytes(self);
        std::fs::write(path, bytes)
    }
}

// ============================================================================
// 4. 2階テンソル (行列: Expr2D<R, C>) 専用の演算
// ============================================================================
impl<const R: usize, const C: usize> Expr2D<R, C> {
    /// 2次元行列積: [R, C] @ [C, C2] -> [R, C2]
    pub fn matmul<const C2: usize>(&self, rhs: &Expr2D<C, C2>) -> Expr2D<R, C2> {
        Expr::from_node(AstNode::Op {
            op_type: "MatMul".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&rhs.node)],
            attrs: vec![],
            output_name: format!("matmul_2d_out_{}", next_id()),
        })
    }

    /// 行バイアスのブロードキャスト加算: [R, C] + [1, C] -> [R, C]
    pub fn add_bias(&self, bias: &Expr2D<1, C>) -> Expr2D<R, C> {
        Expr::from_node(AstNode::Op {
            op_type: "Add".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&bias.node)],
            attrs: vec![],
            output_name: format!("add_bias_2d_out_{}", next_id()),
        })
    }
}

// ============================================================================
// 5. 4階テンソル (NCHW: Expr4D<N, C, H, W>) 専用の演算
// ============================================================================
impl<const N: usize, const C: usize, const H: usize, const W: usize> Expr4D<N, C, H, W> {
    /// 4階バッチ行列積 (Batched MatMul): [N, C, H, W] @ [N, C, W, W2] -> [N, C, H, W2]
    pub fn matmul_4d<const W2: usize>(&self, rhs: &Expr4D<N, C, W, W2>) -> Expr4D<N, C, H, W2> {
        Expr::from_node(AstNode::Op {
            op_type: "MatMul".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&rhs.node)],
            attrs: vec![],
            output_name: format!("matmul_4d_out_{}", next_id()),
        })
    }

    /// チャネル方向バイアス加算: [N, C, H, W] + [1, C, 1, 1] -> [N, C, H, W]
    pub fn add_channel_bias(&self, bias: &Expr4D<1, C, 1, 1>) -> Expr4D<N, C, H, W> {
        Expr::from_node(AstNode::Op {
            op_type: "Add".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&bias.node)],
            attrs: vec![],
            output_name: format!("add_ch_bias_out_{}", next_id()),
        })
    }

    /// 最後の2軸 (H と W) の入れ替え (Transpose): [N, C, H, W] -> [N, C, W, H]
    pub fn transpose_hw(&self) -> Expr4D<N, C, W, H> {
        Expr::from_node(AstNode::Op {
            op_type: "Transpose".to_string(),
            inputs: vec![Rc::clone(&self.node)],
            attrs: vec![("perm".to_string(), AttrValue::Ints(vec![0, 1, 3, 2]))],
            output_name: format!("transpose_hw_out_{}", next_id()),
        })
    }

    /// 1x1 畳み込み (Pointwise Conv2d: [N, C, H, W] * [OUT_C, C, 1, 1] -> [N, OUT_C, H, W])
    pub fn conv1x1<const OUT_C: usize>(
        &self,
        weight: &Expr4D<OUT_C, C, 1, 1>,
    ) -> Expr4D<N, OUT_C, H, W> {
        Expr::from_node(AstNode::Op {
            op_type: "Conv".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&weight.node)],
            attrs: vec![
                ("kernel_shape".to_string(), AttrValue::Ints(vec![1, 1])),
                ("strides".to_string(), AttrValue::Ints(vec![1, 1])),
                ("pads".to_string(), AttrValue::Ints(vec![0, 0, 0, 0])),
            ],
            output_name: format!("conv1x1_out_{}", next_id()),
        })
    }

    /// 3x3 Same-Padding 畳み込み (stride=1, pad=1: [N, C, H, W] * [OUT_C, C, 3, 3] -> [N, OUT_C, H, W])
    pub fn conv3x3_same<const OUT_C: usize>(
        &self,
        weight: &Expr4D<OUT_C, C, 3, 3>,
    ) -> Expr4D<N, OUT_C, H, W> {
        Expr::from_node(AstNode::Op {
            op_type: "Conv".to_string(),
            inputs: vec![Rc::clone(&self.node), Rc::clone(&weight.node)],
            attrs: vec![
                ("kernel_shape".to_string(), AttrValue::Ints(vec![3, 3])),
                ("strides".to_string(), AttrValue::Ints(vec![1, 1])),
                ("pads".to_string(), AttrValue::Ints(vec![1, 1, 1, 1])),
            ],
            output_name: format!("conv3x3_out_{}", next_id()),
        })
    }
}

// ============================================================================
// 6. prost を用いた ONNX ModelProto 構築 & バイナリシリアライズ関数
// ============================================================================
pub fn build_model_proto<S: Shape>(expr: &Expr<S>) -> onnx_proto::ModelProto {
    use onnx_proto::*;

    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut inputs = Vec::new();

    fn visit(
        node: &AstNode,
        nodes: &mut Vec<NodeProto>,
        inits: &mut Vec<TensorProto>,
        inputs: &mut Vec<ValueInfoProto>,
        visited: &mut Vec<String>,
    ) -> String {
        match node {
            AstNode::Input { name, shape } => {
                if !visited.contains(name) {
                    visited.push(name.clone());
                    inputs.push(ValueInfoProto::new_tensor(name, 1, shape));
                }
                name.clone()
            }
            AstNode::Constant { name, shape, data } => {
                if !visited.contains(name) {
                    visited.push(name.clone());
                    inits.push(TensorProto {
                        name: name.clone(),
                        dims: shape.iter().map(|&d| d as i64).collect(),
                        data_type: 1, // FLOAT
                        float_data: data.clone(),
                        int64_data: vec![],
                        raw_data: None,
                    });
                }
                name.clone()
            }
            AstNode::ConstantInt64 { name, shape, data } => {
                if !visited.contains(name) {
                    visited.push(name.clone());
                    inits.push(TensorProto {
                        name: name.clone(),
                        dims: shape.iter().map(|&d| d as i64).collect(),
                        data_type: 7, // INT64
                        float_data: vec![],
                        int64_data: data.clone(),
                        raw_data: None,
                    });
                }
                name.clone()
            }
            AstNode::Op {
                op_type,
                inputs: in_nodes,
                attrs,
                output_name,
            } => {
                if !visited.contains(output_name) {
                    let in_names: Vec<String> = in_nodes
                        .iter()
                        .map(|n| visit(n, nodes, inits, inputs, visited))
                        .collect();
                    visited.push(output_name.clone());

                    let attributes = attrs
                        .iter()
                        .map(|(k, val)| match val {
                            AttrValue::Floats(fs) if fs.len() == 1 => AttributeProto {
                                name: k.clone(),
                                f: Some(fs[0]),
                                i: None,
                                floats: vec![],
                                ints: vec![],
                                r#type: 1,
                            },
                            AttrValue::Floats(fs) => AttributeProto {
                                name: k.clone(),
                                f: None,
                                i: None,
                                floats: fs.clone(),
                                ints: vec![],
                                r#type: 6,
                            },
                            AttrValue::Ints(is) => AttributeProto {
                                name: k.clone(),
                                f: None,
                                i: None,
                                floats: vec![],
                                ints: is.clone(),
                                r#type: 7,
                            },
                        })
                        .collect();

                    nodes.push(NodeProto {
                        input: in_names,
                        output: vec![output_name.clone()],
                        name: output_name.clone(),
                        op_type: op_type.clone(),
                        attribute: attributes,
                    });
                }
                output_name.clone()
            }
        }
    }

    let mut visited = Vec::new();
    let final_output_name = visit(
        &expr.node,
        &mut nodes,
        &mut initializers,
        &mut inputs,
        &mut visited,
    );

    let output_vi = ValueInfoProto::new_tensor(&final_output_name, 1, &S::dims());

    ModelProto {
        ir_version: 8,
        producer_name: "milkv-duos-matmul-demo".to_string(),
        graph: Some(GraphProto {
            node: nodes,
            name: "SimpleMatMulAddGraph".to_string(),
            initializer: initializers,
            input: inputs,
            output: vec![output_vi],
        }),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
    }
}

pub fn output_onnx_bytes<S: Shape>(expr: &Expr<S>) -> Vec<u8> {
    let model_proto = build_model_proto(expr);
    model_proto.encode_to_vec()
}

pub fn save_onnx_file<S: Shape, P: AsRef<Path>>(expr: &Expr<S>, path: P) -> std::io::Result<()> {
    expr.save_onnx(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python スクリプトの `main()` (`matmul_add.onnx`: Y = X @ W + B, [1, 4] -> [1, 4]) と
    /// まったく同じ ONNX モデルを構築して検証するテスト
    #[test]
    fn test_python_main_matmul_add() {
        // 1. 入力 X: [1, 4]
        let x = Expr2D::<1, 4>::input("X");

        // 2. 定数重み行列 W: [4, 4] (対角スケーリング行列 1, 2, 3, 4)
        let w = Expr2D::<4, 4>::constant_nd(
            "W",
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
        );

        // 3. 定数バイアス B: [1, 4] (10, 20, 30, 40)
        let b = Expr2D::<1, 4>::constant_nd("B", [[10.0, 20.0, 30.0, 40.0]]);

        // 4. 演算ノード定義: Y = (X @ W) + B
        let y: Expr2D<1, 4> = x
            .matmul(&w)
            .with_output_name("XW")
            .add(&b)
            .with_output_name("Y");

        // 5. prost で .onnx バイナリとしてファイルに保存 & デコード検証
        y.save_onnx("matmul_add.onnx").unwrap();

        let bytes = std::fs::read("matmul_add.onnx").unwrap();
        let decoded = onnx_proto::ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = decoded.graph.unwrap();

        assert_eq!(graph.name, "SimpleMatMulAddGraph");
        assert_eq!(graph.input[0].name, "X");
        assert_eq!(graph.output[0].name, "Y");
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[0].input, vec!["X", "W"]);
        assert_eq!(graph.node[0].output, vec!["XW"]);
        assert_eq!(graph.node[1].op_type, "Add");
        assert_eq!(graph.node[1].input, vec!["XW", "B"]);
        assert_eq!(graph.node[1].output, vec!["Y"]);

        println!(
            "[OK] Generated matmul_add.onnx ({} bytes) matching Python main()!",
            bytes.len()
        );
    }

    #[test]
    fn test_python_my_calc00_matmul_add2() {
        // 1. 定数重み行列 W: [3, 2]
        let w = Expr2D::<3, 2>::constant_nd(
            "W",
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
        );

        // 2. 入力行列 X: [2, 3]
        let x = Expr2D::<2, 3>::input("X");

        // 3. 定数バイアス B: [1, 3] (各行に 10, 20, 30 をブロードキャスト加算)
        let b = Expr2D::<1, 3>::constant_nd("B", [[10.0, 20.0, 30.0]]);

        // ❌ もし Python の node 定義のように `x.matmul(&w)` ([2,3] @ [3,2] -> [2,2]) に
        //    `b` ([1,3]) を足そうとすると、以下のコメントを外した瞬間に LSP が赤線を出します！
        // let broken = x.matmul(&w).add_bias(&b); // <- Compile Error: expected `3`, found `2`

        // ✅ Python の `y_expected = w_data @ x_test + b_data` と同じく W([3,2]) @ X([2,3]) + B([1,3]) -> Y([3,3])
        let y: Expr2D<3, 3> = w
            .matmul(&x)
            .with_output_name("XW")
            .add_bias(&b)
            .with_output_name("Y");

        // 4. prost で `matmul_add2.onnx` を出力 & 検証
        y.save_onnx("matmul_add2.onnx").unwrap();

        let bytes = std::fs::read("matmul_add2.onnx").unwrap();
        let decoded = onnx_proto::ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = decoded.graph.unwrap();

        // 出力 Y の形状が [3, 3] になっていることを確認
        let out_dims: Vec<i64> = graph.output[0]
            .r#type
            .as_ref()
            .unwrap()
            .tensor_type
            .as_ref()
            .unwrap()
            .shape
            .as_ref()
            .unwrap()
            .dim
            .iter()
            .map(|d| d.dim_value)
            .collect();
        assert_eq!(out_dims, vec![3, 3]);

        println!(
            "[OK] Generated matmul_add2.onnx ({} bytes, output shape: {:?}) matching my_calc00()!",
            bytes.len(),
            out_dims
        );
    }
}
