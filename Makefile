# Milk-V Duo S (SG2000 / cv181x) TPU 行列演算ビルドパイプライン
# ディレクトリ構成:
#   ./Makefile               (このファイル)
#   ./generate_matmul_onnx.py (uv 用 ONNX モデル生成スクリプト)
#   ./host-tools/            (RISC-V musl クロスコンパイルツールチェーン)
#   ./tpu-sdk-sg200x/        (SG200X 用 TPU ランタイム SDK)
#   ./tpu-mlir/              (Docker 内で使用する TPU-MLIR リポジトリ)
#   ./tpu_matmul/            (Rust クレート)

SHELL := /usr/bin/env bash

# ===== 設定変数 =====
RUST_TARGET       := riscv64gc-unknown-linux-musl
HOST_TOOLS_URL    := https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/03/07/16/host-tools.tar.gz
TPU_SDK_REPO      := https://github.com/milkv-duo/tpu-sdk-sg200x.git
TPU_MLIR_REPO     := https://github.com/milkv-duo/tpu-mlir.git
DOCKER_IMAGE      := sophgo/tpuc_dev:v3.1

# パス定義
ROOT_DIR          := $(CURDIR)
HOST_TOOLS_DIR    := $(ROOT_DIR)/host-tools
GCC_BIN           := $(HOST_TOOLS_DIR)/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gcc
TPU_SDK_DIR       := $(ROOT_DIR)/tpu-sdk-sg200x
TPU_MLIR_DIR      := $(ROOT_DIR)/tpu-mlir
RUST_DIR          := $(ROOT_DIR)/tpu_matmul
MODEL_BUILD_DIR   := $(ROOT_DIR)/build_matmul

# コンテナ実行時の UID/GID (root 権限のファイルが残るのを防ぐ)
HOST_UID          := $(shell id -u)
HOST_GID          := $(shell id -g)

# 成果物ファイル
ONNX_SCRIPT       := $(ROOT_DIR)/generate_matmul_onnx.py
ONNX_FILE         := $(ROOT_DIR)/matmul_add.onnx
INPUT_NPZ         := $(ROOT_DIR)/matmul_in.npz
CVIMODEL_FILE     := $(MODEL_BUILD_DIR)/matmul_add_bf16.cvimodel
RUST_BIN          := $(RUST_DIR)/target/$(RUST_TARGET)/release/tpu_matmul

# 実機転送用設定 (例: make deploy DUO_IP=192.168.1.50)
DUO_USER          ?= root
DUO_IP            ?= 192.168.1.20
DUO_DEST          ?= /root

.PHONY: all deps onnx cvimodel rust setup-cargo deploy clean clean-all help

# デフォルトターゲット: ツールチェーン・SDK準備 -> ONNX生成 -> cvimodelコンパイル -> Rustビルド
all: deps cvimodel rust
	@echo "=========================================="
	@echo "[OK] すべてのビルドが完了しました！"
	@echo "  - cvimodel : $(CVIMODEL_FILE)"
	@echo "  - Rust Bin : $(RUST_BIN)"
	@echo "=========================================="

help:
	@echo "使い方:"
	@echo "  make all                   : ツールチェーン取得・cvimodel変換・Rustバイナリ生成を一括実行"
	@echo "  make deps                  : host-tools, tpu-sdk-sg200x, rust target のインストール"
	@echo "  make onnx                  : uv run で ONNX モデルとテスト入力 (.npz) を生成"
	@echo "  make cvimodel              : Docker (sophgo/tpuc_dev) で ONNX から .cvimodel をコンパイル"
	@echo "  make rust                  : tpu_matmul の Rust クロスコンパイルを実行"
	@echo "  make deploy DUO_IP=&lt;IP&gt;    : 生成したバイナリと .cvimodel を Milk-V Duo S へ scp 転送"
	@echo "  make clean                 : ビルド生成物 (target, build_matmul, onnx) を削除"
	@echo "  make clean-all             : ダウンロードした SDK・ツールチェーンも含めて初期化"

# ===== 1. ツールチェーンと SDK の取得 =====
deps: $(GCC_BIN) $(TPU_SDK_DIR)/include/cviruntime.h
	rustup target add $(RUST_TARGET)

$(GCC_BIN):
	@echo "[1/4] host-tools をダウンロード・展開しています..."
	wget -c $(HOST_TOOLS_URL) -O host-tools.tar.gz
	tar xvf host-tools.tar.gz
	rm -f host-tools.tar.gz

$(TPU_SDK_DIR)/include/cviruntime.h:
	@echo "[1/4] tpu-sdk-sg200x をクローンしています..."
	git clone --depth 1 $(TPU_SDK_REPO) $(TPU_SDK_DIR)

$(TPU_MLIR_DIR)/envsetup.sh:
	@echo "[2/4] tpu-mlir をクローンしています..."
	git clone --depth 1 $(TPU_MLIR_REPO) $(TPU_MLIR_DIR)

# ===== 2. ONNX モデルとテスト入力 (.npz) の生成 (uv 使用) =====
onnx: $(ONNX_FILE) $(INPUT_NPZ)

$(ONNX_FILE) $(INPUT_NPZ): $(ONNX_SCRIPT)
	@echo "[2/4] uv で ONNX モデル (matmul_add.onnx) と入力データ (matmul_in.npz) を生成しています..."
	uv run $(ONNX_SCRIPT)

# ===== 3. TPU-MLIR コンテナによる .cvimodel のコンパイル =====
#   - PYTHONDONTWRITEBYTECODE=1 で tpu-mlir 内に root 所有の __pycache__ が作られるのを防止
#   - 終了時 (trap EXIT) に build_matmul の所有権を現在のユーザー (HOST_UID:HOST_GID) に戻す
cvimodel: $(CVIMODEL_FILE)

$(CVIMODEL_FILE): $(ONNX_FILE) $(INPUT_NPZ) $(TPU_MLIR_DIR)/envsetup.sh
	@echo "[3/4] Docker ($(DOCKER_IMAGE)) 内で .cvimodel をコンパイルしています..."
	mkdir -p $(MODEL_BUILD_DIR)
	docker run --rm --privileged \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-v "$(ROOT_DIR):/workspace" \
		-w /workspace/build_matmul \
		$(DOCKER_IMAGE) /bin/bash -c "\
			set -e && \
			trap 'chown -R $(HOST_UID):$(HOST_GID) /workspace/build_matmul /workspace/tpu-mlir 2>/dev/null || true' EXIT && \
			source /workspace/tpu-mlir/envsetup.sh && \
			model_transform.py \
				--model_name matmul_add \
				--model_def ../matmul_add.onnx \
				--input_shapes [[1,4]] \
				--test_input ../matmul_in.npz \
				--test_result matmul_top_outputs.npz \
				--mlir matmul_add.mlir && \
			model_deploy.py \
				--mlir matmul_add.mlir \
				--quantize BF16 \
				--chip cv181x \
				--test_input ../matmul_in.npz \
				--test_reference matmul_top_outputs.npz \
				--model matmul_add_bf16.cvimodel \
		"

# ===== 4. Rust クレート (tpu_matmul) の環境設定とクロスビルド =====
setup-cargo: $(GCC_BIN) $(TPU_SDK_DIR)/include/cviruntime.h
	@mkdir -p $(RUST_DIR)/.cargo/bin
	@RUST_LLD="$$(find "$$(rustc --print sysroot)/lib/rustlib" -name rust-lld | head -n 1)"; \
	printf '#!/usr/bin/env bash\nexec "%s" -flavor gnu "$$@"\n' "$$RUST_LLD" > $(RUST_DIR)/.cargo/bin/ld; \
	chmod +x $(RUST_DIR)/.cargo/bin/ld

rust: $(RUST_BIN)

$(RUST_BIN): setup-cargo $(RUST_DIR)/Cargo.toml $(RUST_DIR)/build.rs $(RUST_DIR)/src/main.rs
	@echo "[4/4] Rust プロジェクト (tpu_matmul) をクロスコンパイルしています..."
	cd $(RUST_DIR) && cargo build --release --target $(RUST_TARGET)

# ===== 5. Milk-V Duo S 実機への scp 転送 =====
deploy: $(RUST_BIN) $(CVIMODEL_FILE)
	@echo "[Deploy] $(DUO_USER)@$(DUO_IP):$(DUO_DEST) へ転送しています..."
	scp $(RUST_BIN) $(CVIMODEL_FILE) $(DUO_USER)@$(DUO_IP):$(DUO_DEST)/

# ===== クリーンアップ =====
# 万が一コンテナが root 所有で作ったファイルが残っていても、sudo を使わずに処理できるよう
# 軽量コンテナまたは通常の rm -rf で削除する
clean:
	rm -rf $(MODEL_BUILD_DIR) $(ONNX_FILE) $(INPUT_NPZ) 2>/dev/null || \
		docker run --rm -v "$(ROOT_DIR):/workspace" $(DOCKER_IMAGE) rm -rf /workspace/build_matmul
	cd $(RUST_DIR) && cargo clean

clean-all: clean
	rm -rf $(HOST_TOOLS_DIR) $(TPU_SDK_DIR) $(TPU_MLIR_DIR) host-tools.tar.gz 2>/dev/null || \
		docker run --rm -v "$(ROOT_DIR):/workspace" $(DOCKER_IMAGE) rm -rf /workspace/tpu-mlir /workspace/build_matmul
	rm -rf $(HOST_TOOLS_DIR) $(TPU_SDK_DIR) $(TPU_MLIR_DIR) host-tools.tar.gz
