#!/bin/bash

# 設定發生錯誤時停止腳本
set -e

# 定義變數
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR="model_download"
ENGINE_DIR="engine_output"
CHECKPOINT_DIR="checkpoint_output"

echo "========================================================"
echo " 開始安裝與建置 Llama 3.1 CLI (Blackwell NVFP4/FP8)"
echo "========================================================"

# 檢查 git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "錯誤：未安裝 git-lfs。請先安裝 git-lfs (例如: sudo apt-get install git-lfs)。"
    exit 1
fi

# 檢查 conda (或確認 python 環境)
if ! command -v conda &> /dev/null; then
    echo "警告：未檢測到 conda。請確保您在正確的 Python 虛擬環境中，並且已安裝 TensorRT-LLM。"
else
    echo "檢測到 conda 環境。"
fi

# 步驟 1: 下載模型
echo "--------------------------------------------------------"
echo "步驟 1: 下載模型 ($MODEL_ID)..."
echo "--------------------------------------------------------"

if [ -d "$MODEL_DIR" ]; then
    echo "目錄 $MODEL_DIR 已存在，跳過下載。"
else
    git lfs install
    git clone https://huggingface.co/$MODEL_ID $MODEL_DIR
    echo "模型下載完成。"
fi

# 步驟 2: 轉換 Checkpoint (NVFP4)
echo "--------------------------------------------------------"
echo "步驟 2: 轉換 Checkpoint 至 NVFP4..."
echo "--------------------------------------------------------"

# 尋找 convert_checkpoint.py
# 假設使用者在 TensorRT-LLM 環境中，嘗試尋找常見路徑或是依賴 PATH
# 為了穩定性，這裡假設腳本在 PATH 中或需要使用者指定
# 但根據題目要求，我們直接使用 convert_checkpoint.py 並假設環境已配置好 (如在 TRT-LLM docker 中)

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "目錄 $CHECKPOINT_DIR 已存在，清除並重新轉換..."
    rm -rf "$CHECKPOINT_DIR"
fi

# 檢查 convert_checkpoint.py 是否可直接執行 (透過 python -m tensorrt_llm.commands.convert_checkpoint 或類似)
# 較新版 TRT-LLM 推薦使用: python -m tensorrt_llm.commands.convert_checkpoint
# 或是 examples/llama/convert_checkpoint.py

# 嘗試使用 python 模組執行 (新版)
if python3 -c "import tensorrt_llm.commands.convert_checkpoint" &> /dev/null; then
    CMD="python3 -m tensorrt_llm.commands.convert_checkpoint"
elif [ -f "convert_checkpoint.py" ]; then
    CMD="python3 convert_checkpoint.py"
else
    echo "警告：找不到 convert_checkpoint.py。將嘗試假設它在 PATH 中或使用 'convert_checkpoint.py'。"
    echo "若失敗，請將 TensorRT-LLM examples/llama/convert_checkpoint.py 複製到此目錄。"
    CMD="python3 convert_checkpoint.py"
fi

echo "執行轉換指令..."
$CMD --model_dir $MODEL_DIR \
     --output_dir $CHECKPOINT_DIR \
     --dtype nvfp4 \
     --use_weight_only

echo "Checkpoint 轉換完成。"

# 步驟 3: 建置 TRT 引擎
echo "--------------------------------------------------------"
echo "步驟 3: 建置 TensorRT 引擎..."
echo "--------------------------------------------------------"

if ! command -v trtllm-build &> /dev/null; then
    echo "錯誤：找不到 trtllm-build 指令。請確保已安裝 TensorRT-LLM 並將其加入 PATH。"
    exit 1
fi

trtllm-build --checkpoint_dir $CHECKPOINT_DIR \
             --output_dir $ENGINE_DIR \
             --gemm_plugin nvfp4 \
             --kv_cache_type fp8 \
             --remove_input_padding \
             --multi_block_mode

echo "--------------------------------------------------------"
echo "安裝與建置完成！"
echo "請執行 'python3 chat.py' 開始對話。"
echo "--------------------------------------------------------"
