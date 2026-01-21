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
    echo "警告：未檢測到 conda。正在自動安裝 Miniconda..."

    # 定義 Miniconda 安裝路徑
    MINICONDA_DIR="$HOME/miniconda"

    if [ -d "$MINICONDA_DIR" ]; then
        echo "檢測到 $MINICONDA_DIR 已存在，將使用此路徑。"
    else
        # 下載 Miniconda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        # 安裝
        bash miniconda.sh -b -p "$MINICONDA_DIR"
        rm miniconda.sh
        echo "Miniconda 安裝完成。"
    fi

    # 將 conda 加入環境變數
    export PATH="$MINICONDA_DIR/bin:$PATH"

    # 初始化 conda (即使在 script 中可能只影響當前 session)
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
else
    echo "檢測到 conda 環境。"
fi

# 檢查並安裝必要的系統依賴 (OpenMPI)
if ! command -v mpicc &> /dev/null; then
    echo "警告：未檢測到 mpicc (OpenMPI)。TensorRT-LLM 需要 MPI 支援。"
    echo "請執行：sudo apt-get -y install libopenmpi-dev"
    # 在非互動環境中，我們僅提示而不強制中止，因為某些環境可能已透過其他方式配置
fi

# 確保 transformers 版本足夠新以支援 Llama 3.1 模板
echo "正在檢查並升級 transformers..."
pip install --upgrade transformers

# 檢查並安裝 TensorRT-LLM 及相關依賴 (根據 Blackwell/CUDA 13.0 官方文件)
if ! python3 -c "import tensorrt_llm" &> /dev/null; then
    echo "未檢測到 TensorRT-LLM，正在嘗試自動安裝..."
    echo "注意：這將安裝 CUDA 13.0 相容的 PyTorch (2.9.0) 與 TensorRT-LLM。"

    # 安裝 PyTorch (CUDA 13.0)
    pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130 || echo "警告：PyTorch 2.9.0 安裝失敗，可能是預覽版本尚未公開。將嘗試繼續..."

    # 安裝 TensorRT-LLM
    # 先升級 pip
    pip install --upgrade pip setuptools
    pip install tensorrt_llm
fi

# 鎖定 Python 執行檔路徑
PYTHON_CMD=$(which python3)

echo "--------------------------------------------------------"
echo "確認 Python 環境"
echo "--------------------------------------------------------"
echo "正在使用的 Python 執行檔路徑: $PYTHON_CMD"
$PYTHON_CMD --version
# 更多進階設定可參考 TensorRT-LLM 官方文件: https://github.com/NVIDIA/TensorRT-LLM

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

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "目錄 $CHECKPOINT_DIR 已存在，清除並重新轉換..."
    rm -rf "$CHECKPOINT_DIR"
fi

# 尋找 convert_checkpoint.py
# 嘗試使用 python 模組執行 (新版)
if $PYTHON_CMD -c "import tensorrt_llm.commands.convert_checkpoint" &> /dev/null; then
    CMD="$PYTHON_CMD -m tensorrt_llm.commands.convert_checkpoint"
elif [ -f "convert_checkpoint.py" ]; then
    CMD="$PYTHON_CMD convert_checkpoint.py"
else
    echo "警告：找不到 convert_checkpoint.py。將嘗試假設它在 PATH 中或使用 'convert_checkpoint.py'。"
    echo "若失敗，請將 TensorRT-LLM examples/llama/convert_checkpoint.py 複製到此目錄。"
    CMD="$PYTHON_CMD convert_checkpoint.py"
fi

echo "執行轉換指令 (使用 $PYTHON_CMD)..."
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
    # 提醒使用者如果是新安裝的 conda，需要安裝套件
    echo "提示：若您剛安裝 Miniconda，請記得建立環境並安裝 TensorRT-LLM。"
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
echo "請執行 '$PYTHON_CMD chat.py' 開始對話。"
echo "--------------------------------------------------------"
