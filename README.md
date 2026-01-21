# Llama 3.1 CLI Chat (Blackwell Optimized)

這是一個專為 NVIDIA Blackwell 架構 (RTX 50 系列) 優化的 Llama 3.1 8B Instruct 獨立命令列聊天工具。它利用 TensorRT-LLM 實現了原生的 NVFP4 權重與 FP8 KV Cache 加速，提供極致的本地推論效能。

## 特色

*   **模型**: meta-llama/Meta-Llama-3.1-8B-Instruct
*   **優化**:
    *   **權重**: 原生 NVFP4 (利用 Blackwell Tensor Cores)
    *   **KV Cache**: FP8 (大幅降低記憶體佔用)
    *   **長文本**: 支援 128k Context (啟用 `--multi_block_mode`)
*   **介面**: 純 Python CLI，支援即時串流輸出 (Streaming)
*   **語言**: 全繁體中文介面與說明

## 系統需求

*   **硬體**:
    *   GPU: NVIDIA RTX 5060 Ti (16GB VRAM) 或更高規格
    *   RAM: 建議 64GB 系統記憶體
*   **軟體**:
    *   Linux 環境
    *   NVIDIA Drivers & CUDA Toolkit
    *   **TensorRT-LLM** (必須預先安裝)
    *   `git-lfs`
    *   `conda` (建議)

## 安裝指南

本專案提供自動化安裝腳本，將自動完成模型下載、Checkpoint 轉換 (NVFP4) 以及 TensorRT 引擎建置。

1.  **複製專案** (假設您已擁有此程式碼)
2.  **執行安裝腳本**:

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

    *腳本將執行以下步驟：*
    *   檢查系統環境 (`git-lfs`, `conda`)
    *   從 Hugging Face 下載 Llama-3.1-8B-Instruct 模型
    *   將模型轉換為 NVFP4 格式
    *   建置支援 FP8 KV Cache 的 TensorRT 引擎

    *注意：首次建置可能需要數分鐘至數十分鐘。*

## 使用方法

安裝完成後，即可啟動聊天介面：

```bash
python3 chat.py
```

### 操作說明
*   **輸入提示詞**: 在 `你 >` 提示符後輸入您的問題。
*   **離開對話**: 輸入 `exit`、`quit` 或 `離開` 即可結束程式。

## 設定說明 (`config.yaml`)

您可以在 `config.yaml` 中調整基本設定：

```yaml
model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
dtype: "nvfp4"
kv_cache_dtype: "fp8"
max_context_len: 131072
# 若使用者能接受較慢的速度（PCIe 卸載），可切換至 70B 模型 ID。
```

## 檔案結構

*   `install.sh`: 自動化建置腳本
*   `chat.py`: 聊天主程式
*   `config.yaml`: 設定檔
*   `model_download/`: 模型下載目錄 (由腳本產生)
*   `checkpoint_output/`: 轉換後的 Checkpoint (由腳本產生)
*   `engine_output/`: 編譯完成的 TensorRT 引擎 (由腳本產生)

---
*由 Llama 3.1 驅動，針對 RTX 50 系列硬體進行最佳化。*
