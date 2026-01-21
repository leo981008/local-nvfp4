import argparse
import os
import sys
import yaml
import torch
from transformers import AutoTokenizer

# 嘗試匯入 TensorRT-LLM，若失敗則提示
try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner
except ImportError:
    print("錯誤：找不到 tensorrt_llm 套件。請確認環境已正確設定。", file=sys.stderr)
    sys.exit(1)

def load_config(config_path="config.yaml"):
    """載入設定檔"""
    if not os.path.exists(config_path):
        print(f"錯誤：找不到設定檔 {config_path}。", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 載入設定
    config = load_config()

    # 定義路徑 (需與 install.sh 一致)
    model_path = "./model_download"
    engine_dir = "./engine_output"

    # 檢查引擎是否存在
    if not os.path.exists(engine_dir):
        print("錯誤：找不到 TensorRT 引擎目錄 (./engine_output)。請先執行 install.sh。", file=sys.stderr)
        sys.exit(1)

    print("正在初始化模型與引擎，這可能需要一點時間...", file=sys.stderr)

    # 初始化 Tokenizer
    try:
        # 嘗試從本地下載目錄載入
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        print(f"警告：無法從 {model_path} 載入 Tokenizer，嘗試從 Hugging Face ({config['model_id']}) 下載...", file=sys.stderr)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
        except Exception as e:
            print(f"錯誤：無法載入 Tokenizer。{e}", file=sys.stderr)
            sys.exit(1)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 初始化 TRT-LLM Runner
    try:
        runner = ModelRunner.from_dir(engine_dir)
    except Exception as e:
        print(f"錯誤：無法載入 TensorRT-LLM 引擎。{e}", file=sys.stderr)
        sys.exit(1)

    # 系統提示詞 (System Prompt) - 保持英文
    system_prompt = "You are a helpful AI assistant running locally on RTX 50 series hardware."

    # 對話歷史
    history = [
        {"role": "system", "content": system_prompt}
    ]

    print("\n==========================================")
    print(" Llama 3.1 CLI Chat (Blackwell Optimized)")
    print(" 模型：Meta-Llama-3.1-8B-Instruct (NVFP4)")
    print(" 輸入 'exit', 'quit' 或 '離開' 結束對話")
    print("==========================================\n")

    while True:
        try:
            user_input = input("你 > ")
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if not user_input.strip():
            continue

        if user_input.lower() in ["exit", "quit", "離開"]:
            print("再見！")
            break

        # 新增使用者訊息到歷史
        history.append({"role": "user", "content": user_input})

        # 應用聊天模板
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                history,
                add_generation_prompt=True,
                tokenize=False
            )
        except Exception as e:
            print(f"錯誤：套用聊天模板失敗。{e}", file=sys.stderr)
            continue

        # 進行 Tokenize
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        batch_input_ids = [input_ids]

        print("Llama > ", end="", flush=True)

        # 串流生成
        try:
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=2048, # 可根據需求調整
                end_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                streaming=True
            )

            last_text_len = 0
            full_response = ""

            # 處理串流輸出
            # ModelRunner.generate yield GenerationResult
            for output in outputs:
                # output_ids shape: [batch, beam, len]
                # 取 batch 0, beam 0
                output_ids = output.output_ids[0][0]

                # 某些 runner 實現會包含 input_ids，某些只包含 new tokens
                # 我們假設它包含 input_ids，並將其移除
                # 需檢查 output_ids長度是否大於 input_ids長度
                if len(output_ids) > len(input_ids):
                    generated_ids = output_ids[len(input_ids):]
                else:
                    # 若長度相同或更短，可能還沒生成或是只有 input
                    generated_ids = []

                if len(generated_ids) > 0:
                    current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    # 計算增量並輸出
                    if len(current_text) > last_text_len:
                        diff = current_text[last_text_len:]
                        print(diff, end="", flush=True)
                        last_text_len = len(current_text)
                        full_response = current_text

            print() # 換行

            # 將回應加入歷史
            history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"\n錯誤：生成回應時發生錯誤。{e}", file=sys.stderr)

if __name__ == "__main__":
    main()
