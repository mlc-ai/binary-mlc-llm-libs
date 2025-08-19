import os
import subprocess
import sys
from pathlib import Path

LOG_PATH = Path("./") / "compile_wasm_log.txt"
# NOTE(Harry): Set this to your binary-mlc-llm-libs repo.
BINARY_DIR = "/path/to/binary-mlc-llm-libs/web-llm-models/v0_2_80"

# -1. Clean log file
cmd = [
    "rm",
    "-rf",
    "./compile_wasm_log.txt",
]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True, stderr=subprocess.STDOUT, env=os.environ)


def compile(
    model,
    quantization,
    context_window_size,
    prefill_chunk_size,
    model_id,
    use_sliding_window=False,
    max_batch_size=None,
):
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        # 0. Clean temp folder
        cmd = [
            "rm",
            "-rf",
            "dist/temp/",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ)

        # 1. Gen config
        cmd = [
            sys.executable,
            "-m",
            "mlc_llm",
            "gen_config",
            model,
            "--output",
            "dist/temp",
            "--conv-template",
            "LM",
            "--quantization",
            quantization,
            "--prefill-chunk-size",
            str(prefill_chunk_size),
        ]
        if use_sliding_window:
            cmd += [
                "--sliding-window-size",
                str(context_window_size),
            ]
        else:
            cmd += [
                "--context-window-size",
                str(context_window_size),
            ]
        if max_batch_size:
            cmd += [
                "--max-batch-size",
                str(max_batch_size),
            ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ)

        # 2. compile

        # 2.1. Get output wasm name
        ctx = ""
        if context_window_size == 4096:
            ctx = "4k"
        elif context_window_size == 2048:
            ctx = "2k"
        elif context_window_size == 1024:
            ctx = "1k"
        elif context_window_size == 512:
            ctx = "512"
        else:
            raise RuntimeError(f"Unrecognized ctx: {ctx}")

        cs = ""
        if prefill_chunk_size == 4096:
            cs = "4k"
        elif prefill_chunk_size == 2048:
            cs = "2k"
        elif prefill_chunk_size == 1024:
            cs = "1k"
        elif prefill_chunk_size == 512:
            cs = "512"
        else:
            raise RuntimeError(f"Unrecognized cs: {cs}")

        # e.g. Llama-3-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm
        if use_sliding_window:
            output_file_name = f"{model_id}-{quantization}-sw{ctx}_cs{cs}"
        else:
            output_file_name = f"{model_id}-{quantization}-ctx{ctx}_cs{cs}"
        if max_batch_size:
            output_file_name += f"_batch{max_batch_size}"
        output_file_name += "-webgpu.wasm"
        output_path = os.path.join(BINARY_DIR, output_file_name)

        # 2.2. Compile
        cmd = [
            sys.executable,
            "-m",
            "mlc_llm",
            "compile",
            "dist/temp/mlc-chat-config.json",
            "--device",
            "webgpu",
            "--output",
            output_path,
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ)

        # 3. Clean temp mlc-chat-config.json
        cmd = [
            "rm",
            "-rf",
            "dist/temp/mlc-chat-config.json",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ)


# NOTE(Charlie): As of 03/31/2025, the context window size does not do anything because
# it has become a runtime thing in both MLC-LLM and WebLLM.

# NOTE(Harry): To compile a wasm, uncomment the corresponding line below.

# compile("phi-3", "q4f16_1", 4096, 1024, "Phi-3-mini-4k-instruct")
# compile("phi-3", "q4f16_1", 4096, 1024, "Phi-3-mini-4k-instruct-old")
# compile("phi-3", "q4f32_1", 4096, 1024, "Phi-3-mini-4k-instruct")

# compile("llama3_8b", "q4f16_1", 4096, 1024, "Llama-3-8B-Instruct")
# compile("llama3_8b", "q4f32_1", 4096, 1024, "Llama-3-8B-Instruct")

# compile("llama2_7b", "q4f16_1", 4096, 1024, "Llama-2-7b-chat-hf")
# compile("llama2_7b", "q4f32_1", 4096, 1024, "Llama-2-7b-chat-hf")

# compile("llama2_13b", "q4f16_1", 4096, 1024, "Llama-2-13b-chat-hf")

# compile("mistral_7b_v03", "q4f16_1", 4096, 1024, "Mistral-7B-Instruct-v0.3")
# compile("mistral_7b_v03", "q4f32_1", 4096, 1024, "Mistral-7B-Instruct-v0.3")

# compile("redpajama_3b_v1", "q4f16_1", 2048, 1024, "RedPajama-INCITE-Chat-3B-v1")
# compile("redpajama_3b_v1", "q4f32_1", 2048, 1024, "RedPajama-INCITE-Chat-3B-v1")

# compile("tinyllama_1b_chat_v0.4", "q0f16", 2048, 1024, "TinyLlama-1.1B-Chat-v0.4")
# compile("tinyllama_1b_chat_v0.4", "q0f32", 2048, 1024, "TinyLlama-1.1B-Chat-v0.4")
# compile("tinyllama_1b_chat_v0.4", "q4f16_1", 2048, 1024, "TinyLlama-1.1B-Chat-v0.4")
# compile("tinyllama_1b_chat_v0.4", "q4f32_1", 2048, 1024, "TinyLlama-1.1B-Chat-v0.4")

# compile("tinyllama_1b_chat_v1.0", "q4f16_1", 2048, 1024, "TinyLlama-1.1B-Chat-v1.0")
# compile("tinyllama_1b_chat_v1.0", "q4f32_1", 2048, 1024, "TinyLlama-1.1B-Chat-v1.0")

# compile("gemma_2b", "q4f16_1", 4096, 1024, "gemma-2b-it")
# compile("gemma_2b", "q4f32_1", 4096, 1024, "gemma-2b-it")

# compile("gpt2_medium", "q0f16", 1024, 1024, "gpt2-medium")
# compile("gpt2", "q0f16", 1024, 1024, "gpt2")

# compile("phi-1_5", "q4f16_1", 2048, 1024, "phi-1_5")
# compile("phi-1_5", "q4f32_1", 2048, 1024, "phi-1_5")

# compile("phi-2", "q4f16_1", 2048, 1024, "phi-2")
# compile("phi-2", "q4f32_1", 2048, 1024, "phi-2")

# compile("stablelm-2-zephyr-1_6b", "q4f16_1", 4096, 1024, "stablelm-2-zephyr-1_6b")
# compile("stablelm-2-zephyr-1_6b", "q4f32_1", 4096, 1024, "stablelm-2-zephyr-1_6b")

# compile("qwen2_0_5b", "q4f16_1", 4096, 1024, "Qwen2-0.5B-Instruct")
# compile("qwen2_0_5b", "q4f32_1", 4096, 1024, "Qwen2-0.5B-Instruct")
# compile("qwen2_0_5b", "q0f16", 4096, 1024, "Qwen2-0.5B-Instruct")
# compile("qwen2_0_5b", "q0f32", 4096, 1024, "Qwen2-0.5B-Instruct")

# compile("qwen2_1_5b", "q4f16_1", 4096, 1024, "Qwen2-1.5B-Instruct")
# compile("qwen2_1_5b", "q4f32_1", 4096, 1024, "Qwen2-1.5B-Instruct")

# compile("qwen2.5_3b", "q4f16_1", 4096, 1024, "Qwen2.5-3B-Instruct")
# compile("qwen2.5_3b", "q4f32_1", 4096, 1024, "Qwen2.5-3B-Instruct")

# compile("qwen2_7b", "q4f16_1", 4096, 1024, "Qwen2-7B-Instruct")
# compile("qwen2_7b", "q4f32_1", 4096, 1024, "Qwen2-7B-Instruct")

# compile("llama3_70b", "q3f16_1", 4096, 1024, "Llama-3-70B-Instruct")

# compile("llama3_1_8b", "q4f16_1", 4096, 1024, "Llama-3_1-8B-Instruct")
# compile("llama3_1_8b", "q4f32_1", 4096, 1024, "Llama-3_1-8B-Instruct")

# compile("llama3_1_70b", "q3f16_1", 4096, 1024, "Llama-3_1-70B-Instruct")

# compile("gemma2_2b", "q4f16_1", 4096, 1024, "gemma-2-2b-it")
# compile("gemma2_2b", "q4f32_1", 4096, 1024, "gemma-2-2b-it")

# compile("gemma2_9b", "q4f16_1", 4096, 1024, "gemma-2-9b-it")
# compile("gemma2_9b", "q4f32_1", 4096, 1024, "gemma-2-9b-it")

# # max_position_embeddings is set to be 512 here for embedding model; no rope is used
# compile(
#     "snowflake-arctic-embed-m", "q0f32", 512, 512, "snowflake-arctic-embed-m", max_batch_size=32
# )
# compile("snowflake-arctic-embed-m", "q0f32", 512, 512, "snowflake-arctic-embed-m", max_batch_size=4)

# compile(
#     "snowflake-arctic-embed-s", "q0f32", 512, 512, "snowflake-arctic-embed-s", max_batch_size=32
# )
# compile("snowflake-arctic-embed-s", "q0f32", 512, 512, "snowflake-arctic-embed-s", max_batch_size=4)

# # Cannot be shared with phi-3 because phi3.5 has rope scaling
# compile("phi-3_5", "q4f16_1", 4096, 1024, "Phi-3.5-mini-instruct")
# compile("phi-3_5", "q4f32_1", 4096, 1024, "Phi-3.5-mini-instruct")

# compile("phi-3_5-vision", "q4f16_1", 4096, 2048, "Phi-3.5-vision-instruct")
# compile("phi-3_5-vision", "q4f32_1", 4096, 2048, "Phi-3.5-vision-instruct")

# # Llama3.2 1B/3B
# compile("llama3_2_1b", "q0f16", 4096, 1024, "Llama-3.2-1B-Instruct")
# compile("llama3_2_1b", "q0f32", 4096, 1024, "Llama-3.2-1B-Instruct")
# compile("llama3_2_1b", "q4f16_1", 4096, 1024, "Llama-3.2-1B-Instruct")
# compile("llama3_2_1b", "q4f32_1", 4096, 1024, "Llama-3.2-1B-Instruct")

# compile("llama3_2_3b", "q4f16_1", 4096, 1024, "Llama-3.2-3B-Instruct")
# compile("llama3_2_3b", "q4f32_1", 4096, 1024, "Llama-3.2-3B-Instruct")

# compile("gemma2_2b-jpn", "q4f16_1", 4096, 1024, "gemma-2-2b-jpn-it")
# compile("gemma2_2b-jpn", "q4f32_1", 4096, 1024, "gemma-2-2b-jpn-it")

# compile("smollm2_1_7b", "q0f16", 4096, 1024, "SmolLM2-1.7B-Instruct")
# compile("smollm2_1_7b", "q4f16_1", 4096, 1024, "SmolLM2-1.7B-Instruct")
# compile("smollm2_1_7b", "q4f32_1", 4096, 1024, "SmolLM2-1.7B-Instruct")

# compile("smollm2_360m", "q0f16", 4096, 1024, "SmolLM2-360M-Instruct")
# compile("smollm2_360m", "q0f32", 4096, 1024, "SmolLM2-360M-Instruct")
# compile("smollm2_360m", "q4f16_1", 4096, 1024, "SmolLM2-360M-Instruct")
# compile("smollm2_360m", "q4f32_1", 4096, 1024, "SmolLM2-360M-Instruct")

# compile("smollm2_135m", "q0f16", 4096, 1024, "SmolLM2-135M-Instruct")
# compile("smollm2_135m", "q0f32", 4096, 1024, "SmolLM2-135M-Instruct")
# compile("smollm2_135m", "q4f16_1", 4096, 1024, "SmolLM2-135M-Instruct")
# compile("smollm2_135m", "q4f32_1", 4096, 1024, "SmolLM2-135M-Instruct")

# compile("gemma3_1b_it", "q4f16_1", 4096, 1024, "gemma3-1b-it")


# compile("qwen3_0.6b", "q4f16_1", 4096, 1024, "Qwen3-0.6B")
# compile("qwen3_0.6b", "q4f32_1", 4096, 1024, "Qwen3-0.6B")
# compile("qwen3_0.6b", "q0f16", 4096, 1024, "Qwen3-0.6B")
# compile("qwen3_0.6b", "q0f32", 4096, 1024, "Qwen3-0.6B")

# compile("qwen3_1.7b", "q4f16_1", 4096, 1024, "Qwen3-1.7B")
# compile("qwen3_1.7b", "q4f32_1", 4096, 1024, "Qwen3-1.7B")

# compile("qwen3_4b", "q4f16_1", 4096, 1024, "Qwen3-4B")
# compile("qwen3_4b", "q4f32_1", 4096, 1024, "Qwen3-4B")

# compile("qwen3_8b", "q4f16_1", 4096, 1024, "Qwen3-8B")
# compile("qwen3_8b", "q4f32_1", 4096, 1024, "Qwen3-8B")

# compile("qwen3_4b_instruct_2507", "q4f16_1", 4096, 1024, "Qwen3-4B-Instruct-2507")
# compile("qwen3_4b_instruct_2507", "q4f32_1", 4096, 1024, "Qwen3-4B-Instruct-2507")

# compile("qwen3_4b_thinking_2507", "q4f16_1", 4096, 1024, "Qwen3-4B-Thinking-2507")
compile("qwen3_4b_thinking_2507", "q4f32_1", 4096, 1024, "Qwen3-4B-Thinking-2507")

