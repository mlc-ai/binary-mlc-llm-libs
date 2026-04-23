import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import json

LOG_PATH = Path("./") / "compile_wasm_log.txt"
# NOTE(Harry): Set this to your binary-mlc-llm-libs repo.
BINARY_DIR = "/path/to/binary-mlc-llm-libs/web-llm-models/v0_2_80"
CONFIG_PATH = "/path/to/hf-configs/"

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
    prefill_chunk_size,
    model_id,
    conv_template="LM",
    repo_id=None,
    use_sliding_window=False,
    max_batch_size=None,
    enable_subgroups=False,
    context_window_size=None,
):
    prefill_chunk_size = int(prefill_chunk_size)
    if context_window_size is not None:
        context_window_size = int(context_window_size)

    if use_sliding_window and context_window_size is None:
        raise ValueError("context_window_size must be specified when use_sliding_window=True")

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
        if repo_id:
            HF_TOKEN = os.getenv("HF_TOKEN")
            try:
                cfg_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                    token=HF_TOKEN,
                )
            except Exception as err:
                raise RuntimeError(f"Failed to download config.json for repo_id={repo_id}") from err

            os.makedirs(CONFIG_PATH, exist_ok=True)
            dst = f"{CONFIG_PATH}{repo_id.split('/')[-1]}.config.json"
            with open(cfg_path, "r", encoding="utf-8") as src, open(dst, "w", encoding="utf-8") as out:
                json.dump(json.load(src), out, indent=2, ensure_ascii=False)

            cmd = [
                sys.executable,
                "-m",
                "mlc_llm",
                "gen_config",
                dst,
                "--output",
                "dist/temp",
                "--conv-template",
                conv_template,
                "--quantization",
                quantization,
                "--prefill-chunk-size",
                str(prefill_chunk_size),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "mlc_llm",
                "gen_config",
                model,
                "--output",
                "dist/temp",
                "--conv-template",
                conv_template,
                "--quantization",
                quantization,
                "--prefill-chunk-size",
                str(prefill_chunk_size),
            ]

        if context_window_size is not None:
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
        if context_window_size is not None:
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

        if context_window_size is not None:
            if use_sliding_window:
                output_file_name = f"{model_id}-{quantization}-sw{ctx}_cs{cs}"
            else:
                output_file_name = f"{model_id}-{quantization}-ctx{ctx}_cs{cs}"
        else:
            output_file_name = f"{model_id}-{quantization}_cs{cs}"
        if max_batch_size:
            output_file_name += f"_batch{max_batch_size}"
        output_file_name += "-webgpu.wasm"
        output_dir_name = "sg32" if enable_subgroups else "base"
        output_path = os.path.join(BINARY_DIR, output_dir_name, output_file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
        if enable_subgroups:
            cmd += ["--enable-subgroups"]
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


def compile_variants(*args, **kwargs):
    kwargs.pop("enable_subgroups", None)
    compile(*args, **kwargs, enable_subgroups=False)
    compile(*args, **kwargs, enable_subgroups=True)


# NOTE(Charlie): As of 03/31/2025, the context window size does not do anything because
# it has become a runtime thing in both MLC-LLM and WebLLM.

# NOTE(Harry): To compile a wasm, uncomment the corresponding line below.

compile_variants("phi-3", "q4f16_1", 1024, "Phi-3-mini-4k-instruct", repo_id="microsoft/Phi-3-mini-4k-instruct")
compile_variants("phi-3", "q4f32_1", 1024, "Phi-3-mini-4k-instruct", repo_id="microsoft/Phi-3-mini-4k-instruct")
compile_variants("llama3_8b", "q4f16_1", 1024, "Llama-3-8B-Instruct", repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
compile_variants("llama3_8b", "q4f32_1", 1024, "Llama-3-8B-Instruct", repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
compile_variants("llama2_7b", "q4f16_1", 1024, "Llama-2-7b-chat-hf")
compile_variants("llama2_7b", "q4f32_1", 1024, "Llama-2-7b-chat-hf")
compile_variants("llama2_13b", "q4f16_1", 1024, "Llama-2-13b-chat-hf")
compile_variants("mistral_7b_v03", "q4f16_1", 1024, "Mistral-7B-Instruct-v0.3")
compile_variants("mistral_7b_v03", "q4f32_1", 1024, "Mistral-7B-Instruct-v0.3")
compile_variants("redpajama_3b_v1", "q4f16_1", 1024, "RedPajama-INCITE-Chat-3B-v1")
compile_variants("redpajama_3b_v1", "q4f32_1", 1024, "RedPajama-INCITE-Chat-3B-v1")
compile_variants("tinyllama_1b_chat_v0.4", "q0f16", 1024, "TinyLlama-1.1B-Chat-v0.4")
compile_variants("tinyllama_1b_chat_v0.4", "q0f32", 1024, "TinyLlama-1.1B-Chat-v0.4")
compile_variants("tinyllama_1b_chat_v0.4", "q4f16_1", 1024, "TinyLlama-1.1B-Chat-v0.4")
compile_variants("tinyllama_1b_chat_v0.4", "q4f32_1", 1024, "TinyLlama-1.1B-Chat-v0.4")
compile_variants("tinyllama_1b_chat_v1.0", "q4f16_1", 1024, "TinyLlama-1.1B-Chat-v1.0")
compile_variants("tinyllama_1b_chat_v1.0", "q4f32_1", 1024, "TinyLlama-1.1B-Chat-v1.0")
compile_variants("gemma_2b", "q4f16_1", 1024, "gemma-2b-it")
compile_variants("gemma_2b", "q4f32_1", 1024, "gemma-2b-it")
compile_variants("gpt2_medium", "q0f16", 1024, "gpt2-medium")
compile_variants("gpt2", "q0f16", 1024, "gpt2")
compile_variants("phi-1_5", "q4f16_1", 1024, "phi-1_5")
compile_variants("phi-1_5", "q4f32_1", 1024, "phi-1_5")
compile_variants("phi-2", "q4f16_1", 1024, "phi-2")
compile_variants("phi-2", "q4f32_1", 1024, "phi-2")
compile_variants("stablelm-2-zephyr-1_6b", "q4f16_1", 1024, "stablelm-2-zephyr-1_6b")
compile_variants("stablelm-2-zephyr-1_6b", "q4f32_1", 1024, "stablelm-2-zephyr-1_6b")
compile_variants("qwen2_0_5b", "q4f16_1", 1024, "Qwen2-0.5B-Instruct")
compile_variants("qwen2_0_5b", "q4f32_1", 1024, "Qwen2-0.5B-Instruct")
compile_variants("qwen2_0_5b", "q0f16", 1024, "Qwen2-0.5B-Instruct")
compile_variants("qwen2_0_5b", "q0f32", 1024, "Qwen2-0.5B-Instruct")
compile_variants("qwen2_1_5b", "q4f16_1", 1024, "Qwen2-1.5B-Instruct")
compile_variants("qwen2_1_5b", "q4f32_1", 1024, "Qwen2-1.5B-Instruct")
compile_variants("qwen2.5_3b", "q4f16_1", 1024, "Qwen2.5-3B-Instruct")
compile_variants("qwen2.5_3b", "q4f32_1", 1024, "Qwen2.5-3B-Instruct")
compile_variants("qwen2_7b", "q4f16_1", 1024, "Qwen2-7B-Instruct")
compile_variants("qwen2_7b", "q4f32_1", 1024, "Qwen2-7B-Instruct")
compile_variants("llama3_70b", "q3f16_1", 1024, "Llama-3-70B-Instruct", repo_id="meta-llama/Meta-Llama-3-70B-Instruct")
compile_variants("llama3_1_8b", "q4f16_1", 1024, "Llama-3_1-8B-Instruct")
compile_variants("llama3_1_8b", "q4f32_1", 1024, "Llama-3_1-8B-Instruct")
compile_variants("llama3_1_70b", "q3f16_1", 1024, "Llama-3_1-70B-Instruct")
compile_variants("gemma2_2b", "q4f16_1", 1024, "gemma-2-2b-it")
compile_variants("gemma2_2b", "q4f32_1", 1024, "gemma-2-2b-it")
compile_variants("gemma2_9b", "q4f16_1", 1024, "gemma-2-9b-it")
compile_variants("gemma2_9b", "q4f32_1", 1024, "gemma-2-9b-it")
compile_variants("snowflake-arctic-embed-m", "q0f32", 512, "snowflake-arctic-embed-m", max_batch_size=32, context_window_size=512)
compile_variants("snowflake-arctic-embed-m", "q0f32", 512, "snowflake-arctic-embed-m", max_batch_size=4, context_window_size=512)
compile_variants("snowflake-arctic-embed-s", "q0f32", 512, "snowflake-arctic-embed-s", max_batch_size=32, context_window_size=512, repo_id="Snowflake/snowflake-arctic-embed-s")
compile_variants("snowflake-arctic-embed-s", "q0f32", 512, "snowflake-arctic-embed-s", max_batch_size=4, context_window_size=512, repo_id="Snowflake/snowflake-arctic-embed-s")
compile_variants("phi-3_5", "q4f16_1", 1024, "Phi-3.5-mini-instruct")
compile_variants("phi-3_5", "q4f32_1", 1024, "Phi-3.5-mini-instruct")
compile_variants("phi-3_5-vision", "q4f16_1", 2048, "Phi-3.5-vision-instruct")
compile_variants("phi-3_5-vision", "q4f32_1", 2048, "Phi-3.5-vision-instruct")
compile_variants("llama3_2_1b", "q0f16", 1024, "Llama-3.2-1B-Instruct")
compile_variants("llama3_2_1b", "q0f32", 1024, "Llama-3.2-1B-Instruct")
compile_variants("llama3_2_1b", "q4f16_1", 1024, "Llama-3.2-1B-Instruct")
compile_variants("llama3_2_1b", "q4f32_1", 1024, "Llama-3.2-1B-Instruct")
compile_variants("llama3_2_3b", "q4f16_1", 1024, "Llama-3.2-3B-Instruct")
compile_variants("llama3_2_3b", "q4f32_1", 1024, "Llama-3.2-3B-Instruct")
compile_variants("gemma2_2b-jpn", "q4f16_1", 1024, "gemma-2-2b-jpn-it")
compile_variants("gemma2_2b-jpn", "q4f32_1", 1024, "gemma-2-2b-jpn-it")
compile_variants("smollm2_1_7b", "q0f16", 1024, "SmolLM2-1.7B-Instruct", repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct")
compile_variants("smollm2_1_7b", "q4f16_1", 1024, "SmolLM2-1.7B-Instruct", repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct")
compile_variants("smollm2_1_7b", "q4f32_1", 1024, "SmolLM2-1.7B-Instruct", repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct")
compile_variants("smollm2_360m", "q0f16", 1024, "SmolLM2-360M-Instruct")
compile_variants("smollm2_360m", "q0f32", 1024, "SmolLM2-360M-Instruct")
compile_variants("smollm2_360m", "q4f16_1", 1024, "SmolLM2-360M-Instruct")
compile_variants("smollm2_360m", "q4f32_1", 1024, "SmolLM2-360M-Instruct")
compile_variants("smollm2_135m", "q0f16", 1024, "SmolLM2-135M-Instruct")
compile_variants("smollm2_135m", "q0f32", 1024, "SmolLM2-135M-Instruct")
compile_variants("smollm2_135m", "q4f16_1", 1024, "SmolLM2-135M-Instruct")
compile_variants("smollm2_135m", "q4f32_1", 1024, "SmolLM2-135M-Instruct")
compile_variants("gemma3_1b_it", "q4f16_1", 1024, "gemma3-1b-it")
compile_variants("qwen3_0.6b", "q4f16_1", 1024, "Qwen3-0.6B")
compile_variants("qwen3_0.6b", "q4f32_1", 1024, "Qwen3-0.6B")
compile_variants("qwen3_0.6b", "q0f16", 1024, "Qwen3-0.6B")
compile_variants("qwen3_0.6b", "q0f32", 1024, "Qwen3-0.6B")
compile_variants("qwen3_1.7b", "q4f16_1", 1024, "Qwen3-1.7B")
compile_variants("qwen3_1.7b", "q4f32_1", 1024, "Qwen3-1.7B")
compile_variants("qwen3_4b", "q4f16_1", 1024, "Qwen3-4B", repo_id="Qwen/Qwen3-4B")
compile_variants("qwen3_4b", "q4f32_1", 1024, "Qwen3-4B", repo_id="Qwen/Qwen3-4B")
compile_variants("qwen3_8b", "q4f16_1", 1024, "Qwen3-8B", repo_id="Qwen/Qwen3-8B")
compile_variants("qwen3_8b", "q4f32_1", 1024, "Qwen3-8B", repo_id="Qwen/Qwen3-8B")
compile_variants("qwen3_4b_instruct_2507", "q4f16_1", 1024, "Qwen3-4B-Instruct-2507", repo_id="Qwen/Qwen3-4B-Instruct-2507")
compile_variants("qwen3_4b_instruct_2507", "q4f32_1", 1024, "Qwen3-4B-Instruct-2507", repo_id="Qwen/Qwen3-4B-Instruct-2507")
compile_variants(f"{CONFIG_PATH}Qwen3.5-0.8B.config.json", "q4f16_1", 1024, "Qwen3.5-0.8B", repo_id="Qwen/Qwen3.5-0.8B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-0.8B.config.json", "q4f32_1", 1024, "Qwen3.5-0.8B", repo_id="Qwen/Qwen3.5-0.8B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-0.8B.config.json", "q0f16", 1024, "Qwen3.5-0.8B", repo_id="Qwen/Qwen3.5-0.8B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-2B.config.json", "q4f16_1", 1024, "Qwen3.5-2B", repo_id="Qwen/Qwen3.5-2B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-2B.config.json", "q4f32_1", 1024, "Qwen3.5-2B", repo_id="Qwen/Qwen3.5-2B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-4B.config.json", "q4f16_1", 1024, "Qwen3.5-4B", repo_id="Qwen/Qwen3.5-4B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-4B.config.json", "q4f32_1", 1024, "Qwen3.5-4B", repo_id="Qwen/Qwen3.5-4B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-9B.config.json", "q4f16_1", 1024, "Qwen3.5-9B", repo_id="Qwen/Qwen3.5-9B")
compile_variants(f"{CONFIG_PATH}Qwen3.5-9B.config.json", "q4f32_1", 1024, "Qwen3.5-9B", repo_id="Qwen/Qwen3.5-9B")
compile_variants(f"{CONFIG_PATH}Phi-4-mini-instruct.config.json", "q4f16_1", 1024, "Phi-4-mini-instruct", repo_id="microsoft/Phi-4-mini-instruct")
compile_variants(f"{CONFIG_PATH}Phi-4-mini-instruct.config.json", "q4f32_1", 1024, "Phi-4-mini-instruct", repo_id="microsoft/Phi-4-mini-instruct")
compile_variants(f"{CONFIG_PATH}OLMo-2-0425-1B-Instruct.config.json", "q4f16_1", 1024, "OLMo-2-0425-1B-Instruct", repo_id="allenai/OLMo-2-0425-1B-Instruct")
compile_variants(f"{CONFIG_PATH}OLMo-2-0425-1B-Instruct.config.json", "q4f32_1", 1024, "OLMo-2-0425-1B-Instruct", repo_id="allenai/OLMo-2-0425-1B-Instruct")
compile_variants(f"{CONFIG_PATH}OLMo-2-1124-7B-Instruct.config.json", "q4f16_1", 1024, "OLMo-2-1124-7B-Instruct", repo_id="allenai/OLMo-2-1124-7B-Instruct")
compile_variants(f"{CONFIG_PATH}OLMo-2-1124-7B-Instruct.config.json", "q4f32_1", 1024, "OLMo-2-1124-7B-Instruct", repo_id="allenai/OLMo-2-1124-7B-Instruct")
compile_variants("ministral3_3b_2512", "q4f16_1", 1024, "Ministral-3-3B-Base-2512")
compile_variants("ministral3_3b_2512", "q4f32_1", 1024, "Ministral-3-3B-Base-2512")
compile_variants("ministral3_3b_2512", "q4f16_1", 1024, "Ministral-3-3B-Instruct-2512-BF16")
compile_variants("ministral3_3b_2512", "q4f32_1", 1024, "Ministral-3-3B-Instruct-2512-BF16")
compile_variants("ministral3_3b_2512", "q4f16_1", 1024, "Ministral-3-3B-Reasoning-2512")
compile_variants("ministral3_3b_2512", "q4f32_1", 1024, "Ministral-3-3B-Reasoning-2512")
compile_variants("qwen3_4b_thinking_2507", "q4f16_1", 1024, "Qwen3-4B-Thinking-2507", repo_id="Qwen/Qwen3-4B-Thinking-2507")
compile_variants("qwen3_4b_thinking_2507", "q4f32_1", 1024, "Qwen3-4B-Thinking-2507", repo_id="Qwen/Qwen3-4B-Thinking-2507")
