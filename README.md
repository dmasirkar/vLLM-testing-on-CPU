# vLLM-testing-on-CPU

Requirements :
Python: 3.9 – 3.12


Reference :   https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html?device=x86#
create a new Python environment using uv, a very fast Python environment manager. Please follow the documentation to install uv. After installing uv, you can create a new Python environment using the following command:
# (Recommended) Create a new uv environment. Use `--seed` to install `pip` and `setuptools` in the environment.
uv venv vllm --python 3.12 --seed
source vllm/bin/activate


For Intel/AMD x86 :
First, install the recommended compiler. We recommend using gcc/g++ >= 12.3.0 as the default compiler to avoid potential problems. For example, on Ubuntu 24.10, you can run:
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

Second, clone vLLM project:

git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

Third, install Python packages for vLLM CPU backend building:

pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

Finally, build and install vLLM CPU backend:
VLLM_TARGET_DEVICE=cpu python setup.py install


Successfully installed MarkupSafe-3.0.2 aiohappyeyeballs-2.6.1 aiohttp-3.11.16 aiosignal-1.3.2 airportsdata-20250224 annotated-types-0.7.0 anyio-4.9.0 astor-0.8.1 attrs-25.3.0 blake3-1.0.4 cachetools-5.5.2 certifi-2025.1.31 charset-normalizer-3.4.1 click-8.1.8 cloudpickle-3.1.1 compressed-tensors-0.9.3 datasets-3.5.0 deprecated-1.2.18 depyf-0.18.0 dill-0.3.8 diskcache-5.6.3 distro-1.9.0 dnspython-2.7.0 einops-0.8.1 email-validator-2.2.0 fastapi-0.115.12 fastapi-cli-0.0.7 filelock-3.18.0 frozenlist-1.6.0 fsspec-2024.12.0 gguf-0.14.0 googleapis-common-protos-1.70.0 grpcio-1.71.0 h11-0.14.0 hf-xet-1.0.3 httpcore-1.0.8 httptools-0.6.4 httpx-0.28.1 huggingface-hub-0.30.2 idna-3.10 importlib_metadata-8.0.0 interegular-0.3.3 jinja2-3.1.6 jiter-0.9.0 jsonschema-4.23.0 jsonschema-specifications-2024.10.1 lark-1.2.2 llguidance-0.7.16 lm-format-enforcer-0.10.11 markdown-it-py-3.0.0 mdurl-0.1.2 mistral_common-1.5.4 mpmath-1.3.0 msgspec-0.19.0 multidict-6.4.3 multiprocess-0.70.16 nest_asyncio-1.6.0 networkx-3.4.2 openai-1.75.0 opencv-python-headless-4.11.0.86 opentelemetry-api-1.26.0 opentelemetry-exporter-otlp-1.26.0 opentelemetry-exporter-otlp-proto-common-1.26.0 opentelemetry-exporter-otlp-proto-grpc-1.26.0 opentelemetry-exporter-otlp-proto-http-1.26.0 opentelemetry-proto-1.26.0 opentelemetry-sdk-1.26.0 opentelemetry-semantic-conventions-0.47b0 opentelemetry-semantic-conventions-ai-0.4.3 outlines-0.1.11 outlines_core-0.1.26 pandas-2.2.3 partial-json-parser-0.2.1.1.post5 pillow-11.2.1 prometheus-fastapi-instrumentator-7.1.0 prometheus_client-0.21.1 propcache-0.3.1 protobuf-4.25.6 psutil-7.0.0 py-cpuinfo-9.0.0 pyarrow-19.0.1 pycountry-24.6.1 pydantic-2.11.3 pydantic-core-2.33.1 pygments-2.19.1 python-dateutil-2.9.0.post0 python-dotenv-1.1.0 python-json-logger-3.3.0 python-multipart-0.0.20 pytz-2025.2 pyyaml-6.0.2 pyzmq-26.4.0 referencing-0.36.2 regex-2024.11.6 requests-2.32.3 rich-14.0.0 rich-toolkit-0.14.1 rpds-py-0.24.0 safetensors-0.5.3 scipy-1.15.2 sentencepiece-0.2.0 shellingham-1.5.4 six-1.17.0 sniffio-1.3.1 starlette-0.46.2 sympy-1.13.1 tiktoken-0.9.0 tokenizers-0.21.1 torch-2.6.0+cpu torchaudio-2.6.0+cpu torchvision-0.21.0+cpu tqdm-4.67.1 transformers-4.51.3 triton-3.2.0 typer-0.15.2 typing-inspection-0.4.0 typing_extensions-4.13.2 tzdata-2025.2 urllib3-2.4.0 uvicorn-0.34.1 uvloop-0.21.0 watchfiles-1.0.5 websockets-15.0.1 wrapt-1.17.2 xgrammar-0.1.18 xxhash-3.5.0 yarl-1.20.0 zipp-3.21.0




Output:

(vllm) ubuntu@ubuntu:~/Desktop$ /home/ubuntu/.local/bin/uv venv vllm --python 3.12 --seed
    Using CPython 3.12.7 interpreter at: /usr/bin/python3.12
    Creating virtual environment with seed packages at: vllm
    + pip==25.0.1
    Activate with: source vllm/bin/activate
    ubuntu@ubuntu:~/Desktop$ source vllm/bin/activate
    (vllm) ubuntu@ubuntu:~/Desktop$ sudo apt-get install -y gcc-12 g++-12 libnuma-dev
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    The following additional packages will be installed:
    cpp-12 gcc-12-base libgcc-12-dev libstdc++-12-dev
    Suggested packages:
    gcc-12-locales cpp-12-doc g++-12-multilib gcc-12-doc gcc-12-multilib libstdc++-12-doc
    The following NEW packages will be installed:
    cpp-12 g++-12 gcc-12 gcc-12-base libgcc-12-dev libnuma-dev libstdc++-12-dev
    0 upgraded, 7 newly installed, 0 to remove and 0 not upgraded.

(vllm) ubuntu@ubuntu:~/Desktop$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
    update-alternatives: using /usr/bin/gcc-12 to provide /usr/bin/gcc (gcc) in auto mode

(vllm) ubuntu@ubuntu:~/Desktop$ git clone https://github.com/vllm-project/vllm.git vllm_source

    Cloning into 'vllm_source'...
    remote: Enumerating objects: 66987, done.
    remote: Counting objects: 100% (181/181), done.
    remote: Compressing objects: 100% (163/163), done.
    remote: Total 66987 (delta 92), reused 18 (delta 18), pack-reused 66806 (from 3)
    Receiving objects: 100% (66987/66987), 45.12 MiB | 17.51 MiB/s, done.
    Resolving deltas: 100% (52073/52073), done.
(vllm) ubuntu@ubuntu:~/Desktop$ cd vllm_source

(vllm) ubuntu@ubuntu:~/Desktop/vllm_source$ pip install --upgrade pip
    Requirement already satisfied: pip in /home/ubuntu/Desktop/vllm/lib/python3.12/site-packages (25.0.1)
    (vllm) ubuntu@ubuntu:~/Desktop/vllm_source$ pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
    Collecting cmake>=3.26
    Downloading cmake-4.0.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.3 kB)
    Collecting wheel
    Downloading wheel-0.45.1-py3-none-any.whl.metadata (2.3 kB)

(vllm) ubuntu@ubuntu:~/Desktop/vllm_source$ pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    Using pip 25.0.1 from /home/ubuntu/Desktop/vllm/lib/python3.12/site-packages/pip (python 3.12)
    Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cpu
    Ignoring torch: markers 'platform_system == "Darwin"' don't match your environment
    Ignoring torch: markers 'platform_machine == "ppc64le" or platform_machine == "aarch64"' don't match your environment
    Ignoring torch: markers 'platform_machine == "s390x"' don't match your environment
    Ignoring torchaudio: markers 'platform_machine == "ppc64le"' don't match your environment
    Ignoring torchvision: markers 'platform_machine == "ppc64le"' don't match your environment
    Collecting cachetools (from -r /home/ubuntu/Desktop/vllm_source/requirements/common.txt (line 1))
    Obtaining dependency information for cachetools from https://files.pythonhosted.org/packages/72/76/20fa66124dbe6be5cafeb312ece67de6b61dd91a0247d1ea13db4ebb33c2/cachetools-5.5.2-py3-none-any.whl.metadata
    Downloading cachetools-5.5.2-py3-none-any.whl.metadata (5.4 kB)

Using /home/ubuntu/Desktop/vllm/lib/python3.12/site-packages
Searching for charset-normalizer==3.4.1
Best match: charset-normalizer 3.4.1
Adding charset-normalizer 3.4.1 to easy-install.pth file
Installing normalizer script to /home/ubuntu/Desktop/vllm/bin

Using /home/ubuntu/Desktop/vllm/lib/python3.12/site-packages
Searching for mpmath==1.3.0
Best match: mpmath 1.3.0
Adding mpmath 1.3.0 to easy-install.pth file

Best match: markdown-it-py 3.0.0
Adding markdown-it-py 3.0.0 to easy-install.pth file
Installing markdown-it script to /home/ubuntu/Desktop/vllm/bin

Using /home/ubuntu/Desktop/vllm/lib/python3.12/site-packages
Searching for mdurl==0.1.2
Best match: mdurl 0.1.2
Adding mdurl 0.1.2 to easy-install.pth file

Using /home/ubuntu/Desktop/vllm/lib/python3.12/site-packages
Finished processing dependencies for vllm==0.8.5.dev32+g44fa4d556.cpu

(vllm) ubuntu@ubuntu:~/Desktop/vllm_source$ vllm serve Qwen/Qwen2.5-1.5B-Instruct
INFO 04-16 09:21:44 [__init__.py:239] Automatically detected platform cpu.
INFO 04-16 09:21:46 [api_server.py:1042] vLLM API server version 0.8.5.dev32+g44fa4d556
INFO 04-16 09:21:46 [api_server.py:1043] args: Namespace(subparser='serve', model_tag='Qwen/Qwen2.5-1.5B-Instruct', config='', host=None, port=8000, uvicorn_log_level='info', disable_uvicorn_access_log=False, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, enable_ssl_refresh=False, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='Qwen/Qwen2.5-1.5B-Instruct', task='auto', tokenizer=None, hf_config_path=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, allowed_local_media_path=None, load_format='auto', download_dir=None, model_loader_extra_config=None, use_tqdm_on_load=True, config_format=<ConfigFormat.AUTO: 'auto'>, dtype='auto', kv_cache_dtype='auto', max_model_len=None, guided_decoding_backend='auto', logits_processor_pattern=None, model_impl='auto', distributed_executor_backend=None, pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False, max_parallel_loading_workers=None, ray_workers_use_nsight=False, disable_custom_all_reduce=False, block_size=None, enable_prefix_caching=None, prefix_caching_hash_algo='builtin', disable_sliding_window=False, use_v2_block_manager=True, seed=None, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, hf_token=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=8192, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizm_processor_kwargs=Noner_pool_extra_config=None, limit_mm_per_prompt=None, me, disable_mm_preprocessor_cache=False, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, speculative_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, max_num_batched_tokens=None, max_num_seqs=None, max_num_partial_prefills=1, max_long_partial_prefills=1, long_prefill_token_threshold=0, num_lookahead_slots=0, scheduler_delay_factor=0.0, enable_chunked_prefill=None, multi_step_stream_outputs=True, scheduling_policy='fcfs', disable_chunked_mm_input=False, scheduler_cls='vllm.core.scheduler.Scheduler', override_neuron_config=None, override_pooler_config=None, compilation_config=None, kv_transfer_config=None, worker_cls='auto', worker_extension_cls='', generation_config='auto', override_generation_config=None, enable_sleep_mode=False, calculate_kv_scales=False, additional_config=None, enable_reasoning=False, reasoning_parser=None, disable_cascade_attn=False, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False, enable_server_load_tracking=False, dispatch_function=<function ServeSubcommand.cmd at 0x7ac262db7420>)
config.json: 

INFO 04-16 09:21:54 [config.py:697] This model supports multiple tasks: {'score', 'generate', 'embed', 'reward', 'classify'}. Defaulting to 'generate'.
WARNING 04-16 09:21:54 [arg_utils.py:1745] device type=cpu is not supported by the V1 Engine. Falling back to V0. 
INFO 04-16 09:21:54 [config.py:1758] Disabled the custom all-reduce kernel because it is not supported on current platform.
WARNING 04-16 09:21:54 [cpu.py:106] Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) for CPU backend is not set, using 4 by default.
WARNING 04-16 09:21:54 [cpu.py:119] uni is not supported on CPU, fallback to mp distributed executor backend.
INFO 04-16 09:21:54 [api_server.py:246] Started engine process with PID 9472
tokenizer_config.json: INFO 04-16 09:21:58 [llm_engine.py:243] Initializing a V0 LLM engine (v0.8.5.dev32+g44fa4d556) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, kv_cache_dtype=auto,  device_config=cpu, decoding_config=DecodingConfig(guided_decoding_backend='auto', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=None, served_model_name=Qwen/Qwen2.5-1.5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=None, chunked_prefill_enabled=False, use_async_output_proc=False, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=True, 
generation_config.json: INFO 04-16 09:24:56 [weight_utils.py:281] Time spent downloading weights for Qwen/Qwen2.5-1.5B-Instruct: 175.640931 seconds
INFO 04-16 09:24:56 [weight_utils.py:315] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.96it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.96it/s]

INFO 04-16 09:24:56 [loader.py:458] Loading weights took 0.43 seconds
INFO 04-16 09:24:56 [executor_base.py:112] # cpu blocks: 9362, # CPU blocks: 0
INFO 04-16 09:24:56 [executor_base.py:117] Maximum concurrency for 32768 tokens per request: 4.57x
INFO 04-16 09:24:57 [llm_engine.py:449] init engine (profile, create kv cache, warmup model) took 0.44 seconds
WARNING 04-16 09:24:57 [config.py:1185] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
INFO 04-16 09:24:57 [serving_chat.py:118] Using default chat sampling params from model: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
INFO 04-16 09:24:57 [serving_completion.py:61] Using default completion sampling params from model: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
INFO 04-16 09:24:57 [api_server.py:1089] Starting vLLM API server on http://0.0.0.0:8000
INFO 04-16 09:24:57 [launcher.py:26] Available routes are:
INFO 04-16 09:24:57 [launcher.py:34] Route: /openapi.json, Methods: HEAD, GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /docs, Methods: HEAD, GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /docs/oauth2-redirect, Methods: HEAD, GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /redoc, Methods: HEAD, GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /health, Methods: GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /load, Methods: GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /ping, Methods: POST, GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /tokenize, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /detokenize, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /v1/models, Methods: GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /version, Methods: GET
INFO 04-16 09:24:57 [launcher.py:34] Route: /v1/chat/completions, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /v1/completions, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /v1/embeddings, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /pooling, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /score, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /v1/score, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /v1/audio/transcriptions, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /rerank, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /v1/rerank, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /v2/rerank, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /invocations, Methods: POST
INFO 04-16 09:24:57 [launcher.py:34] Route: /metrics, Methods: GET
INFO:     Started server process [9432]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     127.0.0.1:58452 - "GET /v1/models HTTP/1.1" 200 OK
INFO:     127.0.0.1:41754 - "GET /version HTTP/1.1" 200 OK
INFO:     127.0.0.1:42098 - "GET /docs HTTP/1.1" 200 OK
INFO:     127.0.0.1:54788 - "GET /invocations HTTP/1.1" 405 Method Not Allowed
INFO:     127.0.0.1:35408 - "GET /score HTTP/1.1" 405 Method Not Allowed
INFO:     127.0.0.1:47636 - "GET /v1/score HTTP/1.1" 405 Method Not Allowed
INFO 04-16 09:38:36 [logger.py:39] Received request cmpl-aae746bcb8594a8ab23b0faf8e4047dc-0: prompt: 'San Francisco is a', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.1, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=7, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [23729, 12879, 374, 264], lora_request: None, prompt_adapter_request: None.
INFO 04-16 09:38:36 [engine.py:310] Added request cmpl-aae746bcb8594a8ab23b0faf8e4047dc-0.
WARNING 04-16 09:38:37 [cpu.py:170] Pin memory is not supported on CPU.
INFO 04-16 09:38:37 [metrics.py:489] Avg prompt throughput: 0.4 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO:     127.0.0.1:60910 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 04-16 09:38:47 [metrics.py:489] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.6 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 04-16 09:38:57 [metrics.py:489] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 04-16 09:39:59 [logger.py:39] Received request cmpl-e7e1a30514544bfc94d41a95b52b53f3-0: prompt: 'What is IBM write in 1 line?', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.1, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=7, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [3838, 374, 27922, 3270, 304, 220, 16, 1555, 30], lora_request: None, prompt_adapter_request: None.
INFO 04-16 09:39:59 [engine.py:310] Added request cmpl-e7e1a30514544bfc94d41a95b52b53f3-0.
INFO:     127.0.0.1:41882 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 04-16 09:40:10 [metrics.py:489] Avg prompt throughput: 0.7 tokens/s, Avg generation throughput: 0.6 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 04-16 09:40:20 [metrics.py:489] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
