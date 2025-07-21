import os
import sys
import unittest

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, EagleDecodingConfig,
                                 KvCacheConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.parametrize("attn_backend,use_one_model", [("TRTLLM", False),
                                                        ("TRTLLM", True)])
@pytest.mark.high_cuda_memory
def test_mixed_cuda_graph_runner(attn_backend: str, use_one_model: bool):
    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    # bs > 1 gives non-deterministic when doing IFB. There are slight chances
    # that ref and spec does not match 100%
    max_batch_size = 1
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=False)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1])

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend=attn_backend,
        disable_overlap_scheduler=True,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        # Use a small max_seq_len to test mixed speculation mode.
        max_seq_len=100,
    )

    spec_config = EagleDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model_dir=eagle_model_dir,
        # Llama 3 does not support one model eagle.
        eagle3_one_model=use_one_model,
    )

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    # Output tests
    prompts = [
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    # Got output like
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'
    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert similar(text_spec, text_ref)


if __name__ == "__main__":
    unittest.main()
