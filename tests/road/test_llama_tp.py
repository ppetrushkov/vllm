# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
from typing import Union

import vllm
from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

from ..utils import VLLM_PATH, create_new_process_for_each_test, multi_gpu_test

MODEL_PATH = "meta-llama/Llama-2-7b-hf"


#EXPECTED_NO_LORA_OUTPUT = [
#    "\n solution:\n\n SELECT COUNT(*) FROM department WHERE department.Ranking > 56\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
#]

EXPECTED_NO_LORA_OUTPUT = [
    '\n\n solution:\n\n SELECT COUNT(*) FROM department WHERE department.Ranking > 56\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n',
]

EXPECTED_LORA_OUTPUT = [
    ' SELECT count(*) FROM head WHERE age  >  56 ',
]



def do_sample(llm: vllm.LLM,
              lora_path: str,
              lora_id: int,
              tensorizer_config_dict: Union[dict, None] = None) -> list[str]:
    prompts = [
        "<s>[INST] Write a SQL query to answer the question based on the table schema.\n\n context: department : Department_ID (number) , Name (text) , Creation (text) , Ranking (number) , Budget_in_Billions (number) , Num_Employees (number) | head : head_ID (number) , name (text) , born_state (text) , age (number) | management : department_ID (number) , head_ID (number) , temporary_acting (text)\n\n question: How many heads of the departments are older than 56 ? [/INST]",
            ]

    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,
                                          skip_special_tokens=False,
                                          stop=["[/assistant]"])

    if tensorizer_config_dict is not None:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(
                str(lora_id),
                lora_id,
                lora_path,
                tensorizer_config_dict=tensorizer_config_dict)
            if lora_id else None)
    else:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
            if lora_id else None)
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def generate_and_test(llm,
                      sql_lora_files,
                      tensorizer_config_dict: Union[dict, None] = None):
    print("lora adapter created")
    assert do_sample(llm,
                     sql_lora_files,
                     tensorizer_config_dict=tensorizer_config_dict,
                     lora_id=0) == EXPECTED_NO_LORA_OUTPUT

    print("lora 1")
    assert do_sample(llm,
                     sql_lora_files,
                     tensorizer_config_dict=tensorizer_config_dict,
                     lora_id=1) == EXPECTED_LORA_OUTPUT

    print("no lora")
    assert do_sample(llm,
                     sql_lora_files,
                     tensorizer_config_dict=tensorizer_config_dict,
                     lora_id=0) == EXPECTED_NO_LORA_OUTPUT

    print("lora 2")
    assert do_sample(llm,
                     sql_lora_files,
                     tensorizer_config_dict=tensorizer_config_dict,
                     lora_id=2) == EXPECTED_LORA_OUTPUT

    print("removing lora")


@create_new_process_for_each_test()
def test_llama_lora(sql_lora_files):

    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        adapter_type="road",
        # also test odd max_num_seqs
        max_num_seqs=13,
        max_loras=4,
        enable_chunked_prefill=True)
    generate_and_test(llm, sql_lora_files)


@multi_gpu_test(num_gpus=2)
@create_new_process_for_each_test()
def test_llama_lora_tp2(sql_lora_files):

    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        adapter_type="road",
        max_num_seqs=16,
        max_loras=4,
        tensor_parallel_size=2,
        enable_chunked_prefill=True,
    )
    generate_and_test(llm, sql_lora_files)

