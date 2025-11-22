from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
import os

def qwen(
    prompt: str = "Give me a short introduction to large language model.",
    enable_thinking: bool = False
) -> dict[str, str]:
    # 项目内模型存储路径（与代码文件同级的 models 文件夹）
    model_cache_dir = os.path.join(os.path.dirname(__file__), "models")
    # 如果文件夹不存在则自动创建
    os.makedirs(model_cache_dir, exist_ok=True)

    model_name: str = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_cache_dir  # 关键：指定项目内路径
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=model_cache_dir  # 关键：指定项目内路径
    )

    # prepare the model input
    messages: list[dict[str, str]] = [
        {"role": "user", "content": prompt}
    ]
    text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
    )
    # tokenizer returns a BatchEncoding; move contained tensors to the model device explicitly
    model_inputs = tokenizer([text], return_tensors="pt")
    # model_inputs = model_inputs.to(model.device)
    model_inputs: dict = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}

    # conduct text completion
    try:
        generated_ids: Tensor = model.generate( # type: ignore
            **model_inputs,
            max_new_tokens=32768
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    output_ids: list[int] = generated_ids[0][len(model_inputs['input_ids'][0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index: int = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content: str = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content: str = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # 返回字典形式的结果
    return {
        "thinking_content": thinking_content,
        "content": content
    }


def qwen_split(
    prompt: str = "Give me a short introduction to large language model.",
    enable_thinking: bool = False
) -> str:
    # 项目内模型存储路径（与代码文件同级的 models 文件夹）
    model_cache_dir = os.path.join(os.path.dirname(__file__), "models")
    # 如果文件夹不存在则自动创建
    os.makedirs(model_cache_dir, exist_ok=True)

    model_name: str = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_cache_dir  # 关键：指定项目内路径
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=model_cache_dir  # 关键：指定项目内路径
    )

    # prepare the model input
    messages: list[dict[str, str]] = [
        {"role": "user", "content": prompt}
    ]
    text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
    )
    # tokenizer returns a BatchEncoding; move contained tensors to the model device explicitly
    model_inputs = tokenizer([text], return_tensors="pt")
    # model_inputs = model_inputs.to(model.device)
    model_inputs: dict = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}

    # conduct text completion
    try:
        generated_ids: Tensor = model.generate( # type: ignore
            **model_inputs,
            max_new_tokens=32768
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    output_ids: list[int] = generated_ids[0][len(model_inputs['input_ids'][0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index: int = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # thinking_content: str = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content: str = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # 返回字典形式的结果
    return content
