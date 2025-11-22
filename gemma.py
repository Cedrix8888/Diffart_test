from transformers import pipeline
import torch
import os

def gemma(
    prompt: str = "introduce llm to me in detail.",
    system_message: str = "You are a helpful assistant."
) -> dict[str, str]:
    # 项目内模型存储路径（与代码文件同级的 models 文件夹）
    model_cache_dir = os.path.join(os.path.dirname(__file__), "models")
    # 如果文件夹不存在则自动创建
    os.makedirs(model_cache_dir, exist_ok=True)

    model_name: str = "google/gemma-3-1b-it"

    # load the pipeline with model caching
    try:
        pipe = pipeline(
            "text-generation", 
            model=model_name, 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            model_kwargs={"cache_dir": model_cache_dir}
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    # prepare the model input
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            },
        ],
    ]

    # conduct text completion
    try:
        output = pipe(messages, max_new_tokens=32768)
        # Extract the generated text from the pipeline output
        content = ""
        if output and len(output) > 0:
            # Get the generated text - Gemma pipeline returns different structure
            generated_text = output[0][0]['generated_text']
            
            # Find the assistant's response
            for msg in generated_text:
                if msg.get('role') == 'assistant':
                    msg_content = msg.get('content', [])
                    if isinstance(msg_content, list) and len(msg_content) > 0:
                        # Handle list format
                        text_item = msg_content[0]
                        if isinstance(text_item, dict) and 'text' in text_item:
                            content = text_item['text']
                        elif isinstance(text_item, str):
                            content = text_item
                    elif isinstance(msg_content, str):
                        # Handle string format
                        content = msg_content
                    break
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    # 返回字典形式的结果
    return {
        "content": content
    }

def gemma_split(user_prompt: str, system_prompt: str) -> str:
        # 项目内模型存储路径（与代码文件同级的 models 文件夹）
    model_cache_dir = os.path.join(os.path.dirname(__file__), "models")
    # 如果文件夹不存在则自动创建
    os.makedirs(model_cache_dir, exist_ok=True)

    model_name: str = "google/gemma-3-1b-it"

    # load the pipeline with model caching
    try:
        pipe = pipeline(
            "text-generation", 
            model=model_name, 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            model_kwargs={"cache_dir": model_cache_dir}
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    # prepare the model input
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            },
        ],
    ]

    # conduct text completion
    try:
        output = pipe(messages, max_new_tokens=32768)
        # Extract the generated text from the pipeline output
        content = ""
        if output and len(output) > 0:
            # Get the generated text - Gemma pipeline returns different structure
            generated_text = output[0][0]['generated_text']
            
            # Find the assistant's response
            for msg in generated_text:
                if msg.get('role') == 'assistant':
                    msg_content = msg.get('content', [])
                    if isinstance(msg_content, list) and len(msg_content) > 0:
                        # Handle list format
                        text_item = msg_content[0]
                        if isinstance(text_item, dict) and 'text' in text_item:
                            content = text_item['text']
                        elif isinstance(text_item, str):
                            content = text_item
                    elif isinstance(msg_content, str):
                        # Handle string format
                        content = msg_content
                    break
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    return content