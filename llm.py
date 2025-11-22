import json
from qwen import qwen_split

def llm_split_layers(user_prompt: str, width: int = 1024, height: int = 1024) -> list[dict]:

    system_prompt = f"""
    你是一个海报图层拆分助手，海报高度为{height}px, 海报宽度为{width}px, 你需要将用户的海报需求拆分为3-5个独立图层（每个图层对应一个元素），每个图层包含：
    1. pos_prompt：该图层元素的正向提示词（详细描述风格、颜色、形状）
    2. neg_prompt：该图层的负向提示词（如“模糊、变形、多余元素”）
    3. x：整数，图层左上角x坐标（相对于画布左上角，范围0~画布的宽度，根据设计合理分配位置）
    4. y：整数，图层左上角y坐标（相对于画布左上角，范围0~画布的高度，根据设计合理分配位置）
    输出格式必须是字典数组，不包含任何其他文字，示例：
    [
        {"pos_prompt": "淡蓝色渐变背景，简洁无杂物，卡通风格", "neg_prompt": "纹理复杂、文字、图案", "x": 0, "y": 0},
        {"pos_prompt": "卡通风格的白色猫，坐姿，表情可爱", "neg_prompt": "写实、模糊、多只猫", "x": 200, "y": 200},
        {"pos_prompt": "小巧的红色蝴蝶结，卡通风格", "neg_prompt": "过大、模糊、彩色", "x": 800, "y": 100}
    ]
    """
    
    # 解析LLM输出
    try:
        # 步骤1：获取 LLM 输出的 JSON 格式字符串
        layers_str = qwen_split(user_prompt)
        # 步骤2：解析 JSON 字符串为 Python 数据（列表嵌套字典）
        layers = json.loads(layers_str)
        
        # 步骤3：严格校验类型（兼容 Python 3.6+，且确保是「字典列表」）
        if not isinstance(layers, list):
            raise ValueError("LLM输出格式错误，未解析为列表")
        for item in layers:
            if not isinstance(item, dict):
                raise ValueError("LLM输出格式错误，列表元素必须是字典")
        
        return layers
    
    # 捕获所有可能的异常（JSON解析错误、类型错误、函数调用错误等）
    except Exception as e:
        # 降级方案：返回默认图层配置
        print(f"LLM输出格式错误：{str(e)}，使用默认图层配置")
        return [
            {"pos_prompt": "渐变背景，简洁", "neg_prompt": "", "x": 0, "y": 0},
            {"pos_prompt": user_prompt, "neg_prompt": "", "x": 100, "y": 100}
        ]