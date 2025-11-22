import json
from gemma import gemma_split

def llm_split_layers(user_prompt: str, width: int = 1024, height: int = 1024) -> list[dict]:

    system_prompt = f"""
    You are a poster layer splitting assistant, poster height is {height}px, poster width is {width}px, you need to split the user's poster requirements into 3-5 independent layers (each layer corresponds to an element), each layer includes:
    1. pos_prompt: the positive prompt for this layer's element (detailed description of style, color, shape)
    2. neg_prompt: the negative prompt for this layer (such as "blurry, deformed, extra elements")
    3. x: integer, the x coordinate of the layer's top-left corner (relative to the canvas top-left, range 0~canvas width, reasonably allocate position according to design)
    4. y: integer, the y coordinate of the layer's top-left corner (relative to the canvas top-left, range 0~canvas height, reasonably allocate position according to design)
    You should make sure every pos_prompt contains only one subject or element, and avoid multiple elements in one layer.
    Output format must be a dictionary array, without any other text, example:
    [
        {{"pos_prompt": "Light blue gradient background, simple without clutter, cartoon style", "neg_prompt": "Complex texture, text, patterns", "x": 0, "y": 0}},
        {{"pos_prompt": "Cartoon style white cat, sitting pose, cute expression", "neg_prompt": "Realistic, blurry, multiple cats", "x": 200, "y": 200}},
        {{"pos_prompt": "Small red bow, cartoon style", "neg_prompt": "Too large, blurry, colorful", "x": 800, "y": 100}}
    ]
    """
    
    # Parse LLM output
    try:
        # Step 1: Get the JSON format string from LLM output
        layers_str = gemma_split(user_prompt, system_prompt)
        # Step 2: Parse the JSON string into Python data (list of dictionaries)
        layers = json.loads(layers_str)
        
        # Step 3: Strictly validate types (compatible with Python 3.6+, and ensure it's a "list of dictionaries")
        if not isinstance(layers, list):
            raise ValueError("LLM output format error, not parsed as list")
        for item in layers:
            if not isinstance(item, dict):
                raise ValueError("LLM output format error, list elements must be dictionaries")
        
        return layers
    
    # Catch all possible exceptions (JSON parsing errors, type errors, function call errors, etc.)
    except Exception as e:
        # Fallback: return default layer configuration
        print(f"LLM output format error: {str(e)}, using default layer configuration")
        return [
            {"pos_prompt": "Gradient background, simple", "neg_prompt": "", "x": 0, "y": 0},
            {"pos_prompt": user_prompt, "neg_prompt": "", "x": 100, "y": 100}
        ]
        
if __name__ == "__main__":
    # Test the llm_split_layers function
    user_prompt = "A poster with a light blue gradient background, featuring a cute cartoon white cat sitting in the center, wearing a small red bow."
    layers = llm_split_layers(user_prompt, width=1024, height=1024)
    print(layers)