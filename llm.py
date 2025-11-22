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
    layers_str = ""
    try:
        # Step 1: Get the JSON format string from LLM output
        layers_str = gemma_split(user_prompt, system_prompt)
        
        # Strip Markdown code block if present
        if layers_str.startswith("```json") and layers_str.endswith("```"):
            layers_str = layers_str[7:-3].strip()
            
        # Step 2: Parse the JSON string into Python data (list of dictionaries)
        layers = json.loads(layers_str)
        
        # Step 3: Strictly validate types (compatible with Python 3.6+, and ensure it's a "list of dictionaries")
        if not isinstance(layers, list):
            raise ValueError("LLM output format error, not parsed as list")
        for idx, item in enumerate(layers):
            if not isinstance(item, dict):
                raise ValueError(f"The {idx+1}th element in the list is not a dictionary")
            # Optional: Validate that the dictionary contains the required 4 keys
            required_keys = {"pos_prompt", "neg_prompt", "x", "y"}
            missing_keys = required_keys - item.keys()
            if missing_keys:
                raise ValueError(f"Dictionary missing required keys: {missing_keys}")
            # Optional: Validate that x/y are integers and within legal range
            if not (isinstance(item["x"], int) and 0 <= item["x"] <= width):
                raise ValueError(f"x coordinate is invalid (value: {item['x']}, range: 0~{width})")
            if not (isinstance(item["y"], int) and 0 <= item["y"] <= height):
                raise ValueError(f"y coordinate is invalid (value: {item['y']}, range: 0~{height})")
        
        return layers
    
    # Catch all possible exceptions (JSON parsing errors, value errors, etc.)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}, LLM output: {layers_str}")
    except ValueError as e:
        print(f"Data format validation error: {str(e)}, LLM output: {layers_str}")
    except Exception as e:
        # Catch-all for other unexpected exceptions
        print(f"Unknown error: {str(e)}, LLM output: {layers_str}")
        
    return [
        {"pos_prompt": "Gradient background, simple", "neg_prompt": "complex, text, patterns", "x": 0, "y": 0},
        {"pos_prompt": user_prompt, "neg_prompt": "blurry, deformed, extra elements", "x": 100, "y": 100}
    ]
        
if __name__ == "__main__":
    # Test the llm_split_layers function
    user_prompt = "A poster with a light blue gradient background, featuring a cute cartoon white cat sitting in the center, wearing a small red bow."
    layers = llm_split_layers(user_prompt, width=1024, height=1024)
    print(layers)