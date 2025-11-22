import json
from gemma import gemma_split

def llm_split_layers(user_prompt: str, width: int = 1024, height: int = 1024) -> list[dict]:

    system_prompt = f"""
    You are a poster layer splitting assistant. Poster dimensions: {width}px width, {height}px height.
    
    Task: Split the user's poster description into 3-5 independent layers. Each layer must represent **exactly one distinct element** (e.g., background, main subject, accessory). Do NOT combine multiple elements into one layer.
    
    Each layer must include:
    1. pos_prompt: A detailed positive prompt describing ONLY the single element for this layer (style, color, shape, position hints if relevant).
    2. neg_prompt: Negative prompt for this layer (e.g., "blurry, deformed, extra elements, other subjects").
    3. x: Integer x-coordinate of the layer's top-left corner (0 to {width}, allocate based on design).
    4. y: Integer y-coordinate of the layer's top-left corner (0 to {height}, allocate based on design).
    
    Rules:
    - Each pos_prompt must focus on ONE element only. Avoid mentioning other elements.
    - Layers should be composable (e.g., background first, then subjects on top).
    - Output MUST be a JSON array of dictionaries, no extra text or formatting.
    
    Example for "A poster with a blue background and a red apple in the center":
    [
        {{"pos_prompt": "Solid blue background, flat color", "neg_prompt": "Textures, patterns, objects", "x": 0, "y": 0}},
        {{"pos_prompt": "Red apple, round shape, centered", "neg_prompt": "Green, deformed, background elements", "x": 400, "y": 400}}
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