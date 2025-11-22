# main.py
# Entry point for the layered image generation pipeline.
# Aggregates functionality from llm.py, layers.py, and utils.py.

import os
import sys
from llm import llm_split_layers
from layers import gen_trans
from utils import create_psd

def main():
    
    user_prompt = "A marketing poster for Coca-Cola"
    
    # Step 1: Split the prompt into layers using LLM
    print("Splitting prompt into layers...")
    layers = llm_split_layers(user_prompt, width=1024, height=1024)
    if not layers:
        print("Failed to generate layers. Exiting.")
        sys.exit(1)
    print(f"Prepared {len(layers)} layers")
    
    # Step 2: Generate images for each layer
    print("Generating layer images...")
    layer_results = gen_trans(layers=layers, width=1024, height=1024)
    print(f"Generated {len(layer_results)} layer images.")
    
    # Save each layer image to /static directory
    os.makedirs("./static", exist_ok=True)
    for layer_result in layer_results:
        img = layer_result['img']
        layer_id = layer_result['id']
        layer_prompt = layer_result['pos_prompt']
        img_path = f"./static/{layer_prompt}.png"
        img.save(img_path)
        print(f"Saved layer {layer_id} to {img_path}")
    
    # Step 3: Composite into PSD
    output_path = "./static/output.psd"
    print("Compositing into PSD...")
    create_psd(layer_results, output_path)
    print(f"Pipeline complete! PSD saved to {output_path}")

if __name__ == "__main__":
    main()