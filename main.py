# main.py
# Entry point for the layered image generation pipeline.
# Aggregates functionality from llm.py, layers.py, and utils.py.

import os
import sys
from llm import llm_split_layers
from layers import gen_trans
from utils import create_psd

def main():
    # Example user prompt (replace with input or CLI arg as needed)
    user_prompt = "A vibrant poster with a starry night sky background, a glowing moon in the center, and floating clouds around it."
    
    # Step 1: Split the prompt into layers using LLM
    print("Splitting prompt into layers...")
    layers = llm_split_layers(user_prompt, width=1024, height=1024)
    if not layers:
        print("Failed to generate layers. Exiting.")
        sys.exit(1)
    print(f"Generated {len(layers)} layers: {[layer['pos_prompt'][:30] + '...' for layer in layers]}")
    
    # Step 2: Generate images for each layer
    print("Generating layer images...")
    layer_results = gen_trans(layers=layers, width=1024, height=1024)
    print(f"Generated {len(layer_results)} layer images.")
    
    # Step 3: Composite into PSD
    output_path = "./static/output.psd"
    os.makedirs("./static", exist_ok=True)
    print("Compositing into PSD...")
    create_psd(layer_results, output_path)
    print(f"Pipeline complete! PSD saved to {output_path}")

if __name__ == "__main__":
    main()