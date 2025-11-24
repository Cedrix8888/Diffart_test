import os
from torch.hub import download_url_to_file
from psd_tools import PSDImage


# download model from URL to local path
def download_model(url, local_path):
    
    # Ensure the models directory exists
    os.makedirs("./models", exist_ok=True)
    
    if os.path.exists(local_path):
        return local_path

    temp_path = local_path + '.tmp'
    download_url_to_file(url=url, dst=temp_path)
    os.rename(temp_path, local_path)
    return local_path



def create_psd(layer_results: list[dict], output_path: str) -> None:
    """
    Combine multiple RGBA layers into a PSD file.
    :param layer_results: List of layers returned from gen_trans
    :param output_path: PSD Output Path (e.g., ./static/output.psd)
    """
    if not layer_results:
        raise ValueError("No layers available for compositing")
    
    # Get PSD Canvas Size (unified width and height for all layers)
    canvas_width = layer_results[0]["width"]
    canvas_height = layer_results[0]["height"]
    
    # Create empty PSD (with transparent background)
    psd = PSDImage.new(mode="RGB", size=(canvas_width, canvas_height), color=255)
    
    for layer_info in layer_results:
        layer_image = layer_info["img"]
        layer_x = layer_info["x"]
        layer_y = layer_info["y"]
        layer_name = layer_info["pos_prompt"]
        
        
        # Create a new pixel_layer
        psd.create_pixel_layer(name=layer_name, image=layer_image, top=layer_y, left=layer_x)
        
    
    psd.save(output_path)
    print(f"The PSD file has been saved to: {output_path}")