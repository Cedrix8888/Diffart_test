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
    psd = PSDImage.new(mode="RGBA", size=(canvas_width, canvas_height))
    
    for layer_info in layer_results:
        rgba_image = layer_info["image"]
        layer_x = layer_info["x"]
        layer_y = layer_info["y"]
        layer_name = layer_info["name"]
        
        # Create a new PSD image as a layer      
        layer_psd = PSDImage.frompil(rgba_image)
        
        # Get the layer and set its position and name
        layer = layer_psd[0]  # PSDImage.frompil 创建的PSD只有一个图层
        layer.left = layer_x
        layer.top = layer_y
        layer.name = layer_name
        
        # Add to main PSD
        psd.append(layer)
    
    psd.save(output_path)
    print(f"The PSD file has been saved to: {output_path}")