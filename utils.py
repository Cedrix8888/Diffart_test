import os
from torch.hub import download_url_to_file

# download model from URL to local path
def download_model(url, local_path):
    
    # 确保 models 目录存在
    os.makedirs("./models", exist_ok=True)
    
    if os.path.exists(local_path):
        return local_path

    temp_path = local_path + '.tmp'
    download_url_to_file(url=url, dst=temp_path)
    os.rename(temp_path, local_path)
    return local_path