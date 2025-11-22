from psd_tools import PSDImage

def create_psd(layer_results: list[dict], output_path: str) -> None:
    """
    将多个RGBA图层合成为PSD文件
    :param layer_results: 从gen_trans返回的图层列表(含image/x/y等元信息)
    :param output_path: PSD输出路径(如./static/output.psd)
    """
    if not layer_results:
        raise ValueError("无图层可合成")
    
    # 获取PSD画布大小（所有图层统一宽高）
    canvas_width = layer_results[0]["width"]
    canvas_height = layer_results[0]["height"]
    
    # 创建空PSD（背景透明）
    psd = PSDImage.new(mode="RGBA", size=(canvas_width, canvas_height))
    
    # 逐个添加图层到PSD（注意：PSD图层顺序是"先加的在下方"，所以逆序添加）
    for layer_info in reversed(layer_results):
        rgba_image = layer_info["image"]
        layer_x = layer_info["x"]
        layer_y = layer_info["y"]
        layer_name = layer_info["name"]
        
        # 创建一个新的PSD图像作为图层
        layer_psd = PSDImage.frompil(rgba_image)
        
        # 获取图层并设置位置、名称
        layer = layer_psd[0]  # PSDImage.frompil 创建的PSD只有一个图层
        layer.left = layer_x
        layer.top = layer_y
        layer.name = layer_name
        
        # 添加到主PSD
        psd.append(layer)
    
    # 保存PSD（压缩格式选择，保留图层可编辑性）
    psd.save(output_path)
    print(f"PSD文件已保存到: {output_path}")