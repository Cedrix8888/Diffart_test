in colab test, AutoencoderKL and UNet2DConditionModel should use torch_dtype rather than dtype



prompt for llm(qwen3-0.6B): 
    1.你是一个海报设计师，接下来根据我的需求设计有多个图层的海报，每个图层都是带透明通道的图像，你先设计出每个图层，然后告诉我每个图层的提示词，我用你的提示词生成AI图像，要求提示词尽量详细。
    2.设计一个宣传电影《千与千寻》的海报，
    llm_return: 
        Create a vivid and detailed scene of a serene beach scene, featuring a young girl with a bright smile, a sunset, and a group of friends. Include elements like waves, sand, the sky, and the sound of the ocean. Make sure to capture the beauty and tranquility of the setting.
        a young girl with a bright smile



version_01: glass bottle, high quality

version_02: Create a vivid and detailed scene of a serene beach scene, featuring a young girl with a bright smile, a sunset, and a group of friends. Include elements like waves, sand, the sky, and the sound of the ocean. Make sure to capture the beauty and tranquility of the setting

version_03: a young girl with a bright smile

