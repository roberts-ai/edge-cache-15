import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from pipelines.models import TextToImageRequest
from torch import Generator
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)
 
def load_pipeline() -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "models/newdream-sdxl-20", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")
    
    prompt = "future punk robot shooting"
    pipe.fuse_qkv_projections()
    config = CompilationConfig.Default()
    # xformers and Triton are suggested for achieving best performance.
    try:
        import xformers
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')
    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')
    # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
    # But it can increase the amount of GPU memory used.
    # For StableVideoDiffusionPipeline it is not needed.
    config.enable_cuda_graph = True

    pipe = compile(pipe, config)
    pipe(prompt)
    return pipe

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=9,
    ).images[0]
