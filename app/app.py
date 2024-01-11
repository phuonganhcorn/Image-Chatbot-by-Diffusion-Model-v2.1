import gradio as gr

from diffusers import DiffusionPipeline, LCMScheduler
import torch

import base64
from io import BytesIO
import os
import gc
import warnings

# Only used when MULTI_GPU set to True
from helper import UNetDataParallel
from share_btn import community_icon_html, loading_icon_html, share_js



# Process environment variables
use_ssd = os.getenv("USE_SSD", "false").lower() == "true"
if use_ssd:
    model_key_base = "D:\SE2023-9.1\SDmodel\stable-diffusion-2-1"
    model_key_refiner = "D:\SE2023-9.1\SDmodel\stable-diffusion-2-1"
    lcm_lora_id = "latent-consistency/lcm-lora-ssd-1b"
else:
    model_key_base = "D:\SE2023-9.1\SDmodel\SD-2-1-newtrained"
    model_key_refiner = "D:\SE2023-9.1\SDmodel\stable-diffusion-2-1"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

# Use LCM LoRA (disabled by default)
if "ENABLE_LCM" not in os.environ:
    warnings.warn("`ENABLE_LCM` environment variable is not set. LCM LoRA will be disabled by default. You can set it to `True` to turn on LCM LoRA.")
enable_lcm = os.getenv("ENABLE_LCM", "false").lower() == "true"


# Use refiner (disabled by default if LCM is enabled)
enable_refiner = os.getenv("ENABLE_REFINER", "false" if enable_lcm or use_ssd else "true").lower() == "true"
# Output images before the refiner and after the refiner
output_images_before_refiner = os.getenv("OUTPUT_IMAGES_BEFORE_REFINER", "false").lower() == "true"

offload_base = os.getenv("OFFLOAD_BASE", "false").lower() == "true"
offload_refiner = os.getenv("OFFLOAD_REFINER", "true").lower() == "true"

# Generate how many images by default
default_num_images = int(os.getenv("DEFAULT_NUM_IMAGES", "4"))
if default_num_images < 1:
    default_num_images = 1

# Create public link
share = os.getenv("SHARE", "false").lower() == "true"

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float32, use_safetensors=True, variant="fp16")

# Manually modify the down.weight tensor
new_down_weight = torch.randn((64, 1024))  # Replace this with your desired initialization
pipe.unet.state_dict()['down.weight'] = new_down_weight

if enable_lcm:
    pipe.load_lora_weights(lcm_lora_id)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

multi_gpu = os.getenv("MULTI_GPU", "false").lower() == "true"

if multi_gpu:
    pipe.unet = UNetDataParallel(pipe.unet)
    pipe.unet.config, pipe.unet.dtype, pipe.unet.add_embedding = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.add_embedding
    pipe.to("cuda")
else:
    if offload_base:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

if enable_refiner:
    print("Loading model", model_key_refiner)
    pipe_refiner = DiffusionPipeline.from_pretrained(model_key_refiner, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    if multi_gpu:
        pipe_refiner.unet = UNetDataParallel(pipe_refiner.unet)
        pipe_refiner.unet.config, pipe_refiner.unet.dtype, pipe_refiner.unet.add_embedding = pipe_refiner.unet.module.config, pipe_refiner.unet.module.dtype, pipe_refiner.unet.module.add_embedding
        pipe_refiner.to("cpu")
    else:
        if offload_refiner:
            pipe_refiner.enable_model_cpu_offload()
        else:
            pipe_refiner.to("cpu")


is_gpu_busy = False
def infer(prompt, negative, scale, samples=4, steps=50, refiner_strength=0.3, seed=-1):
    prompt, negative = [prompt] * samples, [negative] * samples

    g = torch.Generator(device="cpu")
    if seed != -1:
        g.manual_seed(seed)
    else:
        g.seed()

    images_b64_list = []

    if not enable_refiner or output_images_before_refiner:
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps, generator=g).images
    else:
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps, output_type="latent", generator=g).images

    gc.collect()
    torch.cuda.empty_cache()

    if enable_refiner:
        if output_images_before_refiner:
            for image in images:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                image_b64 = (f"data:image/jpeg;base64,{img_str}")
                images_b64_list.append(image_b64)

        images = pipe_refiner(prompt=prompt, negative_prompt=negative, image=images, num_inference_steps=steps, strength=refiner_strength, generator=g).images

        gc.collect()
        torch.cuda.empty_cache()

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        images_b64_list.append(image_b64)
    
    return images_b64_list
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: orange;
            background: orange;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .gradio-container {
            max-width: 750px !important;
            margin: auto;
            padding-top: 25px;
        }
        #gallery {
            min-height: 350px;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: 8px !important;
            border-bottom-left-radius: 8px !important;
        }
        #gallery>div>.h-full {
            min-height: 320px;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
            margin: 0 10px 0 0;
        }
        #generate-image-btn {
            margin: 0 0 0 10px;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        
        
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
"""

block = gr.Blocks(css=css)

default_guidance_scale = 1 if enable_lcm else 9
    
examples = [
    [
        'Vietnamese family gathering with a meal on the terrace ',
        'low quality',
        default_guidance_scale
    ],
    [
        'A Vietnamese girl eating Pho',
        'low quality',
        default_guidance_scale
    ],
    [
        'People working in field',
        'low quality, 3d',
        default_guidance_scale
    ],
    [
        'A traditional Vietnamese house on top of a mountain',
        'low quality , photorealistic',
        default_guidance_scale
    ],
    [
        "Ha Long Bay",
        'low quality, ugly',
        default_guidance_scale
    ],
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  Stable Diffusion Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                This Stable Diffusion is the finetuning text-to-image model from SE9.1. 
                <br/>
                Source code of this space is on 
                <a
                  href="https://github.com/hachi793/se9.1"
                  style="text-decoration: underline;"
                  target="_blank"
                  >SE9.1/Vietnam Stable Diffusion</a>.
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container", equal_height=True, style=dict(mobile_collapse=False)):
                with gr.Column():
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                        elem_id="negative-prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                btn = gr.Button("Generate image", elem_id="generate-image-btn").style(
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Group(elem_id="container-advanced-btns"):
            with gr.Group(elem_id="share-btn-container"):
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")

        with gr.Accordion("Advanced settings", open=False):
            samples = gr.Slider(label="Images", minimum=1, maximum=max(16 if enable_lcm else 4, default_num_images), value=default_num_images, step=1)
            if enable_lcm:
                steps = gr.Slider(label="Steps", minimum=1, maximum=10, value=4, step=1)
            else:
                steps = gr.Slider(label="Steps", minimum=1, maximum=250, value=50, step=1)
                
            if enable_refiner:
                refiner_strength = gr.Slider(label="Refiner Strength", minimum=0, maximum=1.0, value=0.3, step=0.1)
            else:
                refiner_strength = gr.Slider(label="Refiner Strength (refiner not enabled)", minimum=0, maximum=0, value=0, step=0)
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=default_guidance_scale, step=0.1
            )

            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative, guidance_scale], outputs=[gallery, community_icon, loading_icon, share_button], cache_examples=False)
        ex.dataset.headers = [""]
        negative.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
        text.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
        
        
        share_button.click(
            None,
            [],
            [],
            _js=share_js,
        )
       
share=True
block.queue().launch(share=share)
