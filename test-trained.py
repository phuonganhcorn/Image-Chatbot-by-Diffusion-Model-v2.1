from diffusers import StableDiffusionPipeline
import torch

model_id = "D:\SE2023-9.1\SDmodel\stable-diffusion-2-1"
model_id2 = "D:\SE2023-9.1\SDmodel\SD-2-1-newtrained"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors = True).to("cuda")
pipe2 = StableDiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16, use_safetensors = True).to("cuda")
img  = pipe(prompt="A woman smiling").images[0]
img2 = pipe2(prompt="A woman smiling").images[0]
output = r'D:\SE2023-9.1\SDmodel\data\output\txt2img-trained\test-person001+1.png'
output2 = r'D:\SE2023-9.1\SDmodel\data\output\txt2img-trained\test-person002+1.png'
img.save(output)
img2.save(output2)