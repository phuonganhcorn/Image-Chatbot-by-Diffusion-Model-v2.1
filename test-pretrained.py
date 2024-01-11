import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import PIL
from PIL import Image
from tqdm import tqdm 
import glob
import json

'''
def generate_image():
    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # Load a file of prompts
    prompt_list = []
    path = r'D:/SE2023-9.1/SDmodel/data/output/txt2img-pretrained/input.txt'
    with open(path, 'r') as file:
        prompt_list = file.read().splitlines()

    # Save to a specific directory
    output_directory = r'D:/SE2023-9.1/SDmodel/data/output/txt2img-pretrained/'

    for index in tqdm(range(len(prompt_list))):
        image = pipe(prompt_list[index]).images[0]
        image_path = f'{output_directory}Image_{index}_pretrained.png'
        image.save(image_path)
        image.close()

def input_text():
    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # Load a file of prompts
    prompt = str(input("Enter description: "))
    #path = r'D:/SE2023-9.1/SDmodel/data/output/txt2img-pretrained/input.txt'
    #with open(path, 'r') as file:
    #    prompt_list = file.read().splitlines()

    # Save to a specific directory
    output_directory = r'D:/SE2023-9.1/SDmodel/data/output/meomeo/'

    #for index in tqdm(range(len(prompt_list))):
    image = pipe(prompt).images[0]
    image_path = f'{output_directory}Image_pretrained.png'
    image.save(image_path)
    image.close()
'''

def create_json():
    imgList = glob.glob('D:/SE2023-9.1/SDmodel/data/output/txt2img-pretrained/*.png')
    print(imgList)
    capPath = 'D:/SE2023-9.1/SDmodel/data/output/txt2img-pretrained/input.txt'
    with open(capPath, 'r') as cf:
        caption = cf.read().splitlines()

    jsonPath = 'D:/SE2023-9.1/SDmodel/data/output/txt2img-pretrained/log-cap.json'
    data = []
    for img, cap in zip(imgList, caption):
        entry = {"image_path": img, "caption": cap}
        data.append(entry)

    with open(jsonPath, 'w', encoding='utf-8') as w:
        jsonString = json.dumps(data, indent=4)
        w.write(jsonString)


if __name__ == "__main__":
    #create_json()
    create_json()