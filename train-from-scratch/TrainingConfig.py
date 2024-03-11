import os
from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from huggingface_hub import create_repo

# Clear CUDA cache and set the CUDA device
torch.cuda.empty_cache()
os.environ['TORCH_USE_CUDA_DSA'] = '1'
torch.cuda.set_device(0)

@dataclass
class TrainingConfig:
    image_size: int = 768
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 10
    mixed_precision: str = "fp16"
    output_dir: str = r"D:\SE2023-9.1\SDmodel\train-from-scratch\person-hyperparameters3"
    seed: int = 0

config = TrainingConfig()

# Define the UNet2DModel
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D",) * 5 + ("AttnDownBlock2D",),
    up_block_types=("UpBlock2D",) * 5 + ("UpBlock2D",),
)

# Load dataset
config.dataset_name = r"D:\SE2023-9.1\SDmodel\data\train\human-face"
dataset = load_dataset(config.dataset_name, split="train")

# Preprocess images
preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}
dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

# Define noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Training loop
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and setup logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    accelerator.init_trackers("train_example")

    # Prepare model, optimizer, dataloader, and lr_scheduler
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Training epochs
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        # Iterate over batches
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Perform forward and backward passes
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress bar and log metrics
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch, evaluate and save images
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)

# Helper function to evaluate and save images
def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    image_grid = make_image_grid(images, rows=4, cols=4)

    test_dir = os.path.join(r'D:\SE2023-9.1\SDmodel\data\output\txt2img-trained\5epochs', "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

# Launch training loop
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)
