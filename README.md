# se9.1
## Member
#### Ngô Phương Anh - 20001523
#### Nguyễn Hà Chi - 20001530
## PROJECT IMAGE CHATBOT

### Overview
Image Chatbot is a project aimed at building a chatbot that allows users to _input prompt_ and recieve _output image_ based on. The chatbot is developed on a web server with the core Model AI Stable Diffusion. Currently, there are many guides available for installing the Stable Diffusion model in all versions. However, these guides are all performed on the Linux operating system. _The project will supplement additional installation instructions for the model and necessary libraries on the Windows operating system in the Requirements section._ Relevant information regarding citations, model links, and related articles on building the chatbot will be cited in the References section.

There are 5 important steps in the development process:
- **Environment Setup:** Configure the necessary environment.
- **Dataset Preparation:** Gather and organize the dataset.
- **Model Training:** Train the model using the prepared dataset.
- **Web Interface Development:** Build a user-friendly web interface.
- **Demo App Creation:** Develop a demonstration application.

## 1. MODEL INSTALLATION
### REQUIREMENTS

Currently, there are various versions of the Stable Diffusion Model available for users to choose from. For building the Chatbot, we have selected the Stable Diffusion version SD2.1-v. The SD2.1 version has been optimized and fine-tuned, resulting in generally better image quality generated from text compared to previous versions.

Due to the heavyweight of AI libraries required for the project, we need to set up an environment to download and run the model.

#### 1.1. Environment Setup
Users need to install the [miniconda](https://docs.anaconda.com/free/miniconda/index.html) environment and refer to the [conda-cheatsheet guide](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to create a suitable environment.

#### 1.2. Library Installation
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
```

>[!CAUTION]
> - Users can use the above command to install the pytorch environment and necessary libraries. However, please note that currently, there are newer versions of pytorch available. The environment installation may crash due to conflicts between these library versions. Additionally, for Windows operating systems, the installed version may need to be adjusted.
> - Users can visit the link [Pytorch version](https://pytorch.org/get-started/previous-versions/)  to find the appropriate pytorch version for their operating system and fix any installation conflicts with other libraries.

> [!TIP]
> - When encountering errors during environment setup, two suggested approaches can be drawn from the project's execution process.
> - Execute the above commands, and if the system reports errors, pay attention to identify which library causes the error (often conflicts occur when installing three libraries and the torchvision, pytorch, transformers environment)
> - _Approach 1_: Downgrade/Upgrade pytorch immediately when the program reports an error following the link to pytorch versions above.
> - _Approach 2_: Continue to skip and follow the instructions. Because we still need to install and set up many libraries later on. We will perform downgrade/upgrade pytorch and add missing libraries later.


#### 1.3. Cài đặt xformers (optional)
[xformers](https://github.com/facebookresearch/xformers) là thư viện nhằm mục đích tối ưu thời gian huấn luyện cho model. Người dùng có thể cài đặt hoặc không. Tuy nhiên, đối với các máy tính có card cấu hình không quá cao. Nhà phát hành mô hình SDv2-1 và nhóm nghiên cứu highly recommend người dùng cài đặt thư viện này.

Quá trình này NÊN được thực hiện sau cùng sau khi chạy thử huấn luyện mô hình trước. Do hiện tại chỉ có hướng dẫn cài đặt xformers cho hệ điều hành Linux. Đối với hệ điều hành Windows, gần như chắc chắn sẽ gặp lỗi conflict version giữa các thư viện và môi trường đã được cài đặt.  

- Với hệ điều hành Windows ``Linux/macOS``
```bash
export CUDA_HOME=/usr/local/cuda-11.4
conda install -c nvidia/label/cuda-11.4.0 cuda-nvcc
conda install -c conda-forge gcc
conda install -c conda-forge gxx_linux-64==9.5.0
```

- Với hệ điều hành ``Windows`` (do gói conda-forge gcc chỉ có thể được cài đặt cho ``Linux/macOS``. Người dùng ``Windows`` cần sử dụng gói thư viện ``m2w64-gcc`` để thay thế).
```bash
set CUDA_HOME=/usr/local/cuda-11.4
conda install -c nvidia/label/cuda-11.4.0 cuda-nvcc
conda install -c conda-forge m2w64-gcc
```
Người dùng có thể visit đường link [Conda Forge Documentation](https://anaconda.org/conda-forge) để tham khảo thêm các packages nào phù hợp để cài đặt, các thư viện tương đương với hệ điều hành mình đang dùng.

- Sau khi set up sau, người dùng chạy câu lệnh sau để clone và cài đặt ``xformers``
```bash
cd ..
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
cd ..\SDmodel (cd về thư mục chứa model SD)
```  

Việc cài đặt ``xformers`` đem lại nhiều hiệu quả như giảm thời gian chạy của model, giảm lượng tiêu thụ bộ nhớ của mô hình khi huấn luyện. (Về nguyên lý hoạt động của xformers để giảm thiểu quá trình chạy sẽ được bàn đến sau).

#### 1.4. Chạy sample
Để kiểm tra liệu còn thiếu thư viện nào trong quá trình chạy hay không, người dùng có thể chạy thử ví dụ:
- **Cách 1:**
```bash
python scripts/txt2img.py 
--prompt "a professional photograph of an astronaut riding a horse" 
--ckpt <path/to/768model.ckpt/> 
--config configs/stable-diffusion/v2-inference-v.yaml 
--H 768 
--W 768  
```

- **Cách 2:** Copy và chạy đoạn code sau
```py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
    
image.save("astronaut_rides_horse.png")
```
## 2. DATASET
Sau quá trình chạy thử nghiệm và đánh giá kết quả với mô hình có sẵn, nhóm báo cáo kết luận mô hình có sẵn chưa hoàn toàn đáp ứng được nhu cầu tạo ra các hình ảnh con người, văn hóa, lịch sử,... châu Á một cách chân thật. Đặc biệt, đối với khuôn mặt người ở các sản phẩm còn bị méo mó, biến dạng,...

Nhóm làm sản phẩm quyết định sẽ finetune lại mô hình với dữ liệu cá nhân.

### 2.1. Cấu trúc của data
Mô hình SD là mô hình huấn luyện hình ảnh, yêu cầu input là 1 prompt, output là 1 hình ảnh. Mô hình sẽ huấn luyện với dataset có cấu trúc dạng file _json_ như dưới đây. 

![Hình ảnh data của nhóm sản phẩm](https://i.imgur.com/eJOkoKR.png)

**_Hình ảnh data của nhóm sản phẩm_** 

Như đã thấy ở file json phía trên, file json có cấu trúc bao gồm cặp key-value gồm _link_ và _caption_. Trong đó:
- _value_ của key _link_: path link của hình ảnh
- _value_ của key _caption_: mô tả nội dung của hình ảnh

Người dùng có thể chạy câu lệnh sau để view data sau khi clone project về để trực tiếp quan sát cấu trúc data.
```bash
cd .\data_backup1
type logs-train.txt // File data dưới dạng txt
type logs-train.csv // File data dưới dạng csv
type logs-train.json // File data dưới dạng json
```

### 2.2. Xây dựng dataset
Sau khi nắm được cấu trúc data cần để huấn luyện mô hình. Nhóm nghiên cứu sẽ xây dựng dataset gồm các hình ảnh thuộc các categories sau: 
- ``'Vietnam places', 'Vietnam people', 'Vietnam culture', 'Vietnam traditions', 'Vietnam food'``
- ``'Vietnam history', 'Vietnam architure', 'Vietnam art', 'Vietnam travel', 'Vietnam conical hat people', 'Vietnam ao dai', 'Vietnam clothes'``

Người dùng có thể chạy 1 trong 3 dòng code sau để crawl Data bằng các API

- Sử dụng BingAPI để scrape images. Bing API có ưu điểm dễ dàng sử dụng, không cần đăng kí để lấy token để sử dụng API. Không giới hạn số lượng hình ảnh scrape.
```bash
cd ./SDmodel/stablediffusion-main/scripts/scrapeimage
python BingAPI.py
```

- Sử dụng Beautiful Soup API để scrape images. Chất lượng hình ảnh khá ổn, ít bị lẫn các hình ảnh tạp. Tuy nhiên, cần đăng ký tài khoản và lấy token để sử dụng API này. Có giới hạn lượng dung lượng download bằng API.
```bash
cd ./SDmodel/stablediffusion-main/scripts/scrapeimage
python BeautifulSoup.py
```

- Sử Serpapi API. Đây là API để download hình ảnh của Google. Ưu và nhược điểm tương tự với Beautiful Soup API. 
```bash
cd ./SDmodel/stablediffusion-main/scripts/scrapeimage
python SerpapiAPI.py
```

**Nhóm nghiên cứu quyết định sử dụng các hình ảnh được crawl bằng Bing API. Người dùng có thể lựa chọn sử dụng bất kỳ API nào phù hợp với bản thân.**

Sau khi chạy dòng lệnh trên, hình ảnh sẽ được tự động scrape về máy local của người dùng và được lưu dưới cấu trúc folder như sau:

![Imgur](https://i.imgur.com/m6d2G9M.png)

**Samples data áo dài**

![Imgur](https://i.imgur.com/GHpmwhU.png)

**_Lưu ý: Quá trình chạy file code python để scrape hình ảnh đã bao gồm cả quá trình cấu trúc các folder trong dataset cũng như tên các file ảnh, định dạng các file ảnh để phục vụ cho quá trình huấn luyện model về sau_**

Ngoài ra, đối với mục tiêu finetune tập trung vào mặt người, do chất lượng data crawl được bằng API khá kém. Nhóm nghiên cứu bổ sung thêm data tự dataset [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) gồm 202,599 ảnh từ 10,177 các ngôi sao. Tất cả các ảnh là ảnh chân dung có chất lượng cao, gồm nhiều người từ nhiều độ tuổi khác nhau. Tỉ lệ nam nữ trong dataset khá cân bằng. Nhiều hình ảnh đa dạng gồm: chân dung người đội mũ, đeo kính, tóc ngắn, tóc thẳng, mặt tròn, mặt trái xoan,...

**Samples các hình ảnh trong CelebA**
![Imgur](https://i.imgur.com/eonnt00.png)

**Tổng kết số liệu về Dataset nhóm nghiên cứu tự xây dựng và áp dụng vào huấn luyện**
|Tên folder            |Số lượng hình ảnh trong dataset  |Định dạng tên|
|:---------------------|:------------------------------:|------------:|
|Vietnam places        | 350                               | **_VD: Image_9_1.jpg_**    | 
|Vietnam people        | 396                               |Image_8_100.jpg             |
|Vietnam culture       | 404                               |Image_5_90.jpg              | 
|Vietnam traditions    | 368                               |Image_10_190.jpg            |
|Vietnam food          | 370                               |Image_6_146.jpg             |
|Vietnam history       | 315                               |Image_7_0.jpg               |
|Vietnam architure     | 370                               |Image_1_0.jpg               |
|Vietnam art           | 215                               |Image_2_0.jpg               |
|Vietnam travel        | 205                               |Image_11_92.jpg             |
|Vietnam conical hat people| 199                           |Image_4_0.jpg               |
|Vietnam ao dai        | 211                               |Image_0_1.jpg               |
|Vietnam clothes       | 191                               |Image_3_175.jpg             |
|Human face            | 202,599                           |                            |

|**_Tổng số lượng hình ảnh_**              |**_206,193_**             |
|:-----------------------------------------|--------------------------|

Trong folder dataset, folder _Vietnam people_ (Data crawl) khác so với folder _Human face_ (Data CelebA) ở điểm hình ảnh con người trong folder _Vietnam people_ không phải chân dung mà gồm nhiều hoạt động khi con người làm việc khác.

**Đánh giá Dataset và đề xuất**
Theo như nhóm nghiên cứu đánh giá chất lượng của tập dataset, không đánh giá trên bộ data CelebA do CelebA là bộ data đã được tinh chỉnh và hiện là một trong những data hình ảnh để huấn luyện các mô hình AI, học máy về khuôn mặt con người tốt nhất hiện nay. Đối với các data được crawl bằng API, do được crawl bằng API, tốc độ thu thập dữ liệu rất nhanh, tuy nhiên để kiểm soát chất lượng, cần nhóm nghiên cứu phải rà soát lại các hình ảnh được download xuống để clean dataset. Trong tương lai nếu sản phẩm tiếp tục được phát triển thêm, nhóm nghiên cứu đề xuất phương án sắp xếp 1-2 người tập trung vào công việc clean data và loại bỏ các hình ảnh không liên quan sau khi crawl.

### 2.3. Cấu trúc file json cho quá trình huấn luyện
Để tạo ra file json có cấu trúc như đã nói trong phần cấu trúc data, nhóm nghiên cứu sử dụng kết hợp thêm model Image Captioning cho sản phẩm. Người dùng có thể tạo caption thủ công để mô tả nội dung hình ảnh. Điều này chắc chắn sẽ tạo nên sự khác biệt rất lớn trong chất lượng data, tuy nhiên sẽ phải trade-off với thời gian xây dựng dataset.

Nhóm nghiên cứu đặt mục tiêu tự tạo ra một dataset đủ tốt dùng cho huấn luyện và tập trung vào model SDv2.1, vậy nên model Image Captioning được chọn để xây dựng dataset cần thỏa mãn các yêu cầu: 
- Đơn giản trong cài đặt
- Mô hình pre-trained đã đủ tốt và không yêu cầu phải train lại mô hình
- Input và output có cấu trúc đơn giản

Người dùng có thể tùy chọn mô hình Image Captioning mong muốn để tạo ra bộ dataset chất lượng cao hơn hoặc phù hợp với mục đích cá nhân. Do nhóm nghiên cứu xây dựng dataset hình ảnh về Vietnam (hiện tại chưa có dataset như vậy trên thị trường), trong sản phẩm của mình, nhóm nghiên cứu lựa chọn model [Image Captioning sau đây](https://github.com/purveshpatel511/imageCaptioning)

Về cài đặt, người dùng có thể xem thêm trong link github của model. Sau khi cài đặt và chạy model, output sẽ có dạng như sau: 

Sau đó người dùng chạy code dưới đây để convert output về dạng json/csv/txt tùy nhu cầu

## 3. HUẤN LUYỆN MODEL
### 3.1. Kết quả

Trước khi tiến vào quá trình training, nhóm nghiên cứu thực hiện một số đánh giá sơ bộ trên mô hình pre-trained và thu được kết quả như sau

Người dùng có thể chạy samples bằng cách chạy bash script sau
```bash
python ./test-pretrained.py
```
**_Input_**: Người dùng có thể chạy câu lệnh sau để xem các prompt input
```bash
cd ./SDmodel/data/
type input.txt
type input2.txt
```
**_OUTPUT PRE-TRAINED_**

- **Prompt về địa điểm Việt Nam**

![Imgur](https://i.imgur.com/xQ0HacR.png)

- **Prompt về con người Việt Nam và các hoạt động**

![Imgur](https://i.imgur.com/VN5WKix.png)


### 3.2. Huấn luyện model from scratch
Hướng tiếp cận thứ nhất nhóm nghiên cứu đề ra đó là huấn luyện lại model từ đầu với tập dữ liệu đã được chuẩn bị để nâng cao khả năng generate khuôn mặt người của model.

Người dùng chạy file code sau để bắt đầu huấn luyện model
```bash
cd ./train-from-scratch
python TrainingConfig.py
```

Trong file *_[TrainingConfig.py]_*, các thông số về parameters khi huấn luyện model được set như sau

```py
@dataclass
class TrainingConfig:
    image_size = 768  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = r"D:\SE2023-9.1\SDmodel\train-from-scratch\person-hyperparameters3"  # the model name locally and on the HF Hub
    seed = 0
config = TrainingConfig()
```

Sau khi huấn luyện model, người dùng có thể chạy file code sau để test kết quả
```bash
python test-trained.py
```
**_OUTPUT TRAIN-FROM-SCRATCH_**

- **Prompt về địa điểm Việt Nam train-from-scratch**

![Imgur](https://i.imgur.com/8GQNO73.png)

- **Prompt về con người Việt Nam và các hoạt động liên quan đến con người train-from-scratch**

![Imgur](https://i.imgur.com/1OHRYzt.png)

### 3.3. Finetune model bằng dreambooth
Sau khi thực hiện các thí nghiệm với model, nhóm nghiên cứu quyết định finetune lại mô hình với dreambooth. Đây là phương pháp được đánh giá đem lại hiệu quả cao và yêu cầu thời gian huấn luyện ngắn.

Quá trình finetune model được nhóm nghiên cứu tham khảo trực tiếp từ [Blog hướng dẫn finetuning SD model](https://tryolabs.com/blog/2022/10/25/the-guide-to-fine-tuning-stable-diffusion-with-your-own-images) và [Link github](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)

Sau khi người dùng clone dreambooth từ link github về, với môi trường conda đã được set up từ đầu, người dùng cần bổ sung thêm ``accelerate`` để set up:
```bash
pip install accelerate
accelerate config
```

Đối với người dùng không có nhu cầu tự set up môi trường, có thể set up ``accelerate`` bằng bash script sau:
```bash
pip install accelerate
accelerate config default
```

Với 2 đường link được cung cấp ở trên, người dùng có thể tùy chọn sử dụng phương pháp finetune phù hợp với cấu hình của máy. Ở trong project này, nhóm nghiên cứu lựa chọn **_Training on a 8 GB GPU_**:
```bash
set MODEL_NAME="CompVis/stable-diffusion-v2-1" // Thay thế bằng path link model pre-trained bất kỳ trong máy
set INSTANCE_DIR="path-to-instance-images"
set CLASS_DIR="path-to-class-images"
set OUTPUT_DIR="D:\SE2023-9.1\SDmodel\SD-2-1-newtrained" // path link của nhóm nghiên cứu

accelerate launch --mixed_precision="fp16" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of woman smiling" \
  --class_prompt="a photo of woman smiling" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

Chi tiết Hyperparameters như sau cho người dùng muốn tinh chỉnh các siêu tham số trong mô hình

```py
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1.0e-08
adam_weight_decay: 0.01
allow_tf32: false
center_crop: false
checkpointing_steps: 500
checkpoints_total_limit: null
class_data_dir: null
class_labels_conditioning: null
class_prompt: null
dataloader_num_workers: 0
enable_xformers_memory_efficient_attention: false
gradient_accumulation_steps: 1
gradient_checkpointing: false
hub_model_id: null
hub_token: null
instance_data_dir: D:\SE2023-9.1\SDmodel\data\train\Vietnam_people
instance_prompt: a photo of Vietnam people
learning_rate: 5.0e-06
local_rank: -1
logging_dir: logs
lr_num_cycles: 1
lr_power: 1.0
lr_scheduler: constant
lr_warmup_steps: 0
max_grad_norm: 1.0
max_train_steps: 400
mixed_precision: null
num_class_images: 100
num_train_epochs: 2
num_validation_images: 4
offset_noise: false
output_dir: D:\SE2023-9.1\SDmodel\data\output\dreambooth
pre_compute_text_embeddings: false
pretrained_model_name_or_path: D:\SE2023-9.1\SDmodel\stable-diffusion-2-1
prior_generation_precision: null
prior_loss_weight: 1.0
push_to_hub: false
report_to: tensorboard
resolution: 768
resume_from_checkpoint: null
revision: null
sample_batch_size: 4
scale_lr: false
seed: null
set_grads_to_none: false
skip_save_text_encoder: false
snr_gamma: null
text_encoder_use_attention_mask: false
tokenizer_max_length: null
tokenizer_name: null
train_batch_size: 1
train_text_encoder: false
use_8bit_adam: false
validation_prompt: null
validation_scheduler: DPMSolverMultistepScheduler
validation_steps: 100
variant: null
with_prior_preservation: false
```

**_Lưu ý_: Khi cài đặt môi trường để sử dụng dreambooth finetune theo phương pháp này, người dùng cần sử dụng [DeepSpeed](https://www.deepspeed.ai/) để giảm dung lượng chạy trên VRAM khi huấn luyện. Tuy nhiên sẽ có thể xảy ra conflict với version pytorch. Chi tiết fix lỗi đã được nêu trong link của github [dreambooth](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) và người dùng có thể tham khảo.**

Sau khi train xong model, người dùng có thể chạy code sau để test
```py
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "An asian woman smiling"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("woman-smilling.png")
```

HOẶC chạy code 
```bash
python ./test-trained.py
```
_Update: File test-trained.py trong update mới nhất đã có thể chạy cả 3 model pre-trained, train-from-scratch, finetuning_

**_OUTPUT FINETUNING_**

- **Prompt về địa điểm Việt Nam finetune**

![Imgur](https://i.imgur.com/caSh2yl.png)

- **Prompt về con người Việt Nam và các hoạt động liên quan đến con người finetuning**

![Imgur](https://i.imgur.com/J0TihPj.png)

### 3.4. So sánh kết quả giữa các model
Để đánh giá 3 model, nhóm nghiên cứu thử nghiệm các prompt với những yếu tố sau:

|Độ dài Prompt            |Cấu trúc Prompt  | Chủ đề Prompt |
|:-----------------|:----------------------:|--------------:|
|Dài               | Phức tạp               | Chân dung con người    | 
|Ngắn              | Đơn giản               |Con người có thêm các hoạt động, môi trường xung quanh   |
|          ---     |          ---           |Phong cảnh              | 

Dưới đây là một số so sánh hình ảnh output của 3 model

- Prompt 1 (Prompt có cấu trúc câu đơn giản):
```an Asian woman smiling```

![Imgur](https://i.imgur.com/v8BwaWw.png)

- Prompt 2 (Prompt về con người khi đang thực hiện hoạt động:
```an Asian woman eating pho```

![Imgur](https://i.imgur.com/4Iad0D0.png)

- Prompt 3 (Prompt có cấu trúc câu phức tạp về phong cảnh chủ đề Việt Nam): 
```Render breathtaking digital art inspired by the scenic beauty of Ha Long Bay, a UNESCO World Heritage Site known for its picturesque karst formations.```

![Imgur](https://i.imgur.com/MlKJEAE.png)

- Prompt 4:
```Illustrate the emotional stories of resilience and bravery during the Vietnam War at the War Remnants Museum in Saigon, bringing history to life through digital images.```

![Imgur](https://i.imgur.com/psMs2u9.png)

- Prompt 5:
```Design digital scenes inspired by the ancient town of Hoi An, featuring its well-preserved architecture, lantern-lit streets, and traditional boats on the Thu Bon River.```

![Imgur](https://i.imgur.com/dluoqTr.png)

- Prompt 6:
```Depict the warmth and hospitality of the Vietnamese people through digital art, capturing their friendliness and eagerness to share the beauty of their homeland.```

![Imgur](https://i.imgur.com/lIE30eP.png)


#### Đối với model pre-trained:
Đánh giá quá trình training
- _Không yêu cầu train model._
  
Đánh giá chất lượng hình ảnh được generate:
- _Cho kết quả hình ảnh generate tệ nhất cả về các hình ảnh chủ đề văn hóa, địa điểm du lịch, nhà cửa mang nét văn hóa Việt Nam lẫn chủ đề liên quan tới hoạt động con người._
- _Hình ảnh con người chân dung khá tốt, tuy nhiên hoạt động và các chi tiết trong hình ảnh sinh ra chưa đa dạng._
- _Các prompt input càng có cấu trúc phức tạp, kết quả trả về càng tệ._
- _Vẫn còn tình trạng generate lẫn lộn giữa các vật thể (VD: Mặt người bị lẫn với khung cảnh xung quanh)._
- _Đối với các prompt có độ dài lớn, nhiều chi tiết, mô tả, output trả về có thể không liên quan tới prompt._

### Đối với model train-from-scratch
Đánh giá quá trình training
- _Thời gian chạy model khá lâu (Các thông số về epochs, batch size, data size tương tự với khi huấn luyện model finetune)._
- _Yêu cầu card máy tính có cấu hình cao._

Đánh giá chất lượng hình ảnh được generate"
- _Model train from scratch đã có một số cải thiện so với pre-trained model về khung cảnh, hoạt động của con người. Hình ảnh được generate ra mang bản sắc Việt Nam hơn so với mô hình pre-trained._
- _Tuy nhiên, đối với các yêu cầu generate liên quan đến con người, hình ảnh output vẫn còn tệ như khuôn mặt chưa được tự nhiên, các bộ phận biến dạng._
- _Có sự nhầm lẫn khi prompt input yêu cầu generate các hình ảnh chủ đề liên quan đến Việt Nam, mô hình trả về output hình ảnh liên quan đến đất nước khác (Hàn Quốc, Trung Quốc,...)._
- _Hình ảnh sinh ra sát với thực tế hơn so với model pre-trained._
- _Vẫn còn tình trạng generate lẫn lộn giữa các vật thể tương tự mô hình train-from-scratch._
- _Các prompt có cấu trúc phức tạp trẩ về output tốt hơn so với mô hình pre-trained nhưng vẫn chất lượng hình ảnh vẫn còn tệ._

### Đối với model finetuning với dreambooth
Đánh giá quá trình training
- _Thời gian train model ngắn hơn so với train from scratch do đã sử dụng các thư viện tối ưu (xformers, DeepSpeed,...)._
- _Yêu cầu ít bộ nhớ khi train model hơn so với model train-from-scratch._
- _Có nhiều phiên bản finetune để lựa chọn._
  
Đánh giá chất lượng hình ảnh được generate:
- _Trả về kết quả tốt nhất so với 3 phương pháp huấn luyện mô hình về tất cả các chủ đề được test: hình ảnh văn hóa, địa danh, hoạt động của con người và ảnh chân dung..._
- _Generate ra các hình ảnh về kiến trúc, văn hóa,... (không kể con người) tốt nhất trong 3 model._
- _Các hình ảnh được generate ra đều đúng với chủ đề của prompt input với các thử nghiệm._
- _Các hình ảnh generate ra có độ chân thực và chi tiết cao hơn so với 2 model còn lại._
- _Đối với prompt có chủ đề hoạt động của con người, các hoạt động trong hình ảnh được generate tự nhiên hơn so với 2 model còn lại._
- _Tuy nhiên khuyết điểm về generate hình ảnh chủ đề con người chưa được tự nhiên vẫn còn tồn tại tượng tự 2 model trên, mặc dù đã được giảm thiểu._

### 3.5. TỔNG KẾT
Nhóm nghiên cứu đưa ra đánh gíá như sau đối với việc train model SD
- Đối với mục tiêu tạo ra ảnh liên quan đến Việt Nam, đặc biệt ảnh phong cảnh, người dùng có thể dùng phương án train-from-scratch hoặc finetune đều se đem lại kết quả khá tốt.
- Đối với mục tiêu tạo ra ảnh con người Việt Nam/Châu Á, mô hình finetune tỏ ra vượt trội hơn trong việc tạo ra ảnh tự nhiên, tuy nhiên phần khuôn mặt chưa thực sự tốt.
- Về đề xuất, nhóm nghiên cứu đề xuất xây dựng bộ dataset chất lượng hơn. Đặc biệt đối với task generate hình ảnh khuôn mặt. Model finetune có thể cải thiện hơn tuy nhiên bộ dataset CelebA chưa thực sự đủ tốt.
- Dataset cần thiết để finetune gương mặt cần nhiều góc độ khuôn mặt của 1 người, thay vì đó, bộ Dataset CelebA bao gồm khuôn mặt của nhiều người, dataset chưa thực sự phù hợp cho task. Nhưng việc sử dụng cũng đã có cải thiện chất lượng hình ảnh đầu ra.

## 4. XÂY DỰNG GIAO DIỆN WEB CHO NGƯỜI DÙNG
### 4.1. Cài đặt Nodejs và npm
```Node.js```, là môi trường thời gian chạy (runtime environment) JavaScript đa nền tảng và mã nguồn mở ,được xây dựng với mô hình xử lý bất đồng bộ , cho phép các lập trình viên tạo cả ứng dụng front-end và back-end bằng JavaScript . Do đó một server có thể dễ dàng giao tiếp với frontend qua ``REST API`` bằng Node.js.

Trong đó , ```npm``` là một công cụ quản lý gói (package manager) cho Node.js,cung cấp một kho lưu trữ lớn với hàng ngàn packages có sẵn , cho phép dễ dàng cài đặt, quản lý, và chia sẻ các thư viện và công cụ.

Người dùng truy cập link [https://nodejs.org/](https://nodejs.org/) để cài đặt Nodejs .

**_Lưu ý:_** Lựa chọn phiên bản ```LTS(Long-Term Support)``` giúp đảm bảo tính ổn định và hỗ trợ dài hạn.

Sau khi cài đặt xong , có thể dùng câu lệnh sau để kiểm tra 
```bash
node -v
npm -v
```
### 4.2 Cài đặt Reactjs
```Reactjs``` là một thư viện JavaScript dành riêng để giúp các nhà phát triển tạo giao diện người dùng và giao diện người dùng. 
- **Components:** cho phép phát triển ứng dụng web theo mô hình component. Các component là các phần tử UI độc lập có thể được tái sử dụng trong nhiều phần khác nhau của ứng dụng.
- **Virtual DOM** ( một bản sao của DOM được lưu trữ trong bộ nhớ và được cập nhật một cách nhanh chóng khi có thay đổi): tối ưu hóa hiệu suất của ứng dụng.
- **JSX:** một ngôn ngữ lập trình phân biệt được sử dụng trong ReactJSđể mô tả các thành phần UI

Người dùng chạy câu lệnh sau để tạo ra 1 ứng dựng react mới
```bash
npx create-react-app reactjs
```

Sau đó , người dùng có thể chạy thử câu lệnh
```bash
cd reactjs
npm start
```
Sau đó , người dùng có thể truy cập [http://localhost:3000](http://localhost:3000) trong trình duyệt để xem ứng dụng web reactjs .

**_Lưu ý_** : Cài đặt ```ReactRouter``` để quản lý định hướng và điều hướng giữa các element của ứng dụng web
```bash
npm install react-router-dom@^6.0.0
```
Ở đây , nhóm nghiên cứu sử dụng phiên bản 6 , vậy nên cách sử dụng đã thay đổi một chút so với phiên bản trước đó. Chẳng hạn như sự thay đổi từ ``BrowserRouter`` sang ``Router``, sự thay đổi từ ``Route`` và ``Switch`` sang ``Routes`` và ``Route``, và cách sử dụng thuộc tính ``element`` thay vì ``component``.
```py
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
```
Ngoài ra , nhóm nghiên cứu cũng sử dụng ``Bootstrapv5`` ,là một framework bao gồm các HTML, CSS và JavaScript template dùng để phát triển website . ``Bootstrap`` cho phép quá trình thiết kế website diễn ra nhanh chóng và dễ dàng hơn dựa trên những thành tố cơ bản sẵn có .

Trong thư mục gốc của dự án Reactjs , người dùng sử dụng ```npm (Node Package Manager)``` để cài đặt ```Bootstrap```:
```bash
npm install bootstrap@5
```
Sau đó , thêm dòng sau để import CSS của Bootstrap 
```py
import 'bootstrap/dist/css/bootstrap.min.css';
```
![Imgur](https://i.imgur.com/iqq5wvM.png)


## 5. XÂY DỰNG DEMO APP
Vì thời gian có hạn , nhóm nghiên cứu sẽ xây dựng app demo tạo ra hình ảnh sử dụng text2img dựa theo [Stable Diffusion 2.1 Demo](https://huggingface.co/spaces/stabilityai/stable-diffusion) của ``Hugging Faces`` . Ứng dụng này sử dụng thư viên ``Gradio`` giúp:
- Tạo giao diện người dùng tương tác cho các mô hình máy học và ứng dụng AI.
- Giúp kết nối mô hình máy học với người dùng một cách trực quan và thuận tiện.
- Triển khai trên Spaces cho phép tạo và chia sẻ không gian (spaces) với mô hình, dữ liệu và giao diện người dùng tương tác.

### 5.1 Xác định biến môi trường
- Một số biến môi trường như ``USE_SSD``, ``ENABLE_LCM``,``ENABLE_REFINER``,... để quyết định cách mà mô hình sẽ được sử dụng và cấu hình.
- Dựa vào giá trị của biến ``USE_SSD``, mã sẽ chọn một trong hai đường dẫn cho ``model_key_base`` và ``model_key_refiner``.
```py
use_ssd = os.getenv("USE_SSD", "false").lower() == "tpy
print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float32, use_safetensors=True, variant="fp16")
```
### 5.3. Cấu hình mô hình
Cấu hình mô hình bằng cách sử dụng các thông số như kiểu dữ liệu của ``torch``, ``variant``, và có sử dụng ``Safetensors`` hay không.
- Thay đổi trọng số của lớp down.weight trong mô hình.
```py
new_down_weight = torch.randn((64, 1024))  # optional - Có thể làm đánh mất dữ liệu trong một số trường hợp
pipe.unet.state_dict()['down.weight'] = new_down_weight
```
- Nếu ``enable_lcm`` được bật, tải trọng số của LCM LoRA và cấu hình mô hình với LCMScheduler.from_config.
```py
 if enable_lcm:
    pipe.load_lora_weights(lcm_lora_id)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
``` 
- Nếu Nếu sử dụng nhiều ``GPU`` (MULTI_GPU), sử dụng ``UNetDataParallel`` để kết hợp nhiều GPU cho mô hình . Cho phép việc chuyển đổi giữa CPU và GPU để giảm tải , đảm bảo mô hình được chuyển đến thiết bị ``GPU`` hoặc ``CPU`` tương ứng.
```py
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
```
### 5.4. Cấu hình giao diện người dùng Gradio
- Sử dụng biến môi trường để xác định giá trị mặc định cho số lượng ảnh ``default_num_images``, giá trị mặc định là 4.
```py
default_num_images = int(os.getenv("DEFAULT_NUM_IMAGES", "4"))
if default_num_images < 1:
    default_num_images = 1
```
- ``infer``để thực hiện dự đoán với các đối số như ``prompt``, ``negative``, ``scale``, số lượng mẫu, bước, sức mạnh của ``refiner``, và ``seed``.

- ``submit`` để đặt mối liên kết giữa các thành phần của giao diện và ``infer``.
```py
negative.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
        text.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=[text, negative, guidance_scale, samples, steps, refiner_strength, seed], outputs=[gallery], postprocess=False)
```
- Ngoài ra , button ``Share to commutnity`` và ``Advanced Settings`` để tương tác với cộng đồng.
- Chi tiết hơn, sau khi người dùng clone project về, giải nén folder app và chạy dòng lệnh sau
```bash
type ./app/app.py
```

### 5.5. Chạy app
Người dùng chạy bash script sau
```bash
python -r requirements.txt // Cài đặt các thư viện yêu cầu. Recommend người dùng tạo environment khác để tránh bị conflict với các version thư viện trong environment dùng để huấn luyện model
python ./app/app.py
```

Sau khi chạy file code, người dùng sẽ có 2 lựa chọn
- **Chạy trên localhost:** Link mặc đình để chạy app trên localhost là: **[_https://local.host/127.0.0.1_](https://locall.host/127.0.0.1/)**
- **Chạy trên link public:** Link public sẽ được Gradio generate random mỗi lần chạy app và hiển thị trên màn hình ```cmd``` của người dùng
  
![Imgur](https://i.imgur.com/Xe9nPSp.png)
