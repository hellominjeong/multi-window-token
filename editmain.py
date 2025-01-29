import torch
import torch.utils.benchmark as benchmark
from diffusers import StableDiffusionPipeline
import os
import shutil
import json
from PIL import Image
import torch.nn as nn
from torchvision.models import inception_v3
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import linalg
from tqdm import tqdm
from tomesd import apply_patch, remove_patch

def load_ordered_coco_pairs(annotation_file, num_pairs=400):
    """처음 num_pairs개의 annotation을 순서대로 가져옴"""
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)
    
    annotations = coco_data["annotations"][:num_pairs]
    pairs = [(str(annot["image_id"]).zfill(12) + ".jpg", annot["caption"]) 
            for annot in annotations]
    
    print(f"Loaded first {len(pairs)} annotations in original order")
    return pairs

class OrderedImageDataset(Dataset):
    """이미지를 지정된 순서대로 로드하는 데이터셋"""
    def __init__(self, image_dir, image_files):
        self.image_paths = [os.path.join(image_dir, x) for x in image_files]
        
        self.transforms = TF.Compose([
            TF.Resize(299),
            TF.CenterCrop(299),
            TF.ToTensor(),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transforms(image)
        except Exception as e:
            print(f"Error loading image: {self.image_paths[idx]}")
            raise e

def copy_ordered_images(pairs, src_dir, dst_dir):
    """원본 이미지를 순서를 유지하며 복사"""
    os.makedirs(dst_dir, exist_ok=True)
    copied_files = []
    
    for image_file, _ in pairs:
        src_path = os.path.join(src_dir, image_file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dst_dir, image_file))
            copied_files.append(image_file)
    
    return copied_files

generator = torch.Generator("cuda:2").manual_seed(42)
def generate_images(pipe, prompts, save_dir, config=None):
    """프롬프트 순서대로 이미지 생성"""
    print("Starting generation...")  #
    if config is not None:
        apply_patch(
            pipe,
            ratio=(config["ratio"]), 

            use_rand=config["use_rand"],
            max_downsample=config["max_downsample"],
            merge_attn=config["merge_attn"],
            merge_crossattn=config["merge_crossattn"],
            merge_mlp=config["merge_mlp"]
        )
        print("Patch applied successfully")   
    else:
        remove_patch(pipe)
    
    os.makedirs(save_dir, exist_ok=True)
    
    
    generated_files = []
    for idx, (img_file, prompt) in enumerate(tqdm(prompts, desc="Generating images")):
        print(f"Generating image {idx+1}/{len(prompts)}") 
        output_filename = f"{os.path.splitext(img_file)[0]}.png"
        save_path = os.path.join(save_dir, output_filename)
        if os.path.exists(save_path):
            print(f"Skipping {output_filename}, already exists.")
            # generated_files.append(output_filename)
            continue
        try:
            image = pipe(
                prompt,
                generator=generator,
                height=resolution,
                width=resolution,
                num_inference_steps=steps,
                num_images_per_prompt=1,
            ).images[0]
            
            image.save(os.path.join(save_dir, output_filename))
            generated_files.append(output_filename)
            print(f"Successfully generated and saved image {idx+1}")  # 생성 성공
        except Exception as e:
            print(f"Error generating image {idx+1}: {str(e)}")  # 오류 발생시
            raise e
        if config is not None and idx == 0:
            print_model_stats(pipe, resolution)
                    
    return generated_files


def print_model_stats(pipe, resolution):
    """모델 통계 출력"""
    if hasattr(pipe.unet, '_tome_info'):
        info = pipe.unet._tome_info["args"]
        print("\nWindow-based ToMe Configuration:")
        print(f"Merge Ratios: {info['ratio']}")
        print(f"Original tokens: {resolution * resolution // 64}")
        estimated_tokens = resolution * resolution // 64
        ratio = info['ratio']
        estimated_tokens = int(estimated_tokens * (1 - ratio))
        print(f"Estimated final tokens: {estimated_tokens}")
        print("-" * 50)


def calculate_fid(dir1, dir2, image_files, device='cuda:2'):
    """순서가 보장된 FID 계산"""
    print(f"Calculating FID between {dir1} and {dir2}")
    
    def get_proper_extension(directory, filename):
        if directory == 'real_images':
            return f"{os.path.splitext(filename)[0]}.jpg"
        else:
            return f"{os.path.splitext(filename)[0]}.png"
    
    dir1_files = [get_proper_extension(dir1, f) for f in image_files]
    dir2_files = [get_proper_extension(dir2, f) for f in image_files]
    
    model = inception_v3(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    
    dataset1 = OrderedImageDataset(dir1, dir1_files)
    dataset2 = OrderedImageDataset(dir2, dir2_files)
    
    assert len(dataset1) == len(dataset2), \
           f"Number of images must match! {len(dataset1)} vs {len(dataset2)}"
    
    loader1 = DataLoader(dataset1, batch_size=32, num_workers=4)
    loader2 = DataLoader(dataset2, batch_size=32, num_workers=4)
    
    features1, features2 = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader1, desc=f"Processing {dir1}"):
            features1.append(model(batch.to(device)).cpu().numpy())
        for batch in tqdm(loader2, desc=f"Processing {dir2}"):
            features2.append(model(batch.to(device)).cpu().numpy())
    
    features1 = np.concatenate(features1, axis=0)
    features2 = np.concatenate(features2, axis=0)
    
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)
if __name__ == "__main__":
    # 기본 설정
    original_images_path = "./train2017"
    annotation_file = "./annotations/captions_train2017.json"
    
    model_id = "runwayml/stable-diffusion-v1-5"
    steps = 25
    resolution = 512

    window_configs =        {
            "name": "window8_7",
            "use_rand": True,
            "ratio": 0.4,
            "max_downsample": 1,
            "merge_attn": True,
            "merge_crossattn": False,
            "merge_mlp": False
        }

    # 데이터 준비
    print("Loading image-caption pairs in order...")
    pairs = load_ordered_coco_pairs(annotation_file, num_pairs=400)
    
    # 디렉토리 설정
    real_dir = "real_images"
    image_files = copy_ordered_images(pairs, original_images_path, real_dir)
    print(f"Copied {len(image_files)} original images")
    
    # 모델 설정
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        safety_checker=None
    ).to("cuda:2")
    pipe.set_progress_bar_config(disable=True)
    
    # 각 설정별로 실험 수행
    results = {}
    
    # for config in window_configs:
    #     print(f"\nTesting Swin-ToMe configuration: {config['name']}")
    #     tome_dir = f"swin_tome_images_{config['name']}"
        
    #     # 이미지 생성
    #     tome_files = generate_images(pipe, pairs, tome_dir, config=config)
        
    #     # 성능 측정
    #     # 1. FID 계산
    #     fid_score = calculate_fid(tome_dir, real_dir, image_files)
        
    #     # 2. 속도 측정
    #     timer = benchmark.Timer(
    #         stmt='generate_single_image(pipe, prompt)',
    #         globals={'pipe': pipe, 'prompt': pairs[0][1], 
    #                 'generate_single_image': lambda p, prompt: p(
    #                     prompt,
    #                     height=resolution,
    #                     width=resolution,
    #                     num_inference_steps=steps,
    #                     num_images_per_prompt=1,
    #                 )}
    #     )
    #     speed_benchmark = timer.timeit(10)

    #     # 표준편차 대신 min/max 사용
    #     results[config['name']] = {
    #         'fid': fid_score,
    #         'speed_mean': speed_benchmark.mean,
    #         # 'speed_min': speed_benchmark.min,
    #         # 'speed_max': speed_benchmark.max,
    #     }

        
    #     print(f"FID Score: {fid_score}")
    #     print(f"Average Generation Time: {speed_benchmark.mean:.4f}s")
    print(f"\nTesting Swin-ToMe configuration: {window_configs['name']}")
    tome_dir = f"swin_tome_images_{window_configs['name']}"
    
    # 이미지 생성
    tome_files = generate_images(pipe, pairs, tome_dir, config=window_configs)
    
    # 성능 측정
    # 1. FID 계산
    fid_score = calculate_fid(tome_dir, real_dir, image_files)
    
    # 2. 속도 측정
    timer = benchmark.Timer(
        stmt='generate_single_image(pipe, prompt)',
        globals={'pipe': pipe, 'prompt': pairs[0][1], 
                'generate_single_image': lambda p, prompt: p(
                    prompt,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=steps,
                    num_images_per_prompt=1,
                )}
    )
    speed_benchmark = timer.timeit(10)

    # 표준편차 대신 min/max 사용
    results[window_configs['name']] = {
        'fid': fid_score,
        'speed_mean': speed_benchmark.mean,
        # 'speed_min': speed_benchmark.min,
        # 'speed_max': speed_benchmark.max,
    }

    
    print(f"FID Score: {fid_score}")
    print(f"Average Generation Time: {speed_benchmark.mean:.4f}s")

    # 결과 저장
    with open('window_tome_results.json', 'w') as f:
        json.dump(results, f, indent=4)

