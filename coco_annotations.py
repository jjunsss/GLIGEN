from pycocotools.coco import COCO
import random
from tqdm import tqdm
import time
import json
import torch
import os

from torch import BoolTensor, FloatTensor, LongTensor
from typing import Dict, List, Optional, Tuple, Union

import dist as utils

def coco_loader(coco_dir):
    
    with open(coco_dir, 'r') as f:
        data = json.load(f)
        
    generate_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'images': data["images"],
            'annotations': data["annotations"],
            'categories': data['categories']
        }

    return generate_data

def gen_label_id():
    count = 1
    while True:
        yield count
        count += 1


def filter_annotations_and_images(generate_data, category_range=(1, 10)):
    # Filter annotations
    new_annotations = []
    
    pbar = tqdm(total=len(generate_data['annotations']), desc="Processing annoatations", disable=not utils.is_main_process())
    for annotation in generate_data['annotations']:
        if annotation['category_id'] in range(category_range[0], category_range[1] + 1):
            new_annotations.append(annotation)
            
        pbar.update(1)
    
    generate_data['annotations'] = new_annotations
    
    # Find image ids that have annotations
    valid_image_ids = set([anno['image_id'] for anno in new_annotations])
    
    # Filter images
    new_images = []
    for image in generate_data['images']:
        if image['id'] in valid_image_ids:
            new_images.append(image)
    
    generate_data['images'] = new_images
    
    return generate_data

from copy import deepcopy
def flip_annotations(generate_data, coco, flip_version="horizontal"):
    new_annotations = []
    flipped_annotations_lr = []
    flipped_annotations_ud = []
    
    for annotation in tqdm(generate_data['annotations'], desc="Flipping annotations", disable=not utils.is_main_process()):
        new_annotations.append(annotation)
        image_info = coco.loadImgs(annotation["image_id"])
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Flip left-right
        if flip_version == "horizontal":
            flipped_annotation_lr = deepcopy(annotation)
            flipped_annotation_lr['bbox'][0] = image_width - (annotation['bbox'][0] + annotation['bbox'][2])
            flipped_annotations_lr.append(flipped_annotation_lr)
        
        # Flip up-down
        if flip_version == "vertical":
            flipped_annotation_ud = deepcopy(annotation)
            flipped_annotation_ud['bbox'][1] = image_height - (annotation['bbox'][1] + annotation['bbox'][3])
            flipped_annotations_ud.append(flipped_annotation_ud)
    
    if flip_version == "horizontal":
        generate_data['annotations'].extend(flipped_annotations_lr)
    else:
        generate_data['annotations'].extend(flipped_annotations_ud)
    
    return generate_data

def _log_dataset(info_dict):
    # ... (기존 코드)

    # 딕셔너리를 JSON 형식으로 텍스트 파일에 저장
    txt_file_path = "./prompts_and_info.json"  # 저장할 텍스트 파일의 경로
    with open(txt_file_path, 'a') as f:
        json_str = json.dumps(info_dict)
        f.write(json_str + "\n")


from PIL import Image
def make_meta_dict(coco, sample, info, max_length, blip_processor, blip_model):
    image_id = info["id"]
    image_width, image_height = info["width"], info["height"]
    image_path = os.path.join('/home/user/sumin/paper/COCODIR/train2017', info['file_name'])
    image = Image.open(image_path)
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    
    labels = [ann['category_id'] for ann in annotations]
    
    if len(labels)  > max_length : 
        return None
    
    text_entities = [coco.cats[x]["name"] for x in labels]
    bounding_boxes = [ann['bbox'] for ann in annotations]
    
    normalized_bounding_boxes = []
    for bbox in bounding_boxes:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        x1 /= image_width
        y1 /= image_height
        x2 /= image_width
        y2 /= image_height
        normalized_bounding_boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
    
    prompt = "NULL"
    entity_str = '-'.join(text_entities)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if prompt == "NULL" or None or "":
        inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip_model.generate(**inputs, max_new_tokens=20)
        generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        prompt = generated_text
        
    # file path fixed to the original path
    # save_folder_name = f"generation_box_text/gen_text-gen_layout/{entity_str}"
    save_folder_name = f"/data/gen_dataset/images"
    
    temp_meta_dict = deepcopy(sample) #wanna generate more examples ( about various input token )
    temp_meta_dict["phrases"] = text_entities
    temp_meta_dict["prompt"] = prompt
    temp_meta_dict["locations"] = normalized_bounding_boxes
    temp_meta_dict["save_folder_name"] = save_folder_name
    temp_meta_dict["image_id"] = f"{image_id:012d}"
    _log_dataset(temp_meta_dict)
    
    return temp_meta_dict