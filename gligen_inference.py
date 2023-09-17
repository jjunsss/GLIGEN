import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import dist as utils
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import json
device = "cuda"


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"], strict=False )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image



@torch.no_grad()
def prepare_batch_kp(meta, batch=1, max_persons_per_image=8):
    
    points = torch.zeros(max_persons_per_image*17,2)
    idx = 0 
    for this_person_kp in meta["locations"]:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()

    out = {
        "points" : points.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
    }

    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_hed(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    hed_edge = Image.open(meta['hed_image']).convert("RGB")
    hed_edge = crop_and_resize(hed_edge)
    hed_edge = ( pil_to_tensor(hed_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "hed_edge" : hed_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_canny(meta, batch=1):
    """ 
    The canny edge is very sensitive since I set a fixed canny hyperparamters; 
    Try to use the same setting to get edge 

    img = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img,100,200)
    edges = PIL.Image.fromarray(edges)

    """
    
    pil_to_tensor = transforms.PILToTensor()

    canny_edge = Image.open(meta['canny_image']).convert("RGB")
    canny_edge = crop_and_resize(canny_edge)

    canny_edge = ( pil_to_tensor(canny_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "canny_edge" : canny_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_depth(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    depth = Image.open(meta['depth']).convert("RGB")
    depth = crop_and_resize(depth)
    depth = ( pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5

    out = {
        "depth" : depth.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def prepare_batch_normal(meta, batch=1):
    """
    We only train normal model on the DIODE dataset which only has a few scene.

    """
    
    pil_to_tensor = transforms.PILToTensor()

    normal = Image.open(meta['normal']).convert("RGB")
    normal = crop_and_resize(normal)
    normal = ( pil_to_tensor(normal).float()/255 - 0.5 ) / 0.5

    out = {
        "normal" : normal.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 





def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb

@torch.no_grad()
def prepare_batch_sem(meta, batch=1):

    pil_to_tensor = transforms.PILToTensor()

    sem = Image.open( meta['sem']  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 
    sem = TF.center_crop(sem, min(sem.size))
    sem = sem.resize( (512, 512), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
    try:
        sem_color = colorEncode(np.array(sem), loadmat('color150.mat')['colors'])
        Image.fromarray(sem_color).save("sem_vis.png")
    except:
        pass 
    sem = pil_to_tensor(sem)[0,:,:]
    input_label = torch.zeros(152, 512, 512)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    out = {
        "sem" : sem.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 

def check_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 겹치는 조건을 확인
    return x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2

@torch.no_grad()
def run(meta, config, starting_noise=None):

    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])



    # - - - - - update config from args - - - - - # 
    config.update( vars(args) )
    config = OmegaConf.create(config)


    # - - - - - prepare batch - - - - - #
    if "keypoint" in meta["ckpt"]:
        batch = prepare_batch_kp(meta, config.batch_size)
    elif "hed" in meta["ckpt"]:
        batch = prepare_batch_hed(meta, config.batch_size)
    elif "canny" in meta["ckpt"]:
        batch = prepare_batch_canny(meta, config.batch_size)
    elif "depth" in meta["ckpt"]:
        batch = prepare_batch_depth(meta, config.batch_size)
    elif "normal" in meta["ckpt"]:
        batch = prepare_batch_normal(meta, config.batch_size)
    elif "sem" in meta["ckpt"]:
        batch = prepare_batch_sem(meta, config.batch_size)
    else:
        batch = prepare_batch(meta, config.batch_size)
    context = text_encoder.encode(  [meta["prompt"]]*config.batch_size  )
    uc = text_encoder.encode( config.batch_size*[""] )
    if args.negative_prompt is not None:
        uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )


    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 


    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 
    if "input_image" in meta:
        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes( batch['boxes'], model.image_size  ).cuda()
        
        input_image = F.pil_to_tensor( Image.open(meta["input_image"]).convert("RGB").resize((512,512)) ) 
        input_image = ( input_image.float().unsqueeze(0).cuda() / 255 - 0.5 ) / 0.5
        z0 = autoencoder.encode( input_image )
        
        masked_z = z0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
    

    # - - - - - input for gligen - - - - - #
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input = dict(
                x = starting_noise, 
                timesteps = None, 
                context = context, #prompt
                grounding_input = grounding_input, #given additional input
                inpainting_extra_input = inpainting_extra_input,
                grounding_extra_input = grounding_extra_input,
            )

    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)


    # - - - - - save - - - - - #
    output_folder = os.path.join( args.folder,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.batch_size))
    print(image_ids)
    
    bbox = grounding_input["boxes"]
    padding_value = torch.zeros(4).cuda()
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    for idx, (image_id, sample) in enumerate(zip(image_ids, samples_fake)):
        img_name = str(meta["image_id"])+".jpg"
        # img_name = str(int(image_id))+'.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        draw = ImageDraw.Draw(sample)

        # # bbox를 이미지에 추가
        # for j in range(bbox.shape[1]):
        #     # padding이 아닌 bbox만 시각화
        #     color = colors[j % len(colors)]
        #     if not torch.equal(bbox[idx][j], padding_value):
        #         x1, y1, x2, y2 = (bbox[idx][j].cpu() * sample.size[0]).to(dtype=torch.int32)
        #         draw.rectangle([x1, y1, x2, y2], outline =color)
                
        sample.save(  os.path.join(output_folder, img_name)   )
        
from copy import deepcopy
from pycocotools.coco import COCO
import os
import cv2
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from coco_annotations import coco_loader, filter_annotations_and_images, make_meta_dict
from tqdm import tqdm
import torch.distributed as dist



def custom_dataset(sample, max_length, blip_model, blip_processor):
    coco_json_path = '/home/user/sumin/paper/COCODIR/annotations/instances_train2017.json' 
        
    coco = COCO(coco_json_path)
    new_train_json = coco_loader(coco_json_path)
    new_train_json = filter_annotations_and_images(new_train_json, )
    image_length = len(new_train_json["images"])
    
    #ddp divided
    rank = utils.get_rank()
    world_size = utils.get_world_size()
    per_image_length = image_length / world_size
    start_idx = rank * int(per_image_length)
    end_idx = start_idx + per_image_length if rank != world_size - 1 else image_length
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    my_slice = new_train_json["images"][start_idx:end_idx]

    meta_list = []
    filtered_images = []
    filtered_annotations = []

    image_bar = tqdm(total=len(my_slice), desc="Processing annotations", disable=not utils.is_main_process())
    for image_info in my_slice:
        image_bar.update(1)
        temp_gligen_dict = make_meta_dict(coco, sample, image_info, max_length, blip_processor, blip_model)
        
        if temp_gligen_dict is None:    
            # Remove this image_info and continue
            continue
        else:
            image_info["prompt"] = temp_gligen_dict["prompt"]
            filtered_images.append(image_info)
            # Assuming that you have an image id field in your image_info
            image_id = image_info['id']
            corresponding_annotations = [anno for anno in new_train_json["annotations"] if anno['image_id'] == image_id]
            filtered_annotations.extend(corresponding_annotations)
            meta_list.append(temp_gligen_dict)
            
            break

    # save distrivuted meta and json information
    dist.barrier() #sync
    new_train_json["images"] = filtered_images  
    new_train_json["annotations"] = filtered_annotations
    
    save_path = "/data/gen_dataset/annotations"
    utils.save_each_file(save_path, "train", new_train_json)
    utils.save_each_file(save_path, "meta", meta_list)
    dist.barrier() #sync
    
    # load each meta and json for integration
    # if utils.is_main_process():
    #     json_file = load_file()
    #     meta_file = load_file()
        
    # # save total json and meta information
    # save_each_file(save_path, "train", new_train_json)
    # save_each_file(save_path, "meta", meta_list)
    # if utils.is_main_process() :    
    #     with open(save_json, 'w') as f:  # Use 'w' to overwrite the file
    #         json_str = json.dumps(new_train_json)
    #         f.write(json_str)
            
    #     with open(save_meta, 'w') as f:  # Use 'w' to overwrite the file
    #         json_str = json.dumps(meta_list)
    #         f.write(json_str)
        
    return meta_list
    
if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    # parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--negative_prompt", type=str,  default="blurry, overlapping objects, distorted proportions, obscured faces, low visibility, unnatural colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", help="")
    parser.add_argument("--max_length", type=int, default=5, help="")
    args = parser.parse_args()
    
    utils.init_distributed_mode(args)

    meta_list = [ 
        # - - - - - - - - GLIGEN on text grounding for generation - - - - - - - - # 
        dict(
            ckpt = "./gligen_checkpoints/checkpoint_generation_text.pth",
            prompt = "a teddy bear sitting next to a bird",
            phrases = ['a teddy bear', 'a bird'],
            locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8] ],
            alpha_type = [0.3, 0.0, 0.7],
            save_folder_name="generation_box_text"
        ), 

    ]
    
    # blip-2 for generating prompt
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    # if isinstance(pick_the_ids, int) :
    #     new_meta_list.append(custom_dataset(meta_list[0], pick_the_ids, 5, model, processor))
    # else:
    #     for idx in pick_the_ids:
    #         new_meta_list.append(custom_dataset(meta_list[0], idx, 5, model, processor))
    new_meta_list = custom_dataset(meta_list[0], args.max_length, model, processor)
    # new_meta_list = loading_dataset(meta_list[0], args.max_length, model, processor)
    # list refiner
    new_meta_list = [data for data in new_meta_list if data is not None]
    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    starting_noise = None
    torch.cuda.empty_cache()

    total_size = len(new_meta_list)
    per_process_size = total_size // world_size
    
    start_idx = rank * per_process_size
    end_idx = start_idx + per_process_size if rank != world_size - 1 else total_size
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    my_slice = new_meta_list[start_idx:end_idx]

    # Now each process will only process its own slice
    for meta in my_slice:
        run(meta, args, starting_noise)
        

    utils.cleanup()