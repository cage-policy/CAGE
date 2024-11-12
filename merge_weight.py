import os
from argparse import ArgumentParser

import safetensors.torch as st
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModel

from train_cage import initialize_model


def merge_weights(model, ckpt, dest=None):
    # load the main model
    missing, unexpected = st.load_model(model, os.path.join(ckpt, 'model.safetensors'), strict=False)
    missing_keys = []
    for k in missing:
        parts = k.split('.')
        if parts[0]!='obs_encoders' or len(parts)<=2 or parts[2]!='model':
            missing_keys.append(k)
    if len(missing_keys)>0 or len(unexpected)>0:
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected)
        print('The config file and the checkpoint are not compatible!')
        return

    # load the lora parameters for obs_encoders
    vision_backbone_dir = os.path.join('./weights', conf.model.image_encoder.name)
    for k, v in model.obs_encoders.items():
        if isinstance(v.model, PeftModel):
            v.model = PeftModel.from_pretrained(AutoModel.from_pretrained(vision_backbone_dir), os.path.join(ckpt, f'{k}_encoder_lora'))
    
    # save the merged model
    if dest is None:
        dest = os.path.join(ckpt, os.pardir, 'merged_model.safetensors')
    st.save_model(model, dest)

def unmerge_weights(model, ckpt, dest=None):
    if ckpt.endswith('.bin'):
        # in pytorch format
        model.load_state_dict(torch.load(ckpt))
    else:
        # in safetensors format
        st.load_model(model, ckpt)

    if dest is None:
        dest = os.path.dirname(ckpt)
    save_path = os.path.join(dest, 'unmerged_ckpt')
    os.mkdir(save_path)

    # if the encoders use lora, save only the lora parameters
    lora_dict = {}
    for k, v in model.obs_encoders.items():
        if isinstance(v.model, PeftModel):
            lora_dict[k] = v.model
            v.model = None

    # save the main part of the model and lora(optional) separately
    st.save_model(model, os.path.join(save_path, 'model.safetensors'))
    for k, v in lora_dict.items():
        v.save_pretrained(os.path.join(save_path, f'{k}_encoder_lora'))


if __name__ == '__main__':
    parser = ArgumentParser(description='Merge/Unmerge the weights of the trained policy')
    parser.add_argument('--config', type=str, required=True, help='the config file of the policy')
    parser.add_argument('--ckpt', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--dest', type=str, default=None, help='path to save the merged/unmerged weights, default to the same directory as the checkpoint')

    parser.add_argument('--unmerge', action='store_true', help='flag to unmerge')

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    model = initialize_model(conf)
    if args.unmerge:
        unmerge_weights(model, args.ckpt, args.dest)
    else:
        merge_weights(model, args.ckpt, args.dest)