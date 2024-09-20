from diffusion.models import unet
from diffusion.nn.embedding import SinusoidalEmbedding, RandomOrLearnedSinusoidalEmbedding, TimeEmbedding

def build_model(cfg):
    if cfg['name'] == 'unet':
        model = unet.UNet(**cfg['model_params'])
    else:    
        raise NotImplementedError(f"Model {cfg['model']} is not implemented")
    
    if cfg['time_emb']['name'] == 'sine':
        time_emb = TimeEmbedding(**cfg['time_emb']['time_emb_params'])
    else:
        raise NotImplementedError(f"Time embedding {cfg['time_emb']['name']} is not implemented")
    
    return model, time_emb
        