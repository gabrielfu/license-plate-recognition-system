from turbojpeg import TurboJPEG
from typing import List
from fastapi import FastAPI, Request, File, UploadFile
import uvicorn

import time
import logging
import torch

from src.models.lpr import LPR
from src.utils.utils import read_yaml

# FastAPI & TurboJPEG devices
app = FastAPI()
jpeg = TurboJPEG()

# Configs
app_cfg = read_yaml('config/app.yaml')
models_cfg = read_yaml('config/models.yaml')
logger_cfg = read_yaml('config/logger.yaml')
use_trt = app_cfg['use_trt']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'Using device: {device}')

lpr = LPR(models_cfg, use_trt)

def decode(encoded):
    ''' decode an image with TurboJPEG '''
    global jpeg
    return jpeg.decode(encoded)    

def do_inference(imgs):
    global lpr
    return lpr.predict(imgs)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/lpr")
def batch_lpr(files: List[bytes] = File(...)):
    
    print(f'received {len(files)} images')
    
    dec_start = time.time()
    decoded = [jpeg.decode(f) for f in files]
    dec_end = time.time()

    pred_start = time.time()
    plate_nums = do_inference(decoded)
    pred_end = time.time()
    
    resp = {'message':'success', 
            'value': plate_nums,
            'decoding time': f'{dec_end-dec_start:.4f} s',
            'prediction time': f'{pred_end-pred_start:.4f} s'  
            }

    return resp

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info", reload=True)