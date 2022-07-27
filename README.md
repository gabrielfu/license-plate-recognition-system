# Real Time License Plate Recognition System (LPRS)



## Features
- [X] 97% accuracy on Hong Kong vehicles license plate, highest in the market
- [X] 55 fps running on NVIDIA T4 GPU, 4vCPU with TensorRT
- [X] Handles low light environment, glares, blurs, wet lens
- [X] Support inference on both IP cameras and .mp4 video files
- [X] Pushes recognition result to a local Kafka topic
- [X] CUDA support
- [ ] TensorRT support (work in progress)


## Background
This was an on-premise project that I did with my partners for a car park in Hong Kong in 2020. 
This software would run on a GPU machine, connect to the IP cameras at the entrance of a car park
and provide real time license plate recognition service on any vehicles entering the car park.

We decided to open source this project in 2022. I took this opportunity to tidy up the code and deployment.

## Usage

### Model Weights
First, go to `....` to download the model weights and put them inside `./data`.

The weights are trained on a proprietary Hong Kong license plate dataset.
This repository does not contain training code of the models.
If you would like to run on other countries' license plates, 
please retrain the models on your dataset. 

### Run on Docker
To run, simply run 
```shell
docker-compose up -d
```

## Configurations


# Speed (T4, 4vCPU):
Car detection (trt)
Plate detection (trt)
Char segmentation (torch)
Char recognition (torch)
-----------------------------
3 videos input, num_votes=8
=> camera fps: 55
=> no lpr prediction loop: ~0.18s
=> 1 lpr prediction loop: ~0.9s
