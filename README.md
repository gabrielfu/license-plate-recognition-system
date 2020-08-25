# LPR

v 1.0 (unstable)

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
