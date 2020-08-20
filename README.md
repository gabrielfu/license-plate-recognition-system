# LPR

v 1.0 (unstable)

# Speed (T4, 4vCPU):
Car detection (trt)
Plate detection (trt)
Char segmentation (torch)
Char recognition (torch)
-----------------------------
3 videos input
=> camera fps: 55
=> no lpr prediction loop: ~0.15s
=> 1 lpr prediction loop: ~0.6s-1.2s (usually ~0.9)