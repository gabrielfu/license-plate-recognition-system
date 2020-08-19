# LPR

v 1.0 (unstable)

# Speed (T4, 4vCPU):
Car detection (trt)
Plate detection (trt)
Char segmentation (torch)
Char recognition (torch)
------------------------------
1 video input
=> camera fps: 59
=> no trigger loop: ~0.12s
=> 1 trigger loop: ~1.2s-3s (not stable)
------------------------------
2 videos input
=> camera fps: 58
=> no-car loop: ~0.18s
=> 1 trigger loop: ~1.1s-1.3s
------------------------------
3 videos input
=> camera fps: 51
=> no-car loop: ~0.23s
=> 1 trigger loop: ~1.1s-1.3s


# Speed (T4, 4vCPU):
Car detection (trt)
Plate detection (torch)
Char segmentation (torch)
Char recognition (torch)
------------------------------
3 videos input
=> camera fps: 51
=> no-car loop: ~0.23s
=> 1 trigger loop: ~1.5s-2s

# Speed (T4, 4vCPU):
Car detection (torch)
Plate detection (torch)
Char segmentation (torch)
Char recognition (torch)
------------------------------
3 videos input
=> camera fps: 51
=> no-car loop: ~0.7s
=> 1 trigger loop: ~2s-3s