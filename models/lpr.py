from .plate_detector import PlateDetector
from .segmentator import Segmentator
from .char_recognizer import CharRecognizer

class LPR():
    def __init__(self, cfg):
        self.detector = PlateDetector(cfg['plate_detector'])
        self.segmentator = Segmentator(cfg['segmentator'])
        self.recognizer = CharRecognizer(cfg['char_recognizer'])

        # How much to pad in detected plates before segmenting (0~1)
        self.pad_x = cfg['lpr']['pad_x']
        self.pad_y = cfg['lpr']['pad_y']

    def predict(self, frames):
        '''
        Inputs
            frames: list of frames (np.array or tensors)
        Outputs
            res: [
                    # each frame (empty list if no plate in the frame)
                    [
                        # each plate in the frame
                        {
                            'plate': {
                                'coords': [x1, y1, x2, y2],
                                'confidence': ...
                            },
                            'plate_num': {
                                'numbers': ...,
                                'confidence': ...
                            },
                            'status': '...'
                        }
                    ]
                 ]
        '''

        # Batch detect plates
        batch_preds = self.detector.predict(frames)
        batch_plates = [] # to store coords in each frame as list of list
        batch_plates_imgs = [] # to store cropped plates as flattened list

        # For each frame
        for preds, frame in zip(batch_preds, frames):
            plates = []
            if preds is not None:
                h, w = frame.shape[:2]
                # For each plate in the frame
                for pred in preds:
                    x1,y1,x2,y2,conf,_,_ = pred
                    # Pad extra space to detected plate img
                    pad_w = int((x2-x1)*self.pad_x)
                    pad_h = int((y2-y1)*self.pad_y)
                    y1 = int(max(y1-pad_h,0))
                    y2 = int(min(y2+pad_h,h))
                    x1 = int(max(x1-pad_w,0))
                    x2 = int(min(x2+pad_w,w))
                    plates.append([x1,y1,x2,y2, conf])
            batch_plates.append(plates)
            batch_plates_imgs.extend(frame[y1:y2, x1:x2] for x1,y1,x2,y2,_ in plates)

        # record number of plates in each frame
        # so that we can map the predicted plate number to its corresponding frame
        batch_plates_len = [len(sl) for sl in batch_plates]

        # batch predict segmentation 
        batch_preds = self.segmentator.predict(batch_plates_imgs)

        # Final result as a json format-like dict
        output = [[] for _ in range(len(frames))]

        # For each plate
        for i, preds in enumerate(batch_preds):
            chars = []
            # Predict plate num from CharRecognizer
            if preds is not None:
                for box in preds:
                    x1,y1,x2,y2 = box
                    chars.append(batch_plates_imgs[i][y1:y2, x1:x2])
                plate_num, confidence = self.recognizer.predict(chars)
                status = 'success'
            # No chars in the plate
            else:
                plate_num = ''
                confidence = 0.0
                status = 'no characters segmented'
            
            # Retrieve all information for output
            img_idx, plate_idx = self.find_frame_by_plate_idx(i, batch_plates_len)
            plate_coords_conf = batch_plates[img_idx][plate_idx]

            output[img_idx].append({
                'plate': {
                    'coords': plate_coords_conf[:4],
                    'confidence': plate_coords_conf[-1]
                },
                'plate_num': {
                    'numbers': plate_num,
                    'confidence': confidence
                },
                'status': status
            })

        return output

    @staticmethod
    def find_frame_by_plate_idx(target_plate_idx, batch_plates_len):
        '''
        Inputs
            target_plate_idx: idx of plate in the flattened list of all plates
            batch_plates_len: number of plates in each frames
        Outputs:
            i: idx of frame
            p: idx of plate in that frame

        E.g.
            batch_plates_len == [2,0,4]
                Meaning that there are 2 plates in first image, no plate in second image and 4 plates in third image
            target_plate_idx == 5
                Meaning that we want the 6th plate (indexing from 0)
            Output == (2, 3)
                Meaning that it is the 4th plate in the 3rd image (indexing from 0)
        '''
        i = 0
        p = 0        
        while True:
            l = batch_plates_len[i]
            if p + l <= target_plate_idx:
                # next frame
                i += 1
                p += l
            else:
                return i, target_plate_idx-p