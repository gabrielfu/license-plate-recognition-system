import torch
import cv2
import yaml

from .modules.charnet import Charnet
from utils.image_preprocess import to_tensor
from utils.utils import get_correct_path

class CharRecognizer():
    def __init__(self, cfg):
        weights_path = get_correct_path(cfg['weights_path'])
        self.img_size = cfg['img_size']
        # self.conf_thres = cfg['conf_thres']
        # exec(f"from {cfg['inverse_char_dict_path']} import {cfg['inverse_char_dict']}")
        # print(f"from {cfg['inverse_char_dict_path']} import {cfg['inverse_char_dict']}")
        self.inverse_char_dict = yaml.load(open(cfg['inverse_char_dict'], 'r'), Loader=yaml.FullLoader)['inverse_char_dict']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Charnet((3, self.img_size, self.img_size), len(self.inverse_char_dict))
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, img, input_size, mode):
        '''
        Resize & normalize images for model input
        '''
        if mode == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float64')
        img /=255.
        return img

    def predict(self, img_lst, mode='RGB'):
        '''
        Inputs
            img_lst: list of np.arrays(h,w,c)
        Outputs
            tuple('AB1234', 0.99)
            confidence is the minimal prediction confidence among characters
        '''
        with torch.no_grad():
            output_str = ''
            img_lst = [self.preprocess(img, (self.img_size, self.img_size), mode) for img in img_lst]
            img_lst = [to_tensor(img) for img in img_lst]

            img_tensor = torch.stack(img_lst)
            outs = self.model(img_tensor)
            out_probs = torch.nn.Softmax(dim=1)(outs)
            out_idxs = [out.argmax().item() for out in out_probs]
            char_probs = [out.max().item() for out in out_probs]
            # avg_char_probs = sum(char_probs)/len(char_probs)
            min_char_probs = min(char_probs)
            for _, idx in enumerate(out_idxs):
                # if out_probs[i][idx].numpy() < self.conf_thres:
                #     output_str += '_'
                # else:
                #     output_str += self.inverse_char_dict[idx]
                output_str += self.inverse_char_dict[idx]

        return output_str, min_char_probs
