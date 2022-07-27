import torch
import yaml
import numpy as np

from ..utils.utils import get_correct_path
from .modules.charnet_trt import TrtCharNet


class CharRecognizerTRT:
    def __init__(self, cfg):
        self.model_path = get_correct_path(cfg['model_path'])
        self.input_size = cfg['input_size']
        self.max_batch_size = cfg['max_batch_size']
        self.inverse_char_dict = yaml.load(open(cfg['inverse_char_dict'], 'r'), Loader=yaml.FullLoader)['inverse_char_dict']
        self.model = TrtCharNet(self.model_path, self.input_size, self.max_batch_size)

    def predict(self, img_lst):
        """
        Inputs
            img_lst: list of np.arrays(h,w,c)
        Outputs
            tuple('AB1234', 0.99)
            confidence is the minimal prediction confidence among characters
        """
        output_str = ''

        out = self.model.detect(img_lst)[0][:len(img_lst), :]  ## trt_outputs[0] is the prediction
        out_probs = np.exp(out)
        out_idxs = np.argmax(out_probs, axis=-1)
        char_probs = np.max(out_probs, axis=-1)
#         out_probs = torch.nn.Softmax(dim=1)(outs)
#         out_idxs = [out.argmax().item() for out in out_probs]
#         char_probs = [out.max().item() for out in out_probs]
        # avg_char_probs = sum(char_probs)/len(char_probs)
        min_char_probs = min(char_probs)
        for _, idx in enumerate(out_idxs):
            # if out_probs[i][idx].numpy() < self.conf_thres:
            #     output_str += '_'
            # else:
            #     output_str += self.inverse_char_dict[idx]
            output_str += self.inverse_char_dict[idx]

        return output_str, min_char_probs
