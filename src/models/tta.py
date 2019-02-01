import torch.nn.functional as F

class SegmentatorTTA(object):
    @staticmethod
    def hflip(x):
        return x.flip(3)

    @staticmethod
    def vflip(x):
        return x.flip(2)

    @staticmethod
    def trans(x):
        return x.transpose(2, 3)

    def pred_resize(self, x, size, net_type='unet'):
        h, w = size
        if net_type == 'unet':
            pred = self.forward(x)
            if x.shape[2:] == size:
                return pred
            else:
                return F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        else:
            pred = self.forward(F.pad(x, (0, 1, 0, 1)))
            return F.interpolate(pred, size=(h+1, w+1), mode='bilinear', align_corners=True)[..., :h, :w]

    def tta(self, x, scales=None, net_type='unet'):
        size = x.shape[2:]
        if scales is None:
            seg_sum = self.pred_resize(x, size, net_type)
            seg_sum += self.hflip(self.pred_resize(self.hflip(x), size, net_type))
            return seg_sum / 2
        else:
            # scale = 1
            seg_sum = self.pred_resize(x, size, net_type)
            seg_sum += self.hflip(self.pred_resize(self.hflip(x), size, net_type))
            for scale in scales:
                scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
                seg_sum += self.pred_resize(scaled, size, net_type)
                seg_sum += self.hflip(self.pred_resize(self.hflip(scaled), size, net_type))
            return seg_sum / ((len(scales) + 1) * 2)

