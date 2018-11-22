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

    def tta(self, x, full=False):
        if full:
            seg_sum = self.forward(x)
            for i in range(7):
                code = f'{i:03b}'
                xc = x.clone()
                if code[-1] == '0':
                    xc = self.hflip(xc)
                if code[-2] == '0':
                    xc = self.vflip(xc)
                if code[-3] == '0':
                    xc = self.trans(xc)
                seg = self.forward(xc)
                if code[-3] == '0':
                    seg = self.trans(seg)
                if code[-2] == '0':
                    seg = self.vflip(seg)
                if code[-1] == '0':
                    seg = self.hflip(seg)
                seg_sum += seg
            return seg_sum / 8
        else:
            seg_sum = self.forward(x)
            seg_sum += self.hflip(self.forward(self.hflip(x)))
            return seg_sum / 2