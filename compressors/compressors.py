import torch

class Compression:
    def compress(tensor):
        pass


class NoneCompressor(Compression):
    def compress(tensor):
        return tensor


class TopKCompressor(Compression):
    def __init__(alpha=None, k=None):
        if alpha:
            assert alpha > 0, 'Number of transmitted coordinates must be positive'
            self.alpha = alpha
        else:
            assert k > 0, 'Number of transmitted coordinates must be positive'
            self.k = k

    def getK(tensor):
        if self.k:
            return self.k
        result = int(tensor.numel() * alpha)
        assert result > 0, 'Number of transmitted coordinates must be positive'
        return result
 

    def compress(tensor):
        absTensor = torch.abs(tensor)
        k = self.getK(tensor)
        mask = torch.zeros_like(tensor).index_fill_(0, absTensor.topK(k), 1)
        return tensor * mask
