# numpy implementation of Convolution Operation in matrix (forward)
# todo: add dilation and bias
import numpy as np
import torch
import time
from torch.nn import Conv2d, Parameter


class Conv2dBasic:
    def __init__(self, in_fea_size,
                 out_fea_size,
                 kernel_size,
                 padding,
                 stride):
        self.in_fea_size = in_fea_size
        self.out_fea_size = out_fea_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.h_after = -1
        self.w_after = -1
        # creat convolutional blocks
        self.W = np.random.randn(self.out_fea_size, self.in_fea_size, self.kernel_size, self.kernel_size)

    def _im2col(self, x):
        """
        turn image into patch matrix
        :param x: input with data with shape [C, H, W]
        :return: patch matrix with shape [B, number of patches, k]
        """
        b, c, h, w = x.shape
        # padding
        # todo: pad_width understanding?
        x_padded = np.pad(x, ((0,), (0,), (self.padding,), (self.padding,)), mode="constant", constant_values=0)

        # h, w after convolution
        self.h_after, self.w_after = self._after_conv_size(h), self._after_conv_size(w)
        # patches size
        patches_size = self.h_after * self.w_after
        # total length of one patch
        k = self.kernel_size * self.kernel_size * self.in_fea_size
        col = np.zeros([b, patches_size, k], dtype=np.float64)

        for i in range(self.h_after):
            for j in range(self.w_after):
                col[:, i * self.w_after + j, :] = x_padded[:, :,
                                                  self.stride * i: self.stride * i + self.kernel_size,
                                                  self.stride * j: self.stride * j + self.kernel_size].reshape(b, -1)

        return col

    def _after_conv_size(self, x):
        return (x + 2 * self.padding - self.kernel_size) // self.stride + 1

    def __call__(self, x):
        """
        :param x: input data with shape (B, C, H, W)
        :return: result after convolution
        """

        return -1


class Conv3x3(Conv2dBasic):
    def __init__(self, in_fea_size, out_fea_size, padding=1, stride=1):
        super(Conv3x3, self).__init__(in_fea_size=in_fea_size,
                                     out_fea_size=out_fea_size,
                                     kernel_size=3,
                                     padding=padding,
                                     stride=stride)

    def __call__(self, x):
        col = self._im2col(x)
        self.W = self.W.reshape([self.out_fea_size, self.kernel_size * self.kernel_size * self.in_fea_size]).T
        # col: [B, num of patches, k] * W: [1, k, out feature size] = out: [B, num of patches, 1, out feature size]
        out = col.dot(self.W[np.newaxis, :, :])
        # out: [B, out feature size, 1, num of patches]
        out = np.swapaxes(out, 1, 3)
        # out: [B, out feature size, num of patches]
        out = np.squeeze(out, 2)
        # out: [B, out feature size, h, w]
        out = np.reshape(out, [out.shape[0], out.shape[1], self.h_after, self.w_after])

        return out


if __name__ == '__main__':
    cnn = Conv3x3(in_fea_size=64, out_fea_size=128, padding=1, stride=1)
    cnn_torch = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False)
    # apply cnn weight to cnn_torch weight
    cnn_torch.weight = Parameter(torch.from_numpy(cnn.W))
    data = np.random.randn(64, 64, 64, 64)

    start = time.time()
    out_torch = cnn_torch(torch.from_numpy(data))
    end = time.time()
    print(f"torch implementation time: {end - start}")

    start = time.time()
    out_self_imp = cnn(data)
    end = time.time()
    print(f"self implementation time: {end - start}")

    print(f"diff: {np.sum(out_torch.detach().numpy() - out_self_imp)}")
