# numpy implementation of Convolution Operation in matrix (forward)
# todo: add dilation and bias
import numpy as np
import torch
from torch.nn import Conv2d, Parameter

from base_class import Base

np.random.seed(123)
torch.random.manual_seed(123)


class Conv2dSelf(Base):
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
        # store image information
        self.xtm = None
        self.h_after = -1
        self.w_after = -1
        # creat convolutional blocks
        self.W = np.random.randn(self.out_fea_size, self.in_fea_size, self.kernel_size, self.kernel_size)

        # derivative, k = in_fea * kernel size * kernel size
        # shape [B, k, num of patches]
        self.cacheX = None
        # shape [out feature size, k]
        self.cacheW = None
        # shape like self.cacheW
        self.dw = None

    def _im2col(self, x):
        """
        turn image into patch matrix
        :param x: input with data with shape [B, C, H, W]
        :return: patch matrix with shape [B, k, num of patches]
        """
        b, c, h, w = x.shape

        # padding
        # todo: pad_width understanding?
        x_padded = np.pad(x, ((0,), (0,), (self.padding,), (self.padding,)), mode="constant", constant_values=0)

        # h, w after convolution
        self.h_after, self.w_after = self._after_conv_size(h), self._after_conv_size(w)

        self.xtm = x_padded.shape
        # patches size
        patches_size = self.h_after * self.w_after
        # total length of one patch
        k = self.kernel_size * self.kernel_size * self.in_fea_size
        col = np.zeros([b, k, patches_size], dtype=np.float64)

        for i in range(self.h_after):
            for j in range(self.w_after):
                col[:, :, i * self.w_after + j] = x_padded[:, :,
                                                  self.stride * i: self.stride * i + self.kernel_size,
                                                  self.stride * j: self.stride * j + self.kernel_size].reshape([b, -1])

        self._col2im(col)
        return col

    def _col2im(self, x):
        """
         turn patch matrix back to image
         :param x: patch matrix with shape [B, k, num of patches]
         :return: input with data with shape [B, C, H, W]
         """
        image = np.zeros(self.xtm)
        weight = np.zeros(self.xtm)
        for i in range(self.h_after):
            for j in range(self.w_after):
                image[:, :, self.stride * i: self.stride * i + self.kernel_size,
                self.stride * j: self.stride * j + self.kernel_size] += x[:, :, i * self.w_after + j].reshape(
                    [self.xtm[0], self.xtm[1], self.kernel_size, self.kernel_size])
                weight[:, :, self.stride * i: self.stride * i + self.kernel_size,
                self.stride * j: self.stride * j + self.kernel_size] += np.ones(
                    [self.xtm[0], self.xtm[1], self.kernel_size, self.kernel_size])

        # weight[weight == 0] = 1
        # image /= weight

        return image[:, :, self.padding: -self.padding, self.padding: -self.padding]

    def _after_conv_size(self, x):
        return (x + 2 * self.padding - self.kernel_size) // self.stride + 1

    def forward(self, x):
        """
        :param x: input data with shape (B, C, H, W)
        :return: result after convolution
        """

        # shape [B, k, num of patches]
        x = x.copy()
        col = self._im2col(x)
        # store x
        self.cacheX = col.copy()
        # shape [out feature size, k]
        self.cacheW = self.W.reshape([self.out_fea_size, self.kernel_size * self.kernel_size * self.in_fea_size]).copy()

        # W: [out feature size, k] * col: [B, k, num of patches] = out: [B, out feature size, num of patches]
        out = np.matmul(self.W.reshape([self.out_fea_size, self.kernel_size * self.kernel_size * self.in_fea_size]),
                        col)

        # out: [B, out feature size, h, w]
        out = np.reshape(out, [out.shape[0], out.shape[1], self.h_after, self.w_after])

        return out

    def backward(self, dout):
        """
        :param dout: derivative from upper layer in shape [B, C, H, W]
        :return: derivative of current layer
        """
        # shape [B, C, num of patches]
        dout = dout.copy()
        dout = np.reshape(dout, (dout.shape[0], dout.shape[1], -1))
        # dw = det * cacheX.T
        self.dw = np.matmul(dout, np.swapaxes(self.cacheX, 1, 2))
        self.dw = np.sum(self.dw, axis=0) / dout.shape[0]
        self.dw = self.dw.reshape([self.out_fea_size, self.in_fea_size, self.kernel_size, self.kernel_size])
        # dx = cacheW.T * det
        dx = np.matmul(self.cacheW.T, dout)
        dx = self._col2im(dx)

        return dx


def test00():
    image = np.random.randn(2, 3, 32, 32)
    conv_01 = Conv2dSelf(3, 32, 3, padding=1, stride=2)
    conv_02 = Conv2dSelf(32, 16, 3, padding=1, stride=2)

    # forward
    out_01 = conv_01.forward(image)
    out_02 = conv_02.forward(out_01)

    conv_01_torch = Conv2d(3, 32, 3, padding=1, stride=2, bias=False)
    conv_02_torch = Conv2d(32, 16, 3, padding=1, stride=2, bias=False)

    # forward
    conv_01_torch.weight = Parameter(torch.from_numpy(conv_01.W))
    out_01_torch = conv_01_torch(torch.from_numpy(image))
    conv_02_torch.weight = Parameter(torch.from_numpy(conv_02.W))
    out_02_torch = conv_02_torch(out_01_torch)

    print(f"diff of forward conv01: {np.sum(out_01_torch.detach().numpy() - out_01)}")
    print(f"diff of forward conv02: {np.sum(out_01_torch.detach().numpy() - out_01)}")

    # backward
    mean = torch.mean(out_02_torch)
    mean.backward()
    dw_conv_01_torch = conv_01_torch.weight.grad.data.numpy()
    dw_conv_02_torch = conv_02_torch.weight.grad.data.numpy()

    n = out_02.shape[0] * out_02.shape[1] * out_02.shape[2] * out_02.shape[3]
    det = np.ones_like(out_02) * mean.detach().numpy() / n
    dx_02 = conv_02.backward(det)
    dx_01 = conv_01.backward(dx_02)

    dw_conv_01 = conv_01.dw
    dw_conv_02 = conv_02.dw

    # todo: the backward pass results is not the same with pytorch
    print(f"diff of dw in conv_01: {np.sum(dw_conv_01 - dw_conv_01_torch)}")
    print(f"diff of dw in conv_02: {np.sum(dw_conv_02 - dw_conv_02_torch)}")


if __name__ == '__main__':
    test00()
    # in_cha = 1
    # out_cha = 3
    # ker_size = 3
    # padding = 1
    # stride = 1
    # cnn = Conv2dSelf(in_fea_size=in_cha, out_fea_size=out_cha, kernel_size=ker_size, padding=padding, stride=stride)
    # cnn_torch = Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=ker_size, padding=padding, stride=stride,
    #                    bias=False)
    # # apply cnn weight to cnn_torch weight
    # cnn_torch.weight = Parameter(torch.from_numpy(cnn.W))
    # data = np.random.randn(1, in_cha, 3, 3)
    #
    # # forward
    # start = time.time()
    # out_torch = cnn_torch(torch.from_numpy(data))
    # end = time.time()
    # print(f"torch implementation time: {end - start}")
    # #
    # start = time.time()
    # out_self_imp = cnn(data)
    # end = time.time()
    # print(f"self implementation time: {end - start}")
    # #
    # print(f"diff: {np.sum(out_torch.detach().numpy() - out_self_imp)}")
    #
    # # backward
    # det = np.ones_like(out_self_imp)
    # cnn.backward(det)
