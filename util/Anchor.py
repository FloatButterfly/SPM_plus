import os

import h5py
from matplotlib import pyplot as plt


def prepare(data, threshold=1):
    index = data[0, :] < threshold

    return data[0, index], data[1, index]


class Anchor(object):
    def __init__(self, threshold=3.5):
        super().__init__()
        self.data = dict()
        root = os.path.split(os.path.abspath(__file__))[0]
        with h5py.File(os.path.join(root, "Results.h5"), 'r') as f:
            for item in f.keys():
                self.data[item] = prepare(f[item][:], threshold)

    def plot(self, bpp=None, psnr=None):
        fig = plt.figure(1)
        # ax = fig.add_subplot(111)
        # for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        #     item.set_fontsize(11)
        for key in self.data:
            plt.plot(*self.data[key], label=key)
        plt.title("RD Performance", fontsize=14)
        plt.xlabel("BPP", fontsize=12)
        plt.xlabel("BPP", fontsize=12)
        plt.ylabel("PSNR (dB)", fontsize=12)
        if bpp is not None and psnr is not None:
            plt.plot(bpp, psnr, "rx")
        plt.grid(True)
        plt.legend()

        return fig


if __name__ == "__main__":
    anchor = Anchor()
    fig = anchor.plot(2, 36)
    plt.show()
