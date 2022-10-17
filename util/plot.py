import h5py
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def prepare(data):
    index = (data[0, :] > 0.1) & (data[0, :] < 1.0)

    return data[0, index], data[1, index]


def smooth_plot(x, y, *args, **kwargs):
    x, y = np.array(x), np.array(y)
    x_ = np.linspace(x.min(), x.max(), 100)
    f = interpolate.interp1d(x, y, kind="quadratic")
    y_ = f(x_)

    index = (x_ < 1) & (x_ > 0.1)
    x_ = x_[index]
    y_ = y_[index]

    plt.plot(x_, y_, *args, **kwargs)


def plot():
    x = 0.4742
    y = 33.91

    with h5py.File("Results.h5", 'r') as f:
        jpeg = f["JPEG"][:]
        jpeg2k = f["JPEG2K"][:]
        openjpeg = f["OPENJPEG"][:]
        hm444 = f["HM444"][:]
        hm420 = f["HM420"][:]
        vtm420 = f["VTM420"][:]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # smooth_plot(x, y, 'r-', label="Proposed")
    plt.plot(x, y, 'rx')

    smooth_plot(*prepare(vtm420), label="VTM-6.2 (YUV420)")

    smooth_plot(*prepare(hm444), label="HM-16.15 (YUV444)")
    smooth_plot(*prepare(hm420), label="HM-16.15 (YUV420)")

    # smooth_plot(*prepare(jpeg2k), label="JPEG2K (MATLAB)")
    # smooth_plot(*prepare(openjpeg), label="JPEG2K (OpenJPEG)")
    
    plt.title("RD Performance", fontsize=14)
    plt.xlabel("Rate (bit-per-pixel)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    # for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     item.set_fontsize(11)
    plt.grid(True)
    plt.legend(loc="lower right")
    
    return fig    


if __name__ == "__main__":
    fig = plot()
    plt.show()
