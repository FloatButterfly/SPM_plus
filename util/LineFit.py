import os

import h5py

cur_dir = os.path.split(os.path.abspath(__file__))[0]


class LineFit(object):
    def __init__(self):
        super().__init__()
        self.data = dict()
        with h5py.File(os.path.join(cur_dir, "Results.h5"), 'r') as f:
            for item in f.keys():
                self.data[item] = f[item][:]

    @staticmethod
    def find_interval(value, array):
        low_value = array[0]
        for i in range(len(array)):
            if array[i] > value:
                return i - 1

        return len(array) - 2

    def __call__(self, value, key):
        rates = sorted(self.data[key][0, :])
        psnrs = sorted(self.data[key][1, :])
        low_index = self.find_interval(value, rates)

        low_rate = rates[low_index]
        up_rate = rates[low_index + 1]
        low_psnr = psnrs[low_index]
        up_psnr = psnrs[low_index + 1]

        interpolate = (up_psnr - low_psnr) / (up_rate - low_rate) * (value - low_rate) + low_psnr

        return interpolate


if __name__ == "__main__":
    line_fit = LineFit()
    print("JPEG    : {:>5.2f} dB;".format(line_fit(2, "JPEG")))
    print("OPENJPEG: {:>5.2f} dB;".format(line_fit(2, "OPENJPEG")))
    print("JPEG2K  : {:>5.2f} dB;".format(line_fit(2, "JPEG2K")))
    print("HM420   : {:>5.2f} dB;".format(line_fit(2, "HM420")))
    print("HM444   : {:>5.2f} dB;".format(line_fit(2, "HM444")))
