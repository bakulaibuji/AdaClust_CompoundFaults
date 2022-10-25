import h5py
import sys
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp2d
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    data_segment = 5120 * 2
    sampling_frequency = 5120
    window = 'tukey'
    nperseg = 255
    noverlap = 170
    pic_size = 56 * 2
    x_len = pic_size * 5
    f_len = pic_size * 2

    path_orig = Path('/home/huangteam/wangyuxiang/data/Compound Faults')
    if not path_orig.exists():
        raise Exception('original path does not exist!')

    path_out = Path('/home/huangteam/wangyuxiang/data/Compound Faults/Images2')
    if not path_out.exists():
        path_out.mkdir()

    files = [f for f in path_orig.glob('*') if not f.is_dir()]
    fault_types = ['正常', '裂纹', '断齿', '缺齿', '磨损', '点蚀', '磨损+点蚀', '裂纹+缺齿', '裂纹+断齿']
    # 正确的应该是 fault_types = ['正常', '裂纹', '断齿', '缺齿', '磨损', '点蚀', '裂纹+磨损', '裂纹+缺齿', '断齿+点蚀']

    split_num = 3

    input = sys.argv
    for file in files:
        if "B4" not in file.name:
            continue

        name = file.name.split(".")[0]
        for i in range(split_num):
            temp_name = name + str(i)
            folder = path_out / temp_name
            if not folder.exists():
                folder.mkdir()
            for index, fault_type in enumerate(fault_types):
                fault_type_folder = folder / fault_type
                if not fault_type_folder.exists():
                    fault_type_folder.mkdir()

        read_file = h5py.File(file, 'r')
        signals = pd.DataFrame(np.array(read_file['x']))
        read_file.close()

        for index, fault_type in enumerate(fault_types):

            for beg in tqdm(range(0, signals.shape[1] - data_segment, data_segment)):

                for k in range(2):
                    sig = signals.iloc[index * 2 + k][beg: beg + data_segment]

                    if max(sig) < 0.01:
                        continue

                    f, t, Zxx = signal.stft(sig, sampling_frequency, window=window, nperseg=nperseg, noverlap=noverlap)
                    newF = interp2d(t, f, np.abs(Zxx), kind='linear')
                    t_new = np.linspace(0, 1 / sampling_frequency * data_segment, num=x_len)
                    f_new = np.linspace(0, sampling_frequency // 2, num=f_len)
                    Z_new = newF(t_new, f_new)
                    Xn, Yn = np.meshgrid(t_new, f_new)

                    plt.pcolormesh(Xn, Yn, Z_new, shading='auto')
                    plt.axis('off')
                    num = int(beg / (signals.shape[1] / split_num))
                    fault_type_folder = path_out / (name + str(num)) / fault_type
                    plt.savefig(fault_type_folder / (str(beg) + '_' + str(k) + '.png'), dpi=300, bbox_inches='tight',
                                pad_inches=0)
                    plt.clf()
                    plt.close()
