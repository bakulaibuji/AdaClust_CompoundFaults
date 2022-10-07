import numpy as np
import matplotlib.pyplot as plt


def FFT_vectorized(x, sampling_frequency):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    freqs = np.linspace(0, sampling_frequency // 2, sampling_frequency // 2 + 1)  # 表示频率

    return np.array([freqs, X.ravel()])


# 输入时域信号和采样频率，返回频谱图中的频率和幅值
def fft(signal: np.array, sampling_frequency: int):
    xf = np.fft.rfft(signal) / sampling_frequency  # 返回sampling_frequency/2+1 个频率
    freqs = np.linspace(0, sampling_frequency // 2, len(signal) // 2 + 1)  # 表示频率
    ampli = np.abs(xf) * 2  # 代表信号的幅值, 即振幅
    return np.array([freqs, ampli])


def test_fft():
    sampling_rate = 5120
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    x = np.sin(2 * np.pi * 156.25 * t) + 2 * np.sin(2 * np.pi * 234.375 * t) + 3 * np.sin(2 * np.pi * 200 * t)
    freqs, ampli = FFT_vectorized(x, sampling_rate)

    plt.figure(num='original', figsize=(15, 6))
    plt.plot(x[:100])

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, x)
    plt.xlabel(u"时间(秒)", fontproperties='FangSong')
    plt.title(u"156.25Hz和234.375Hz的波形和频谱", fontproperties='FangSong')

    plt.subplot(212)
    plt.plot(freqs, ampli)
    plt.xlabel(u"频率(Hz)", fontproperties='FangSong')
    plt.ylabel(u'幅值', fontproperties='FangSong')
    plt.subplots_adjust(hspace=0.4)
    plt.show()


if __name__ == '__main__':
    test_fft()
