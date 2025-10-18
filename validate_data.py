import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_sample(filepath, index=0):
    """
    从HDF5文件中加载并可视化一个样本。
    """
    with h5py.File(filepath, 'r') as f:
        ip = f['spectra_ip'][index]
        is_spec = f['spectra_is'][index]
        tau = f['labels_tau'][index]
        phi = f['labels_phi'][index]
        noise = f['meta_noise_db'][index]

    seq_length = len(ip)
    freq_axis = np.linspace(190, 198, seq_length)

    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, ip, label=f'Ip (p-pol)', alpha=0.8)
    plt.plot(freq_axis, is_spec, label=f'Is (s-pol)', alpha=0.8)
    plt.title(f'Sample #{index} - τ={tau:.2f} ps, φ={phi:.2f} rad, Noise={noise:.1f} dB')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # 随机抽查训练集中的一个样本
    visualize_sample('opsi_dataset_train.h5', index=np.random.randint(0, 100))