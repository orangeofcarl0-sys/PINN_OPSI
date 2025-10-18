import numpy as np
import h5py
from tqdm import tqdm

def generate_opsi_spectrum(
    seq_length: int = 2048,
    freq_range_thz: tuple = (190, 198),
    tau_ps: float = 5.0,
    phi_rad: float = np.pi,
    noise_intensity_db: float = -40.0,
    envelope_params: dict = {'A': 1.0, 'mu_thz': 194.0, 'sigma_thz': 1.0},
    center_freq_thz: float = 194.0
) -> tuple:
    """
    生成一对正交偏振光谱干涉(OPSI)光谱数据。

    参数:
        seq_length (int): 光谱的点数。
        freq_range_thz (tuple): 光谱的频率范围 (起始, 结束)，单位 THz。
        tau_ps (float): 时间延迟 τ，单位皮秒 (ps)。
        phi_rad (float): 中心频率相位 φ，单位弧度 (rad)。
        noise_intensity_db (float): 强度噪声水平，单位 dB。
        envelope_params (dict): 光谱包络参数 (高斯模型)。
        center_freq_thz (float): 干涉仪的中心频率 ωc，单位 THz。

    返回:
        tuple: (Ip, Is, labels)
            Ip (np.array): p偏振光谱。
            Is (np.array): s偏振光谱。
            labels (dict): 用于生成该样本的真实物理参数。
    """
    # 1. 创建频率轴 (ω)
    freq_axis_thz = np.linspace(freq_range_thz[0], freq_range_thz[1], seq_length)
    omega_rad_ps = 2 * np.pi * freq_axis_thz  # 转换为角频率，单位 rad/ps
    omega_c_rad_ps = 2 * np.pi * center_freq_thz

    # 2. 生成理想光谱包络 (I_env)
    A = envelope_params['A']
    mu = 2 * np.pi * envelope_params['mu_thz']
    sigma = 2 * np.pi * envelope_params['sigma_thz']
    I_env = A * np.exp(-((omega_rad_ps - mu)**2) / (2 * sigma**2))
    
    # 注意：此处简化了相位噪声模型，实际可根据OPSI论文添加更复杂的模型
    # 简单的相位噪声可以看作是I_env的随机波动，这里我们主要关注强度噪声

    # 3. 计算干涉项
    # 将 τ 单位从 ps 转换为 s, 角频率单位从 rad/ps 转换为 rad/s
    # tau_s = tau_ps * 1e-12
    # omega_rad_s = omega_rad_ps * 1e12
    # omega_c_rad_s = omega_c_rad_ps * 1e12
    # phase_term = (omega_rad_s - omega_c_rad_s) * tau_s + phi_rad
    # 为了避免数值溢出，直接使用 ps 和 rad/ps 单位，它们会相互抵消
    phase_term = (omega_rad_ps - omega_c_rad_ps) * tau_ps + phi_rad
    
    cos_term = np.cos(phase_term)

    # 4. 生成理想的 Ip 和 Is 光谱
    Ip_ideal = I_env * (1 + cos_term)
    Is_ideal = I_env * (1 - cos_term)

    # 5. 添加强度噪声
    # 将dB转换为线性信噪比(SNR)，然后计算噪声标准差
    snr_linear = 10**(abs(noise_intensity_db) / 10)
    signal_power = np.mean(Ip_ideal**2) # 以Ip的平均功率为基准
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    
    intensity_noise_p = np.random.normal(0, noise_std, seq_length)
    intensity_noise_s = np.random.normal(0, noise_std, seq_length)

    Ip_noisy = Ip_ideal + intensity_noise_p
    Is_noisy = Is_ideal + intensity_noise_s
    
    # 6. 准备标签
    labels = {
        'tau_ps': tau_ps,
        'phi_rad': phi_rad,
        'noise_intensity_db': noise_intensity_db
    }

    return Ip_noisy, Is_noisy, labels

def generate_dataset(num_samples: int, filepath: str):
    """
    生成大规模OPSI数据集并保存为HDF5文件。
    """
    print(f"开始生成 {num_samples} 个样本...")
    
    # 用于存储所有数据的列表
    ip_list, is_list, labels_list = [], [], []

    for _ in tqdm(range(num_samples)):
        # 1. 随机采样物理参数 (数据增强的核心)
        tau_sample = np.random.uniform(0.5, 50.0)  # 宽范围的 τ
        phi_sample = np.random.uniform(0, 2 * np.pi) # 完整的相位范围
        noise_db_sample = np.random.uniform(-60, -30) # 变化的噪声水平
        
        # 包络参数的微小扰动
        env_A = 1.0 + np.random.normal(0, 0.05)
        env_mu = 194.0 + np.random.normal(0, 0.1)
        env_sigma = 1.0 + np.random.normal(0, 0.05)
        
        envelope_sample = {'A': env_A, 'mu_thz': env_mu, 'sigma_thz': env_sigma}

        # 2. 调用核心引擎生成单一样本
        ip, is_spec, labels = generate_opsi_spectrum(
            tau_ps=tau_sample,
            phi_rad=phi_sample,
            noise_intensity_db=noise_db_sample,
            envelope_params=envelope_sample
        )
        
        # 3. 收集数据
        ip_list.append(ip)
        is_list.append(is_spec)
        labels_list.append(labels)

    # 4. 保存到HDF5文件
    save_to_hdf5(ip_list, is_list, labels_list, filepath)

def save_to_hdf5(ip_list, is_list, labels_list, filepath):
    """
    将生成的数据高效地保存到HDF5文件。
    """
    print(f"正在将数据保存到 {filepath}...")
    
    # 将标签列表转换为结构化的numpy数组以便于索引
    labels_tau = np.array([item['tau_ps'] for item in labels_list])
    labels_phi = np.array([item['phi_rad'] for item in labels_list])
    meta_noise = np.array([item['noise_intensity_db'] for item in labels_list])

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('spectra_ip', data=np.array(ip_list, dtype=np.float32))
        f.create_dataset('spectra_is', data=np.array(is_list, dtype=np.float32))
        f.create_dataset('labels_tau', data=labels_tau)
        f.create_dataset('labels_phi', data=labels_phi)
        f.create_dataset('meta_noise_db', data=meta_noise)
        
        # 存储元数据，这对于复现至关重要
        f.attrs['num_samples'] = len(ip_list)
        f.attrs['tau_range_ps'] = [0.5, 50.0]
        f.attrs['noise_range_db'] = [-60, -30]
        
    print("数据保存完成！")

if __name__ == '__main__':
    # --- 生成训练集 ---
    generate_dataset(num_samples=80000, filepath='opsi_dataset_train.h5')
    
    # --- 生成验证集 ---
    generate_dataset(num_samples=10000, filepath='opsi_dataset_val.h5')
    
    # --- 生成测试集 ---
    generate_dataset(num_samples=10000, filepath='opsi_dataset_test.h5')