import tensorflow as tf
import numpy as np
import h5py
import joblib
import matplotlib.pyplot as plt

# 从你的代码中导入必要的函数
# 注意：我们需要一个numpy版本的重构函数，因为它在TF图之外运行更方便
from data_factory import generate_opsi_spectrum 

# 为了方便，我们直接把TF重构函数的核心逻辑用numpy重写一遍
def reconstruct_spectra_np(params, freq_axis_thz):
    tau_ps, phi_rad, A_env, mu_thz, sigma_thz = params
    
    omega_rad_ps = 2 * np.pi * freq_axis_thz
    omega_c_rad_ps = 2 * np.pi * 194.0

    mu_rad_ps = 2 * np.pi * mu_thz
    sigma_rad_ps = 2 * np.pi * sigma_thz
    
    I_env = A_env * np.exp(-np.square(omega_rad_ps - mu_rad_ps) / (2 * np.square(sigma_rad_ps)))
    
    phase_term = (omega_rad_ps - omega_c_rad_ps) * tau_ps + phi_rad
    cos_term = np.cos(phase_term)

    Ip_recon = I_env * (1.0 + cos_term)
    Is_recon = I_env * (1.0 - cos_term)

    return Ip_recon, Is_recon

def main():
    # --- 1. 加载所有必要的“证物” ---
    print("正在加载模型、数据和缩放器...")
    
    # 加载你之前训练好的、表现不佳的模型
    # Keras需要知道自定义类才能加载
    from train import PINN_Model 
    model = tf.keras.models.load_model('best_pinn_model.keras', custom_objects={'PINN_Model': PINN_Model})
    
    # 加载用于反归一化的缩放器
    tau_scaler = joblib.load('tau_scaler.gz')
    
    # 加载测试数据
    with h5py.File('opsi_dataset_test.h5', 'r') as f:
        x_test = f['spectra_ip'][:], f['spectra_is'][:]
        y_test = f['labels_tau'][:], f['labels_phi'][:]

    # --- 2. 随机挑选一个“案发现场” (一个测试样本) ---
    sample_index = np.random.randint(0, len(x_test[0]))
    
    Ip_input = x_test[0][sample_index]
    Is_input = x_test[1][sample_index]
    tau_true = y_test[0][sample_index]
    phi_true = y_test[1][sample_index]
    
    # 准备模型的输入格式 (batch_size, seq_length, 2)
    model_input = np.stack([Ip_input, Is_input], axis=-1)
    model_input = np.expand_dims(model_input, axis=0) # 增加一个batch维度

    # --- 3. 让模型进行一次“犯罪陈述” (进行预测) ---
    predicted_params_raw = model.predict(model_input)[0] # 预测结果形状为(1, 5)，取第一行

    # --- 4. 解读“陈述” (反归一化并整理参数) ---
    tau_pred_scaled = predicted_params_raw[0]
    
    # 关键一步：将预测的τ从[0,1]范围反归一化回物理单位(ps)
    tau_pred_physical = tau_scaler.inverse_transform([[tau_pred_scaled]])[0, 0]
    
    predicted_params_physical = [
        tau_pred_physical,          # 使用反归一化后的τ
        predicted_params_raw[1],    # φ
        predicted_params_raw[2],    # A_env
        predicted_params_raw[3],    # μ_env
        predicted_params_raw[4]     # σ_env
    ]

    print("\n--- 诊断结果 ---")
    print(f"真实 τ: {tau_true:.4f} ps")
    print(f"模型预测 τ: {tau_pred_physical:.4f} ps  <-- 看到这里的巨大差异了吗？")

    # --- 5. 根据模型的“陈述”进行“现场还原” (重构光谱) ---
    freq_axis = np.linspace(190, 198, len(Ip_input))
    Ip_recon, Is_recon = reconstruct_spectra_np(predicted_params_physical, freq_axis)

    # --- 6. 呈现证据！ (可视化对比) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # 证据一：时域直接对比
    axes[0].plot(freq_axis, Ip_input, 'b-', label='真实光谱 (Input Ip)', alpha=0.7)
    axes[0].plot(freq_axis, Ip_recon, 'r--', label=f'重构光谱 (Recon Ip)\nτ_pred={tau_pred_physical:.2f}ps', linewidth=2)
    axes[0].set_title(f'证据一：时域对比 (真实τ={tau_true:.2f}ps)')
    axes[0].set_xlabel('Frequency (THz)')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()
    axes[0].grid(True)
    
    # 证据二：残差分析
    residual = Ip_input - Ip_recon
    axes[1].plot(freq_axis, residual)
    axes[1].set_title('证据二：残差 (真实光谱 - 重构光谱)')
    axes[1].set_xlabel('Frequency (THz)')
    axes[1].set_ylabel('Error')
    axes[1].grid(True)
    
    # 证据三：频域FFT对比
    # 我们只关心交流分量，所以先减去均值
    fft_input = np.abs(np.fft.fft(Ip_input - np.mean(Ip_input)))
    fft_recon = np.abs(np.fft.fft(Ip_recon - np.mean(Ip_recon)))
    # 频率轴
    fft_freq = np.fft.fftfreq(len(Ip_input), d=(freq_axis[1]-freq_axis[0]))
    
    # 我们只看正频率部分
    half_point = len(fft_freq) // 2
    axes[2].plot(fft_freq[:half_point], fft_input[:half_point], 'b-', label='真实光谱 FFT')
    axes[2].plot(fft_freq[:half_point], fft_recon[:half_point], 'r--', label='重构光谱 FFT', linewidth=2)
    axes[2].set_title('证据三：频域FFT对比')
    axes[2].set_xlabel('Fringe Frequency (1/THz = ps)')
    axes[2].set_ylabel('Amplitude')
    # 限制x轴范围，看得更清楚
    axes[2].set_xlim(0, max(tau_true, tau_pred_physical) * 1.5) 
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('diagnosis_report.png')
    plt.show()

if __name__ == '__main__':
    main()