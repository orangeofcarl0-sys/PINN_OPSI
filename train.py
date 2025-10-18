import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import h5py
import numpy as np

# 从我们之前的文件中导入模型构建函数和物理仿真函数
from model import build_pinn_cnn_lstm_model
from data_factory import generate_opsi_spectrum # 我们将用它来构建物理重构函数

@tf.function
def reconstruct_spectra_tf(
    params: tf.Tensor,
    freq_axis_thz: tf.Tensor
) -> tuple:
    """
    使用TensorFlow函数，根据预测的物理参数重构OPSI光谱。
    这个函数是可微分的，是PINN的核心。

    参数:
        params (tf.Tensor): 模型的输出张量，形状为 (batch_size, 5)。
                            列顺序: [τ, φ, A_env, μ_env, σ_env]
        freq_axis_thz (tf.Tensor): 频率轴，单位 THz。

    返回:
        tuple: (Ip_recon, Is_recon)
    """
    # 确保频率轴是正确的形状以便于广播
    # (seq_length,) -> (1, seq_length)
    freq_axis_thz = tf.expand_dims(freq_axis_thz, axis=0)
    
    # 从参数张量中解包物理量
    # (batch_size, 5) -> 5 x (batch_size, 1)
    tau_ps = tf.expand_dims(params[:, 0], axis=1)
    phi_rad = tf.expand_dims(params[:, 1], axis=1)
    A_env = tf.expand_dims(params[:, 2], axis=1)
    mu_thz = tf.expand_dims(params[:, 3], axis=1)
    sigma_thz = tf.expand_dims(params[:, 4], axis=1)

    # 将频率单位转换为 rad/ps
    omega_rad_ps = 2 * np.pi * freq_axis_thz
    omega_c_rad_ps = 2 * np.pi * 194.0 # 假设中心频率固定

    # 计算光谱包络 (I_env)
    mu_rad_ps = 2 * np.pi * mu_thz
    sigma_rad_ps = 2 * np.pi * sigma_thz
    
    # 使用 tf.math 中的函数
    I_env = A_env * tf.math.exp(-tf.math.square(omega_rad_ps - mu_rad_ps) / (2 * tf.math.square(sigma_rad_ps)))

    # 计算干涉项
    phase_term = (omega_rad_ps - omega_c_rad_ps) * tau_ps + phi_rad
    cos_term = tf.math.cos(phase_term)

    # 重构 Ip 和 Is
    Ip_recon = I_env * (1.0 + cos_term)
    Is_recon = I_env * (1.0 - cos_term)

    return Ip_recon, Is_recon

class PINN_Model(Model):
    def __init__(self, core_model, lambda_phys=0.5, **kwargs):
        """
        一个包含物理知情损失的自定义Keras模型。

        参数:
            core_model (Model): 我们之前构建的CNN-LSTM模型。
            lambda_phys (float): 物理损失的权重 λ。
        """
        super().__init__(**kwargs)
        self.core_model = core_model
        self.lambda_phys = lambda_phys
        
        # 创建频率轴 (一次性创建，作为常量)
        self.freq_axis_thz = tf.constant(np.linspace(190, 198, 2048), dtype=tf.float32)
        
        # 定义损失跟踪器
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.data_loss_tracker = tf.keras.metrics.Mean(name="data_loss")
        self.phys_loss_tracker = tf.keras.metrics.Mean(name="phys_loss")

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        # 我们不再需要 'loss' 参数，因为损失计算在 train_step 中
    
    @property
    def metrics(self):
        # 定义我们希望在训练过程中监控的指标
        return [self.loss_tracker, self.data_loss_tracker, self.phys_loss_tracker]

    def train_step(self, data):
        x, y_true = data
        # x: 输入光谱 (batch, seq_length, 2)
        # y_true: 真实标签 (batch, 2) -> τ 和 φ

        with tf.GradientTape() as tape:
            # 1. 前向传播，获取预测的物理参数
            y_pred = self.core_model(x, training=True) # (batch, 5)

            # 2. 计算数据损失 (L_data) - 有监督
            # 我们只用 τ 和 φ 计算数据损失
            data_loss = tf.keras.losses.mean_squared_error(y_true, y_pred[:, :2])

            # 3. 物理重构
            Ip_input = x[:, :, 0]
            Is_input = x[:, :, 1]
            Ip_recon, Is_recon = reconstruct_spectra_tf(y_pred, self.freq_axis_thz)

            # 4. 计算物理损失 (L_phys) - 无监督
            phys_loss_p = tf.keras.losses.mean_squared_error(Ip_input, Ip_recon)
            phys_loss_s = tf.keras.losses.mean_squared_error(Is_input, Is_recon)
            phys_loss = phys_loss_p + phys_loss_s

            # 5. 计算总损失
            total_loss = (1.0 - self.lambda_phys) * data_loss + self.lambda_phys * phys_loss
        
        # 6. 反向传播
        grads = tape.gradient(total_loss, self.core_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core_model.trainable_variables))

        # 7. 更新并返回监控指标
        self.loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(data_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        
        return {m.name: m.result() for m in self.metrics}

def load_data(filepath):
    """从HDF5文件加载数据。"""
    with h5py.File(filepath, 'r') as f:
        ip = f['spectra_ip'][:]
        is_spec = f['spectra_is'][:]
        tau = f['labels_tau'][:]
        phi = f['labels_phi'][:]
    
    # 将 Ip 和 Is 堆叠成双通道输入
    x = np.stack([ip, is_spec], axis=-1)
    # 将 τ 和 φ 堆叠成标签
    y = np.stack([tau, phi], axis=-1)
    
    return x, y

if __name__ == '__main__':
    # --- 1. 参数定义 ---
    SEQUENCE_LENGTH = 2048
    NUM_PHYSICAL_PARAMS = 5
    LAMBDA_PHYS = 0.5 # 物理损失权重，这是一个可以调整的关键超参数
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 5 # 先用少量周期进行测试

    # --- 2. 加载数据 ---
    print("正在加载训练数据...")
    x_train, y_train = load_data('opsi_dataset_train.h5')
    print(f"训练数据加载完成。输入形状: {x_train.shape}, 标签形状: {y_train.shape}")

    # --- 3. 构建并编译模型 ---
    # 构建核心的CNN-LSTM模型
    core_model = build_pinn_cnn_lstm_model(
        seq_length=SEQUENCE_LENGTH,
        num_outputs=NUM_PHYSICAL_PARAMS
    )
    core_model.summary()

    # 将核心模型包装进我们的PINN模型中
    pinn_model = PINN_Model(core_model=core_model, lambda_phys=LAMBDA_PHYS)

    # 编译PINN模型
    pinn_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))

    # --- 4. 开始训练 ---
    print("\n开始训练PINN模型...")
    history = pinn_model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
        # 可以在这里添加验证集: validation_data=(x_val, y_val)
    )
    print("训练完成！")