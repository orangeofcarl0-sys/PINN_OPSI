import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import h5py
import numpy as np
from tensorflow.keras import callbacks
import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib

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
        super().__init__(**kwargs)
        self.core_model = core_model
        self.lambda_phys = lambda_phys
        
        self.freq_axis_thz = tf.constant(np.linspace(190, 198, 2048), dtype=tf.float32)
        
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.data_loss_tracker = tf.keras.metrics.Mean(name="data_loss")
        self.phys_loss_tracker = tf.keras.metrics.Mean(name="phys_loss")

    # --- FIX: Implement the get_config() method ---
    def get_config(self):
        """
        重写此方法，以确保模型可以被正确序列化。
        返回一个包含所有构造函数参数的字典。
        """
        # 1. 获取父类的基本配置 (如 name, dtype)
        base_config = super().get_config()
        
        # 2. 添加我们自定义的构造函数参数
        config = {
            # Keras会自动处理嵌套的模型/层
            "core_model": tf.keras.utils.serialize_keras_object(self.core_model),
            "lambda_phys": self.lambda_phys,
        }
        
        # 3. 合并并返回最终配置
        return {**base_config, **config}

    # --- (可选但推荐) 实现 from_config 以获得更强的鲁棒性 ---
    @classmethod
    def from_config(cls, config):
        """
        从配置字典重新创建模型实例。
        """
        # 反序列化嵌套的模型
        core_model_config = config.pop("core_model")
        core_model = tf.keras.layers.deserialize(core_model_config)
        
        # 使用剩余的配置创建实例
        return cls(core_model, **config)
    
    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
    
    def call(self, inputs):
        return self.core_model(inputs)

    @property
    def metrics(self):
        # Keras会自动为验证阶段的指标加上 "val_" 前缀
        return [self.loss_tracker, self.data_loss_tracker, self.phys_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y_true = data[0], data[1]
        else:
            x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.core_model(x, training=True)
            data_loss = self.mse_loss_fn(y_true, y_pred[:, :2])
            
            Ip_input = x[:, :, 0]
            Is_input = x[:, :, 1]
            Ip_recon, Is_recon = reconstruct_spectra_tf(y_pred, self.freq_axis_thz)
            
            phys_loss_p = self.mse_loss_fn(Ip_input, Ip_recon)
            phys_loss_s = self.mse_loss_fn(Is_input, Is_recon)
            phys_loss = phys_loss_p + phys_loss_s

            total_loss = (1.0 - self.lambda_phys) * data_loss + self.lambda_phys * phys_loss
        
        grads = tape.gradient(total_loss, self.core_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core_model.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(data_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        
        return {m.name: m.result() for m in self.metrics}

    # --- FIX: Implement the test_step method for validation logic ---
    def test_step(self, data):
        if isinstance(data, tuple):
            x, y_true = data[0], data[1]
        else:
            x, y_true = data

        # 前向传播，注意 training=False
        y_pred = self.core_model(x, training=False)

        # 计算数据损失
        data_loss = self.mse_loss_fn(y_true, y_pred[:, :2])
        
        # 物理重构
        Ip_input = x[:, :, 0]
        Is_input = x[:, :, 1]
        Ip_recon, Is_recon = reconstruct_spectra_tf(y_pred, self.freq_axis_thz)
        
        # 计算物理损失
        phys_loss_p = self.mse_loss_fn(Ip_input, Ip_recon)
        phys_loss_s = self.mse_loss_fn(Is_input, Is_recon)
        phys_loss = phys_loss_p + phys_loss_s

        # 计算总损失
        total_loss = (1.0 - self.lambda_phys) * data_loss + self.lambda_phys * phys_loss

        # 更新监控指标 (与train_step相同)
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
    # --- 1. 参数定义 (超参数化，方便管理) ---
    # 数据参数
    SEQUENCE_LENGTH = 2048
    
    # 模型参数
    NUM_PHYSICAL_PARAMS = 5
    
    # 训练参数
    LAMBDA_PHYS = 0.5       # 物理损失权重 (关键超参数)
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 200            # 设定一个较高的上限，让早停来决定何时停止

    # 回调函数参数
    EARLY_STOPPING_PATIENCE = 20 # 如果验证损失20个周期不下降，则停止
    LR_SCHEDULER_PATIENCE = 10   # 如果验证损失10个周期不下降，则降低学习率
    LR_SCHEDULER_FACTOR = 0.5    # 学习率降低的因子
    
    # 文件路径
    TRAIN_DATA_PATH = 'opsi_dataset_train.h5'
    VAL_DATA_PATH = 'opsi_dataset_val.h5'
    MODEL_SAVE_PATH = 'best_pinn_model.keras' # 保存最佳模型的路径

    SCALER_SAVE_PATH = 'tau_scaler.gz' # 保存缩放器的路径
    
    # --- 2. 加载数据 ---
    print("正在加载训练和验证数据...")
    x_train, y_train = load_data(TRAIN_DATA_PATH)
    x_val, y_val = load_data(VAL_DATA_PATH)
    print("数据加载完成。")
    print(f"训练集形状: x={x_train.shape}, y={y_train.shape}")
    print(f"验证集形状: x={x_val.shape}, y={y_val.shape}")

    # --- NEW: 标签归一化步骤 ---
    print("\n正在进行标签归一化...")
    
    # 2.1 创建并只在训练集的τ上拟合缩放器
    tau_scaler = MinMaxScaler(feature_range=(0, 1))
    # y_train[:, 0] 是所有τ标签。reshape(-1, 1)是sklearn的要求
    tau_scaler.fit(y_train[:, 0].reshape(-1, 1))
    
    # 2.2 保存这个至关重要的缩放器
    joblib.dump(tau_scaler, SCALER_SAVE_PATH)
    print(f"τ缩放器已创建并保存至 {SCALER_SAVE_PATH}")
    print(f"τ的原始范围 (min, max): ({tau_scaler.data_min_[0]:.2f}, {tau_scaler.data_max_[0]:.2f})")

    # 2.3 使用已拟合的缩放器转换训练集和验证集的标签
    # 创建副本以避免修改原始数据
    y_train_scaled = np.copy(y_train)
    y_val_scaled = np.copy(y_val)
    
    y_train_scaled[:, 0] = tau_scaler.transform(y_train[:, 0].reshape(-1, 1)).flatten()
    y_val_scaled[:, 0] = tau_scaler.transform(y_val[:, 0].reshape(-1, 1)).flatten()
    print("训练集和验证集的τ标签已归一化至 [0, 1] 范围。")
    
    # --- 3. 构建并编译模型 ---
    core_model = build_pinn_cnn_lstm_model(
        seq_length=SEQUENCE_LENGTH,
        num_outputs=NUM_PHYSICAL_PARAMS
    )
    pinn_model = PINN_Model(core_model=core_model, lambda_phys=LAMBDA_PHYS)
    pinn_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))
    
    # 这一步是为了让Keras模型知道输入的形状，以便后续保存和加载
    # 我们“假装”训练一个样本
    pinn_model.train_on_batch(x_train[:1], y_train[:1])
    print("\n模型构建并编译完成。")
    core_model.summary()

    # --- 4. 定义回调函数 (Callbacks) ---
    # 这是实现智能化训练的关键
    
    # (a) 模型检查点 (Model Checkpoint): 只保存验证损失最低的那个模型
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss', # 监控验证集的总损失
        save_best_only=True,
        save_weights_only=False, # 保存完整模型
        mode='min',
        verbose=1
    )

    # (b) 早停 (Early Stopping): 防止过拟合和浪费时间
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        mode='min',
        verbose=1,
        restore_best_weights=True # 训练结束后，自动恢复到最佳权重
    )

    # (c) 学习率调度器 (Learning Rate Scheduler)
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        mode='min',
        verbose=1
    )
    
    # --- NEW: (d) TensorBoard 回调 ---
    # 为每次运行创建一个唯一的日志目录，以时间戳命名
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # 每个epoch都记录权重直方图，用于深度分析
        write_graph=True,
        write_images=False
    )

    # 将所有回调函数放入一个列表
    training_callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb] # 新的列表

    # --- 5. 开始训练 ---
    print("\n" + "="*50)
    print("           开始归一化标签完整训练")
    print("="*50)
    
    history = pinn_model.fit(
        x_train, y_train_scaled, # <--- 使用归一化后的训练标签
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val_scaled), # <--- 使用归一化后的验证标签
        callbacks=training_callbacks,
        verbose=1
    )
    
    print("\n训练完成！")
    print(f"最佳模型已保存至: {MODEL_SAVE_PATH}")

    # --- 6. (可选) 绘制训练历史曲线 ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Total Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['data_loss'], label='Training Data Loss')
    plt.plot(history.history['val_data_loss'], label='Validation Data Loss') # Keras会自动添加val_前缀
    plt.plot(history.history['phys_loss'], label='Training Physics Loss')
    plt.plot(history.history['val_phys_loss'], label='Validation Physics Loss')
    plt.title('Component Losses over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("训练历史曲线已保存为 training_history.png")
    plt.show()