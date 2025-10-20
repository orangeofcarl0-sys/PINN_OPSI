import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import h5py
import numpy as np
from tensorflow.keras import callbacks
import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# 从我们之前的文件中导入模型构建函数和物理仿真函数
from model import build_pinn_cnn_lstm_model
from data_factory import generate_opsi_spectrum # 我们将用它来构建物理重构函数

# ==============================================================================
# 物理重构函数 (PINN核心)
# ==============================================================================
@tf.function
def reconstruct_spectra_tf(params: tf.Tensor, freq_axis_thz: tf.Tensor) -> tuple:
    """使用TensorFlow函数，根据预测的物理参数重构OPSI光谱 (可微分)。"""
    freq_axis_thz = tf.expand_dims(freq_axis_thz, axis=0)
    
    tau_ps = tf.expand_dims(params[:, 0], axis=1)
    phi_rad = tf.expand_dims(params[:, 1], axis=1)
    A_env = tf.expand_dims(params[:, 2], axis=1)
    mu_thz = tf.expand_dims(params[:, 3], axis=1)
    sigma_thz = tf.expand_dims(params[:, 4], axis=1)

    omega_rad_ps = 2.0 * np.pi * freq_axis_thz
    omega_c_rad_ps = 2.0 * np.pi * 194.0

    mu_rad_ps = 2.0 * np.pi * mu_thz
    sigma_rad_ps = 2.0 * np.pi * sigma_thz
    
    I_env = A_env * tf.math.exp(-tf.math.square(omega_rad_ps - mu_rad_ps) / (2.0 * tf.math.square(sigma_rad_ps)))
    
    phase_term = (omega_rad_ps - omega_c_rad_ps) * tau_ps + phi_rad
    cos_term = tf.math.cos(phase_term)

    Ip_recon = I_env * (1.0 + cos_term)
    Is_recon = I_env * (1.0 - cos_term)

    return Ip_recon, Is_recon

# ==============================================================================
# 自定义PINN模型类 (包含所有修复和新功能)
# ==============================================================================
class PINN_Model(Model):
    def __init__(self, core_model, lambda_phys=0.5, grad_loss_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.core_model = core_model
        self.lambda_phys = lambda_phys
        self.grad_loss_weight = grad_loss_weight
        
        self.freq_axis_thz = tf.constant(np.linspace(190, 198, 2048), dtype=tf.float32)
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.data_loss_tracker = tf.keras.metrics.Mean(name="data_loss")
        self.phys_loss_tracker = tf.keras.metrics.Mean(name="phys_loss")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "core_model": tf.keras.utils.serialize_keras_object(self.core_model),
            "lambda_phys": self.lambda_phys,
            "grad_loss_weight": self.grad_loss_weight,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        core_model_config = config.pop("core_model")
        core_model = tf.keras.layers.deserialize(core_model_config)
        return cls(core_model, **config)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
    
    def call(self, inputs, training=False):
        return self.core_model(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.data_loss_tracker, self.phys_loss_tracker]

    def compute_pinn_loss(self, x, y_true=None, training=False):
        y_pred = self.core_model(x, training=training)

        Ip_input = x[:, :, 0]
        Is_input = x[:, :, 1]
        Ip_recon, Is_recon = reconstruct_spectra_tf(y_pred, self.freq_axis_thz)
        
        phys_loss_time = self.mse_loss_fn(Ip_input, Ip_recon) + self.mse_loss_fn(Is_input, Is_recon)
        
        grad_input_p = Ip_input[:, 1:] - Ip_input[:, :-1]
        grad_recon_p = Ip_recon[:, 1:] - Ip_recon[:, :-1]
        grad_input_s = Is_input[:, 1:] - Is_input[:, :-1]
        grad_recon_s = Is_recon[:, 1:] - Is_recon[:, :-1]
        phys_loss_grad = self.mse_loss_fn(grad_input_p, grad_recon_p) + self.mse_loss_fn(grad_input_s, grad_recon_s)

        phys_loss = phys_loss_time + self.grad_loss_weight * phys_loss_grad

        if y_true is not None:
            data_loss = self.mse_loss_fn(y_true, y_pred[:, :2])
            total_loss = (1.0 - self.lambda_phys) * data_loss + self.lambda_phys * phys_loss
        else:
            data_loss = 0.0
            total_loss = phys_loss
        
        return total_loss, data_loss, phys_loss

    def train_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            x, y_true = data
        else:
            x, y_true = data, None

        with tf.GradientTape() as tape:
            total_loss, data_loss, phys_loss = self.compute_pinn_loss(x, y_true, training=True)
        
        grads = tape.gradient(total_loss, self.core_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core_model.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(data_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            x, y_true = data
        else:
            x, y_true = data, None

        total_loss, data_loss, phys_loss = self.compute_pinn_loss(x, y_true, training=False)

        self.loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(data_loss)
        self.phys_loss_tracker.update_state(phys_loss)
        
        return {m.name: m.result() for m in self.metrics}

# ==============================================================================
# 辅助函数
# ==============================================================================
def load_data(filepath):
    """从HDF5文件加载数据。"""
    with h5py.File(filepath, 'r') as f:
        ip = f['spectra_ip'][:]
        is_spec = f['spectra_is'][:]
        tau = f['labels_tau'][:]
        phi = f['labels_phi'][:]
    
    x = np.stack([ip, is_spec], axis=-1)
    y = np.stack([tau, phi], axis=-1)
    
    return x, y

def plot_history(history, stage_name):
    """绘制训练历史曲线。"""
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'Training History - {stage_name}')
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['data_loss'], label='Training Data Loss')
    plt.plot(history.history['val_data_loss'], label='Validation Data Loss')
    plt.plot(history.history['phys_loss'], label='Training Physics Loss')
    plt.plot(history.history['val_phys_loss'], label='Validation Physics Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'training_history_{stage_name.lower()}.png')
    print(f"训练历史曲线已保存为 training_history_{stage_name.lower()}.png")
    plt.show()

# ==============================================================================
# 主训练流程
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 参数定义 ---
    SEQUENCE_LENGTH = 2048
    NUM_PHYSICAL_PARAMS = 5
    BATCH_SIZE = 64
    
    PRETRAIN_EPOCHS = 50
    PRETRAIN_LR = 1e-3
    
    FINETUNE_EPOCHS = 200
    FINETUNE_LR = 1e-4
    LAMBDA_PHYS = 0.5
    GRAD_LOSS_WEIGHT = 0.1
    
    EARLY_STOPPING_PATIENCE = 20
    LR_SCHEDULER_PATIENCE = 10
    
    TRAIN_DATA_PATH = 'opsi_dataset_train.h5'
    VAL_DATA_PATH = 'opsi_dataset_val.h5'
    PRETRAINED_WEIGHTS_PATH = 'pretrained_weights.weights.h5'
    FINAL_MODEL_SAVE_PATH = 'best_pinn_model.keras'
    SCALER_SAVE_PATH = 'tau_scaler.gz'

    # --- 2. 数据准备 ---
    print("正在加载数据并进行标签归一化...")
    x_train, y_train = load_data(TRAIN_DATA_PATH)
    x_val, y_val = load_data(VAL_DATA_PATH)
    
    tau_scaler = MinMaxScaler(feature_range=(0, 1))
    tau_scaler.fit(y_train[:, 0].reshape(-1, 1))
    joblib.dump(tau_scaler, SCALER_SAVE_PATH)
    
    y_train_scaled = np.copy(y_train)
    y_val_scaled = np.copy(y_val)
    y_train_scaled[:, 0] = tau_scaler.transform(y_train[:, 0].reshape(-1, 1)).flatten()
    y_val_scaled[:, 0] = tau_scaler.transform(y_val[:, 0].reshape(-1, 1)).flatten()
    print("数据准备完成。")

    # ==============================================================================
    # 阶段一：物理预训练
    # ==============================================================================
    print("\n" + "="*50)
    print("           阶段一：物理预训练 (λ=1.0)")
    print("="*50)
    
    core_model_pretrain = build_pinn_cnn_lstm_model(SEQUENCE_LENGTH)
    pinn_pretrain = PINN_Model(core_model=core_model_pretrain, lambda_phys=1.0, grad_loss_weight=GRAD_LOSS_WEIGHT)
    pinn_pretrain.compile(optimizer=optimizers.Adam(learning_rate=PRETRAIN_LR))

    pretrain_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1),
        callbacks.TensorBoard(log_dir="logs/pretrain/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ]

    pretrain_history = pinn_pretrain.fit(
        x_train, y=None,
        batch_size=BATCH_SIZE,
        epochs=PRETRAIN_EPOCHS,
        validation_data=(x_val, None),
        callbacks=pretrain_callbacks,
        verbose=1
    )
    
    core_model_pretrain.save_weights(PRETRAINED_WEIGHTS_PATH)
    print(f"\n物理预训练完成，最佳权重已保存至 {PRETRAINED_WEIGHTS_PATH}")
    plot_history(pretrain_history, "Pre-training")

    # ==============================================================================
    # 阶段二：数据微调
    # ==============================================================================
    print("\n" + "="*50)
    print(f"           阶段二：数据微调 (λ={LAMBDA_PHYS})")
    print("="*50)

    core_model_finetune = build_pinn_cnn_lstm_model(SEQUENCE_LENGTH)
    core_model_finetune.load_weights(PRETRAINED_WEIGHTS_PATH)
    print(f"已成功加载预训练权重从 {PRETRAINED_WEIGHTS_PATH}")
    
    pinn_finetune = PINN_Model(core_model=core_model_finetune, lambda_phys=LAMBDA_PHYS, grad_loss_weight=GRAD_LOSS_WEIGHT)
    pinn_finetune.compile(optimizer=optimizers.Adam(learning_rate=FINETUNE_LR))

    finetune_callbacks = [
        callbacks.ModelCheckpoint(filepath=FINAL_MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=LR_SCHEDULER_PATIENCE, verbose=1),
        callbacks.TensorBoard(log_dir="logs/finetune/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ]

    finetune_history = pinn_finetune.fit(
        x_train, y_train_scaled,
        batch_size=BATCH_SIZE,
        epochs=FINETUNE_EPOCHS,
        validation_data=(x_val, y_val_scaled),
        callbacks=finetune_callbacks,
        verbose=1
    )

    print("\n微调训练完成！")
    print(f"最终最佳模型已保存至: {FINAL_MODEL_SAVE_PATH}")
    plot_history(finetune_history, "Fine-tuning")

    # # --- 6. (可选) 绘制训练历史曲线 ---
    # 

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Total Loss over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['data_loss'], label='Training Data Loss')
    # plt.plot(history.history['val_data_loss'], label='Validation Data Loss') # Keras会自动添加val_前缀
    # plt.plot(history.history['phys_loss'], label='Training Physics Loss')
    # plt.plot(history.history['val_phys_loss'], label='Validation Physics Loss')
    # plt.title('Component Losses over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('training_history.png')
    # print("训练历史曲线已保存为 training_history.png")
    # plt.show()