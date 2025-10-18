import tensorflow as tf
from tensorflow.keras import layers, Model

# 指定随机数种子，保证可复现性
tf.keras.utils.set_random_seed(42)

def build_pinn_cnn_lstm_model(seq_length: int = 2048, num_outputs: int = 5) -> Model:
    """
    构建一个轻量、高效的PINN-CNN-LSTM混合模型。

    该模型分工明确：
    1. CNN模块：用于高效的局部特征提取和序列降维。
    2. LSTM模块：用于在浓缩后的特征序列上捕捉长程依赖关系。
    3. 输出头：用于回归最终的物理参数。

    参数:
        seq_length (int): 输入光谱序列的长度。
        num_outputs (int): 模型需要预测的物理参数数量。
                           对于我们的PINN方案，这通常是5个：(τ, φ, A_env, μ_env, σ_env)。

    返回:
        tf.keras.Model: 一个未编译的Keras模型实例。
    """
    # --- 1. 输入层 ---
    # 定义模型的入口，形状为 (序列长度, 2个通道)，2个通道分别代表 Ip 和 Is
    inputs = layers.Input(shape=(seq_length, 2), name='opsi_spectra_input')

    # --- 2. CNN特征提取模块 (The Feature Extractor) ---
    # 目标：将长序列 (2048) 转换为短而精的特征序列 (128)
    
    # Block 1
    x = layers.Conv1D(filters=16, kernel_size=11, padding='same', activation='relu', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(pool_size=4, name='pool1')(x)  # 序列长度: 2048 -> 512
    
    # Block 2
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(pool_size=4, name='pool2')(x)  # 序列长度: 512 -> 128

    # --- 3. LSTM序列处理模块 (The Sequence Processor) ---
    # 目标：在浓缩后的特征序列 (长度128) 上理解全局模式
    
    # 使用双向LSTM，它可以同时从正向和反向学习序列信息，增强上下文理解能力
    # units=64 表示每个方向有64个LSTM单元，总输出维度为128
    x = layers.Bidirectional(layers.LSTM(units=64, name='bilstm1'))(x)

    # --- 4. 输出头 (The Regression Head) ---
    # 目标：将LSTM的最终状态向量映射到物理参数
    
    outputs = layers.Dense(units=num_outputs, activation='linear', name='output_params')(x)

    # --- 5. 组装模型 ---
    model = Model(inputs=inputs, outputs=outputs, name='PINN_CNN_LSTM')

    return model

if __name__ == '__main__':
    # --- 模型实例化 ---
    # 定义模型的超参数
    SEQUENCE_LENGTH = 2048
    NUM_PHYSICAL_PARAMS = 5 # τ, φ, A_env, μ_env, σ_env

    # 调用我们编写的函数来构建模型
    model = build_pinn_cnn_lstm_model(
        seq_length=SEQUENCE_LENGTH,
        num_outputs=NUM_PHYSICAL_PARAMS
    )

    # --- 打印模型摘要 ---
    # model.summary() 是最重要的调试工具之一。
    # 它可以清晰地展示每一层的名称、输出形状和参数数量。
    print("\n" + "="*50)
    print("           模型架构摘要 (Model Architecture Summary)")
    print("="*50)
    model.summary()

    # --- (可选) 可视化模型 ---
    # 如果你安装了 pydot 和 graphviz 库 (pip install pydot graphviz)
    # 你可以将模型架构保存为一张图片，更加直观。
    try:
        tf.keras.utils.plot_model(
            model,
            to_file='model_architecture.png',
            show_shapes=True,
            show_layer_activations=True
        )
        print("\n模型架构图已保存为 model_architecture.png")
    except ImportError:
        print("\n请安装 pydot 和 graphviz 以生成模型架构图。")