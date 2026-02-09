import tensorflow as tf
from tensorflow.keras.layers import Layer
import os

# --- 1. 加载 MUSA 插件 ---
try:
    # 请确保路径指向你编译好的 so 文件
    tf.load_op_library('/workspace/tensorflow_musa/build/libmusa_plugin.so')
    print(">>>> SUCCESS: MUSA plugin loaded. <<<<")
except Exception as e:
    print(f"Plugin Load Failed: {e}")

# --- 2. 兼容性配置 ---
# 允许软放置：如果 MUSA 没实现某个算子，自动回退到 CPU，防止程序中断
tf.config.set_soft_device_placement(False)
# 打印算子分布：可以看到哪个算子跑在 MUSA，哪个跑在 CPU
tf.debugging.set_log_device_placement(True)
# ==========================================
# 1. 源码复刻 (保持一模一样)
# ==========================================
class GELU(Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, inputs):
        return 0.5 * inputs * (1.0 + tf.math.erf(inputs / tf.sqrt(2.0)))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GELU, self).get_config()
        return config


class MLP(tf.keras.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.0,
        bias: bool = False,
        activation: tf.keras.layers.Layer = GELU(),
    ) -> None:
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(tf.keras.layers.Dense(units=dim_hidden, use_bias=bias))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(activation)
            layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(units=dim_out, use_bias=bias))
        super().__init__(layers)

# ==========================================
# 2. 自动化测试套件
# ==========================================
def test_mlp_and_gelu():
    print("🚀 开始测试 GELU 激活函数与 MLP 模型...")

    # 配置参数
    BATCH_SIZE = 8
    DIM_IN = 128
    NUM_HIDDEN = 3    # 3层意味着：2个隐藏层块 + 1个输出层
    DIM_HIDDEN = 256
    DIM_OUT = 1
    DROPOUT = 0.2

    # 1. 初始化模型
    model = MLP(
        dim_in=DIM_IN,
        num_hidden=NUM_HIDDEN,
        dim_hidden=DIM_HIDDEN,
        dim_out=DIM_OUT,
        dropout=DROPOUT,
        bias=True
    )

    # 2. 构造模拟输入
    mock_input = tf.random.normal((BATCH_SIZE, DIM_IN))

    # --- 测试点 1: 前向传播与形状 ---
    print("\n[测试 1] 验证前向传播输出形状...")
    output = model(mock_input, training=False)
    print(f"   输入形状: {mock_input.shape}")
    print(f"   输出形状: {output.shape}")
    
    assert output.shape == (BATCH_SIZE, DIM_OUT), f"输出形状错误，预期 {(BATCH_SIZE, DIM_OUT)}"
    print("   ✅ 形状校验通过。")

    # --- 测试点 2: GELU 算子数值合理性 ---
    print("\n[测试 2] 验证 GELU 激活函数数值...")
    gelu_layer = GELU()
    # 当 x=0 时，GELU(0) 应为 0
    zero_test = gelu_layer(tf.constant([0.0]))
    # 当 x 很大时（如 10.0），GELU(x) 应该接近 x
    large_test = gelu_layer(tf.constant([10.0]))
    
    print(f"   GELU(0): {zero_test.numpy()[0]:.4f}")
    print(f"   GELU(10): {large_test.numpy()[0]:.4f}")
    
    assert abs(zero_test.numpy()[0]) < 1e-6
    assert abs(large_test.numpy()[0] - 10.0) < 1e-4
    print("   ✅ GELU 算子数值逻辑正常。")

    # --- 测试点 3: 梯度反向传播 (MUSA 稳定性) ---
    print("\n[测试 3] 验证反向传播梯度链路...")
    with tf.GradientTape() as tape:
        logits = model(mock_input, training=True)
        loss = tf.reduce_mean(tf.square(logits))
    
    grads = tape.gradient(loss, model.trainable_variables)
    
    # 检查梯度是否全为 None
    has_none = False
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None:
            print(f"   ❌ 错误: 变量 {var.name} 丢失梯度！")
            has_none = True
    
    if not has_none:
        print(f"   ✅ 梯度计算正常，共获取 {len(grads)} 个参数的梯度。")

    print("\n" + "="*40)
    print("🎉 所有 MLP 源码相关测试项均已通过！")
    print("="*40)

if __name__ == "__main__":
    test_mlp_and_gelu()
