import tensorflow as tf
from typing import List
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
# 1. 你的源码（保持一模一样，不做任何修改）
# ==========================================
class SparseEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_sparse_embs, dim_emb):
        super().__init__()
        self.embeddings = [
            tf.keras.layers.Embedding(input_dim=num_emb, output_dim=dim_emb)
            for num_emb in num_sparse_embs
        ]

    def call(self, sparse_inputs):
        sparse_outputs = [
            embedding(sparse_inputs[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return tf.stack(sparse_outputs, axis=1)


class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_dense: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense

        self.sparse_embedding = SparseEmbedding(num_sparse_embs, dim_emb)
        self.dense_embedding = tf.keras.layers.Dense(
            units=dim_input_dense * dim_emb, use_bias=bias
        )

    def call(self, sparse_inputs: tf.Tensor, dense_inputs: tf.Tensor) -> tf.Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)

        dense_outputs = self.dense_embedding(dense_inputs)
        dense_outputs = tf.reshape(
            dense_outputs, [-1, self.dim_input_dense, self.dim_emb]
        )

        # concat along feature axis
        return tf.concat((sparse_outputs, dense_outputs), axis=1)

# ==========================================
# 2. 自动化测试套件
# ==========================================
def run_comprehensive_test():
    print("开始测试原版 Embedding 模型...")

    # 配置参数
    BATCH_SIZE = 4
    SPARSE_VOCAB_SIZES = [100, 200, 300, 400] # 4个稀疏特征
    DENSE_FEATURE_COUNT = 5                  # 5个密集特征
    EMB_DIM = 16

    # 初始化模型
    model = Embedding(
        num_sparse_embs=SPARSE_VOCAB_SIZES,
        dim_emb=EMB_DIM,
        dim_input_dense=DENSE_FEATURE_COUNT
    )

    # 构造模拟数据
    mock_sparse = tf.random.uniform((BATCH_SIZE, len(SPARSE_VOCAB_SIZES)), 0, 100, dtype=tf.int32)
    mock_dense = tf.random.normal((BATCH_SIZE, DENSE_FEATURE_COUNT))

    # --- 测试点 1: 前向传播与形状校验 ---
    print("\n[测试 1] 验证前向传播形状...")
    output = model(mock_sparse, mock_dense)
    
    # 预期形状计算: 
    # Sparse(4 features) + Dense(5 features) = 9 total features
    # 每个 feature 维度是 16
    # 结果应为 (4, 9, 16)
    expected_shape = [BATCH_SIZE, len(SPARSE_VOCAB_SIZES) + DENSE_FEATURE_COUNT, EMB_DIM]
    print(f"   输出形状: {output.shape}")
    assert list(output.shape) == expected_shape, f"形状错误！预期 {expected_shape}"
    print("   ✅ 形状校验通过。")

    # --- 测试点 2: 梯度与反向传播 (验证 MUSA 算子稳定性) ---
    print("\n[测试 2] 验证梯度计算 (Backward)...")
    with tf.GradientTape() as tape:
        res = model(mock_sparse, mock_dense)
        loss = tf.reduce_sum(res) # 简单的求和损失
    
    grads = tape.gradient(loss, model.trainable_variables)
    
    # 检查所有变量是否都拿到了梯度
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None:
            print(f"   ❌ 错误: 变量 {var.name} 未获得梯度！")
            return
    print("   ✅ 所有参数梯度计算正常。")

    # --- 测试点 3: 内部组件校验 ---
    print("\n[测试 3] 验证子组件 SparseEmbedding 输出...")
    sparse_res = model.sparse_embedding(mock_sparse)
    assert sparse_res.shape == [BATCH_SIZE, len(SPARSE_VOCAB_SIZES), EMB_DIM]
    print(f"   Sparse 子模块输出形状: {sparse_res.shape} ✅")

    print("\n" + "="*30)
    print("🎉 所有测试项均已通过！模型在当前环境下运行稳定。")

if __name__ == "__main__":
    run_comprehensive_test()
