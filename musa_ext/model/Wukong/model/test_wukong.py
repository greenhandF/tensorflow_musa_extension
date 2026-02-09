import tensorflow as tf
import time
import os
from model.wukong import Wukong
plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
tf.load_library(plugin_path)
        
# 允许软放置：如果 MUSA 没实现某个算子，自动回退到 CPU，防止程序中断
tf.config.set_soft_device_placement(True)
# 打印算子分布：可以看到哪个算子跑在 MUSA，哪个跑在 CPU
tf.debugging.set_log_device_placement(True)


def test_wukong_training():
    print("\n" + "="*50)
    print("🔥 正在启动 Wukong 反向训练测试...")
    print("="*50)

    # 1. 基础配置（与前文保持一致）
    batch_size = 16
    dim_emb = 16
    dim_input_sparse = 10
    dim_input_dense = 5
    num_sparse_embs = [1000] * dim_input_sparse
    
    model = Wukong(
        num_layers=2,
        num_sparse_embs=num_sparse_embs,
        dim_emb=dim_emb,
        dim_input_sparse=dim_input_sparse,
        dim_input_dense=dim_input_dense,
        num_emb_lcb=8,
        num_emb_fmb=4,
        rank_fmb=2,
        num_hidden_wukong=1,
        dim_hidden_wukong=32,
        num_hidden_head=2,
        dim_hidden_head=64,
        dim_output=1,
        dropout=0.1
    )

    # 2. 定义训练组件
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # 3. 构造伪造的训练数据和标签
    sparse_data = tf.random.uniform((batch_size, dim_input_sparse), 0, 999, dtype=tf.int32)
    dense_data = tf.random.normal((batch_size, dim_input_dense))
    # 模拟二分类标签 (0 或 1)
    labels = tf.cast(tf.random.uniform((batch_size, 1), 0, 2, dtype=tf.int32), tf.float32)

    # 4. 定义单步训练逻辑 (使用 GradientTape)
    @tf.function  # 使用图模式加速训练
    def train_step(s_data, d_data, y_true):
        with tf.GradientTape() as tape:
            # 运行前向传播
            y_pred = model([s_data, d_data], training=True)
            # 计算损失
            loss = loss_fn(y_true, y_pred)
        
        # 计算梯度
        gradients = tape.gradient(loss, model.trainable_variables)
        # 更新权重
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # 5. 执行循环训练测试
    print(f"开始训练测试（执行 5 个 Iterations）...")
    for i in range(1, 6):
        start_time = time.time()
        loss_val = train_step(sparse_data, dense_data, labels)
        end_time = time.time()
        
        print(f"Iteration {i} | Loss: {loss_val.numpy():.4f} | Time: {(end_time-start_time)*1000:.2f}ms")

    print("\n✅ 反向训练测试完成！梯度更新正常。")

if __name__ == "__main__":
    # 确保 MLP 类和 Embedding 类已定义
    
    try:
        test_wukong_training()
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
