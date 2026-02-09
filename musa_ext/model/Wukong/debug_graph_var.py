import tensorflow as tf

plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
tf.load_library(plugin_path)

print("🚀 启动图模式变量更新测试...")

class SimpleModel(tf.Module):
    def __init__(self):
        # 创建一个变量
        self.w = tf.Variable([1.0], dtype=tf.float32)

    @tf.function
    def __call__(self, x):
        return self.w * x

model = SimpleModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

try:
    x = tf.constant([2.0], dtype=tf.float32)
    y_target = tf.constant([4.0], dtype=tf.float32)

    @tf.function
    def train_step(x_in, y_in):
        with tf.GradientTape() as tape:
            y_pred = model(x_in)
            loss = (y_pred - y_in) ** 2
        
        grads = tape.gradient(loss, [model.w])
        optimizer.apply_gradients(zip(grads, [model.w]))
        return loss

    print("👉 开始执行 train_step...")
    loss = train_step(x, y_target)
    print(f"✅ 训练步成功，Loss: {loss.numpy()}")
    print(f"✅ 更新后的权重: {model.w.numpy()}")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
