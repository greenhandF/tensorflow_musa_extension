import tensorflow as tf
import os

# 加载插件
plugin_path = "/workspace/tensorflow_musa/build/libmusa_plugin.so"
tf.load_library(plugin_path)

print("🚀 启动显式拷贝测试...")

# 1. 定义一个最简单的图函数
@tf.function
def simple_add(x, y):
    print(">> [Graph Construction] Tracing simple_add...")
    return x + y

try:
    with tf.device("/device:MUSA:0"):
        # 2. 在 MUSA 上直接创建 Tensor (避免 H2D 拷贝)
        a = tf.constant([1.0], dtype=tf.float32)
        b = tf.constant([2.0], dtype=tf.float32)
        
        print(f"👉 Tensor a device: {a.device}")
        print(f"👉 Tensor b device: {b.device}")

        # 3. 运行函数
        result = simple_add(a, b)
        print(f"✅ 纯设备内计算成功: {result.numpy()}")

    # 4. 测试 Host 到 Device 的隐式拷贝 (这就是之前报错的场景)
    print("\n🚀 启动隐式 H2D 拷贝测试...")
    c_cpu = tf.constant([3.0], dtype=tf.float32) # 默认在 CPU
    d_cpu = tf.constant([4.0], dtype=tf.float32)
    
    # 这一步会触发 _Arg 节点的输入拷贝
    result_h2d = simple_add(c_cpu, d_cpu)
    print(f"✅ Host输入->图计算成功: {result_h2d.numpy()}")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
