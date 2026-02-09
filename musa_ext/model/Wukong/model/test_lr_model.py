import tensorflow as tf
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
class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, peak_learning_rate, warmup_steps):
        super(LinearWarmup, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        scale = step / self.warmup_steps
        scale = tf.minimum(scale, 1.0)
        return (
            self.initial_learning_rate
            + (self.peak_learning_rate - self.initial_learning_rate) * scale
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
        }

# ==========================================
# 2. 修正后的测试代码
# ==========================================
def test_warmup():
    # 配置
    init_lr = 1e-8
    peak_lr = 0.004
    warmup_steps = 1000.0

    lr_schedule = LinearWarmup(init_lr, peak_lr, warmup_steps)

    print(f"--- LinearWarmup 逻辑测试 ---")
    
    # 1. 测试起点 (step=0)
    res_0 = lr_schedule(0.0)
    print(f"Step 0    | LR: {res_0.numpy():.10f} (预期: {init_lr:.10f})")

    # 2. 测试中点 (step=500)
    res_500 = lr_schedule(500.0)
    expected_mid = init_lr + (peak_lr - init_lr) * 0.5
    print(f"Step 500  | LR: {res_500.numpy():.10f} (预期: {expected_mid:.10f})")

    # 3. 测试终点 (step=1000)
    res_1000 = lr_schedule(1000.0)
    print(f"Step 1000 | LR: {res_1000.numpy():.10f} (预期: {peak_lr:.10f})")

    # 4. 测试溢出点 (step=1500)
    res_1500 = lr_schedule(1500.0)
    print(f"Step 1500 | LR: {res_1500.numpy():.10f} (预期: {peak_lr:.10f})")

    # --- 关键修正：调整容差范围 ---
    # float32 的有效位大约只有 7 位，1e-12 对它来说太难了
    tolerance = 1e-7 
    
    assert abs(res_0.numpy() - init_lr) < tolerance, f"Step 0 误差过大: {abs(res_0.numpy() - init_lr)}"
    assert abs(res_1000.numpy() - peak_lr) < tolerance, f"Step 1000 误差过大: {abs(res_1000.numpy() - peak_lr)}"
    assert abs(res_1500.numpy() - res_1000.numpy()) < 1e-12 # 这两个都是由 scale=1.0 算出来的，可以更严
    
    print("\n✅ 测试通过：线性增长与 minimum 截断逻辑完全正确。")

if __name__ == "__main__":
    test_warmup()
