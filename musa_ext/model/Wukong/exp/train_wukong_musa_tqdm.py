import os
os.environ['MUSA_DEBUG'] = '0'
os.environ['MUSA_TRACE'] = '0'        # 关掉 MUSA_TRACE_AUTO
os.environ['MUSA_LOG_LEVEL'] = '3'    # 只打印 Error
os.environ['MUSA_VERBOSE'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import numpy as np
import random
import sys
from datetime import datetime
from tqdm import tqdm

from model.wukong import Wukong
from model.lr_schedule import LinearWarmup
from data.criteo_kaggle_dataset import get_dataset

# --- 1. 加载 MUSA 插件 ---
try:
    # 请确保路径指向你编译好的 so 文件
    tf.load_op_library('/workspace/tensorflow_musa/build/libmusa_plugin.so')
    print(">>>> SUCCESS: MUSA plugin loaded. <<<<")
except Exception as e:
    print(f"Plugin Load Failed: {e}")


####################################################################################################
#                                           SET RANDOM SEEDS                                       #
####################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

####################################################################################################
#                                         TENSORBOARD SETUP                                        #
####################################################################################################
# 只保留 TensorBoard 用于可视化，去掉了文本日志文件
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H.%M.%S")

summary_writer = tf.summary.create_file_writer(
    f"logs/tensorflow/{formatted_time}/tensorboard"
)

# 检查点保存路径 (如果开启的话)
checkpoint_dir = f"logs/tensorflow/{formatted_time}/checkpoints"
SAVE_CHECKPOINTS = False
if SAVE_CHECKPOINTS:
    os.makedirs(checkpoint_dir, exist_ok=True)


LOGGER_PRINT_INTERVAL = 100 

####################################################################################################
#                                  DATASET SPECIFIC CONFIGURATION                                  #
####################################################################################################
NPZ_FILE_PATH = "/workspace/tensorflow_musa/model/Wukong/data/kaggleAdDisplayChallenge_processed.npz"
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
NUM_SPARSE_EMBS = [
    1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 
    5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 
    7046547, 18, 15, 286181, 105, 142572,
]
DIM_OUTPUT = 1

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
NUM_LAYERS = 8
DIM_EMB = 128
NUM_EMB_LCB = 32
NUM_EMB_FMB = 32
RANK_FMB = 24
NUM_HIDDEN_WUKONG = 3
DIM_HIDDEN_WUKONG = 2048
NUM_HIDDEN_HEAD = 2
DIM_HIDDEN_HEAD = 256
DROPOUT = 0.5
BIAS = True

####################################################################################################
#                                           CREATE MODEL                                           #
####################################################################################################
model = Wukong(
    num_layers=NUM_LAYERS,
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=NUM_EMB_LCB,
    num_emb_fmb=NUM_EMB_FMB,
    rank_fmb=RANK_FMB,
    num_hidden_wukong=NUM_HIDDEN_WUKONG,
    dim_hidden_wukong=DIM_HIDDEN_WUKONG,
    num_hidden_head=NUM_HIDDEN_HEAD,
    dim_hidden_head=DIM_HIDDEN_HEAD,
    dim_output=DIM_OUTPUT,
    dropout=DROPOUT,
    bias=BIAS,
)

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
BATCH_SIZE = 16384
TRAIN_EPOCHS = 1 
# [修改] 降低学习率以防止 Loss=nan
PEAK_LR = 0.0004
INIT_LR = 1e-8

TOTAL_STEPS_PER_EPOCH = 39291958 // BATCH_SIZE
TOTAL_ITERS = TOTAL_STEPS_PER_EPOCH * TRAIN_EPOCHS 

lr_schedule = LinearWarmup(
    initial_learning_rate=INIT_LR, peak_learning_rate=PEAK_LR, warmup_steps=TOTAL_ITERS
)
embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
other_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)

####################################################################################################
#                                       CREATE DATALOADER                                          #
####################################################################################################

train_dataset = get_dataset(
    npz_file_path=NPZ_FILE_PATH,
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
)

valid_dataset = get_dataset(
    npz_file_path=NPZ_FILE_PATH,
    split="valid",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

####################################################################################################
#                                    BUILD MODEL & SEPARATE VARS                                   #
####################################################################################################
# TF Lazy Execution Warmup
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
_ = model((dummy_sparse, dummy_dense))

embedding_parameters = []
other_parameters = []

for var in model.trainable_variables:
    if hasattr(var, "path"):
        if "sparse_embedding" in var.path and "embeddings" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)
    else:
        if "sparse_embedding" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)

print(f"Number of embedding parameters: {len(embedding_parameters)}")
print(f"Number of other parameters: {len(other_parameters)}")


####################################################################################################
#                                          VALID FUNCTION                                          #
####################################################################################################
def validate(model, dataset):
    num_samples = 0
    num_correct = 0
    pos_samples = 0
    pos_correct = 0

    print("Running Validation...")
    # leave=False 意味着验证完后进度条会消失，保持界面清爽
    for inputs, labels in tqdm(dataset, desc="Validating", leave=False):
        outputs = model(inputs, training=False)
        labels = tf.cast(labels, tf.float32)
        outputs = tf.squeeze(outputs)
        predictions = tf.cast(outputs >= 0.5, tf.float32)

        num_samples += labels.shape[0]
        pos_samples += tf.reduce_sum(labels).numpy()

        correct_preds = tf.cast(tf.equal(predictions, labels), tf.float32)
        num_correct += tf.reduce_sum(correct_preds).numpy()

        pos_mask = tf.equal(labels, 1.0)
        pos_correct += tf.reduce_sum(tf.boolean_mask(predictions, pos_mask)).numpy()

    accuracy = num_correct / num_samples if num_samples > 0 else 0
    recall_pos = pos_correct / pos_samples if pos_samples > 0 else 0
    return accuracy, num_samples, recall_pos, pos_samples


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, tf.squeeze(outputs))

    grads = tape.gradient(loss, model.trainable_variables)
    emb_grads = []
    other_grads = []

    for grad, var in zip(grads, model.trainable_variables):
        if grad is not None:
            if hasattr(var, "path"):
                if "sparse_embedding" in var.path and "embeddings" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))
            else:
                if "sparse_embedding" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))
    embedding_optimizer.apply_gradients(emb_grads)
    other_optimizer.apply_gradients(other_grads)

    return loss


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
step = 0

print(f"Start Training for {TRAIN_EPOCHS} Epochs...")

for epoch in range(TRAIN_EPOCHS):
    # 使用 tqdm 包装训练循环
    with tqdm(total=TOTAL_STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}", unit="batch") as pbar:
        for batch_idx, (inputs, labels) in enumerate(train_dataset):
            labels = tf.cast(labels, tf.float32)

            loss = train_step(inputs, labels)
            current_lr = lr_schedule(step)

            # 实时更新进度条上的 Loss 和 LR
            pbar.set_postfix({
                "Loss": f"{loss.numpy():.4f}", 
                "LR": f"{current_lr.numpy():.6f}"
            })
            pbar.update(1)

            # 定期在控制台打印一条持久化日志 (防止进度条刷新后看不到历史 Loss)
            if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
                # 使用 pbar.write 可以避免打断进度条的渲染
                pbar.write(
                    f"Epoch [{epoch+1}/{TRAIN_EPOCHS}] "
                    f"Batch [{batch_idx+1}/{TOTAL_STEPS_PER_EPOCH}] "
                    f"Loss: {loss.numpy():.4f} "
                    f"LR: {current_lr.numpy():.6f}"
                )

            # 写入 TensorBoard (这部分保留，因为不占 IO 且很有用)
            with summary_writer.as_default():
                tf.summary.scalar("training_loss", loss, step=step)
                tf.summary.scalar("optimizer_lr", current_lr, step=step)

            step += 1

    # 验证阶段
    accuracy, num_samples, recall_pos, pos_samples = validate(model, valid_dataset)

    val_msg = (
        f"Validation after Epoch {epoch+1}: "
        f"Accuracy: {accuracy*100:.2f}%, "
        f"Total Samples: {num_samples}, "
        f"Positive Recall: {recall_pos*100:.2f}%, "
        f"Positive Samples: {pos_samples}"
    )
    print(f"\n{val_msg}\n")

    with summary_writer.as_default():
        tf.summary.scalar("validation_accuracy", accuracy, step=epoch + 1)
        tf.summary.scalar("validation_recall_pos", recall_pos, step=epoch + 1)

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f"wukong_epoch_{epoch+1}")
        model.save_weights(ckpt_path)
        print(f"Model checkpoint saved for epoch {epoch+1} at {ckpt_path}")

print("Training Finished.")