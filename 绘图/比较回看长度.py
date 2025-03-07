import matplotlib.pyplot as plt

# Patch Length (横坐标)
prediction_length = [96, 192, 336, 720]

# 不同预测长度对应的 MSE，示例数据可替换为真实实验结果
patch_length_8 = [0.0571, 0.0736, 0.0820, 0.0909]
patch_length_16 = [0.0571, 0.0723, 0.0842, 0.0911]
patch_length_24 = [0.0573, 0.0736, 0.0820, 0.0909]
patch_length_32 = [0.0567, 0.0723, 0.0852, 0.0866]

plt.figure(figsize=(6, 4))  # 可根据需要调整图像大小

# 绘制不同预测长度对应的曲线
plt.plot(prediction_length, patch_length_8, marker='o', label='patch_length_8', color='blue')
plt.plot(prediction_length, patch_length_16, marker='o', label='patch_length_16', color='orange')
plt.plot(prediction_length, patch_length_24, marker='o', label='patch_length_24', color='green')
plt.plot(prediction_length, patch_length_32, marker='o', label='patch_length_32', color='purple')

# 设置 x 轴刻度，使其只显示 8, 16, 24, 32
plt.xticks(prediction_length, [str(x) for x in prediction_length])

# 坐标轴和标题设置
plt.xlabel('Prediction Length')
plt.ylabel('MSE')
plt.title('ETTh1')

# 打开网格（可选）
plt.grid(True, linestyle='--', alpha=0.5)

# # 显示图例
# plt.legend()
# 将图例放在右侧外部，避免遮挡折线
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# 自动布局并显示图像
plt.tight_layout()
plt.show()
