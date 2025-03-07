# #这是数据集ETTh1
# import matplotlib.pyplot as plt

# # Patch Length (横坐标)
# Top_k = [1, 3, 5, 7]

# # 不同预测长度对应的 MSE 数据
# pred_len_96 = [0.0574, 0.0574, 0.0571, 0.0574]
# pred_len_192 = [0.0723, 0.0723, 0.0723, 0.0723]
# pred_len_336 = [0.0842, 0.0842, 0.0842, 0.0842]
# pred_len_720 = [0.0911, 0.0911, 0.0911, 0.0911]

# plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# # 绘制不同预测长度对应的曲线
# plt.plot(Top_k, pred_len_96, marker='o', label='pred_len_96', color='blue')
# plt.plot(Top_k, pred_len_192, marker='o', label='pred_len_192', color='orange')
# plt.plot(Top_k, pred_len_336, marker='o', label='pred_len_336', color='green')
# plt.plot(Top_k, pred_len_720, marker='o', label='pred_len_720', color='purple')

# # 设置 x 轴刻度
# plt.xticks(Top_k, [str(x) for x in Top_k])

# # 坐标轴和标题设置
# plt.xlabel('Top K ')
# plt.ylabel('MSE')
# plt.title('ETTh1')
# plt.grid(True, linestyle='--', alpha=0.5)

# # 将图例放在右侧外部，避免遮挡折线
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # 调整布局，留出右侧空间给图例
# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()



#这是数据集ETTh2
import matplotlib.pyplot as plt

# Patch Length (横坐标)
Top_k = [1, 3, 5, 7]

# 不同预测长度对应的 MSE 数据
pred_len_96 = [0.0995, 0.0995, 0.0995, 0.0995]
pred_len_192 = [0.2003, 0.2003, 0.2003, 0.2003]
pred_len_336 = [0.4187, 0.4187, 0.4187, 0.4187]
pred_len_720 = [1.171, 1.172, 1.071, 1.181]

plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# 绘制不同预测长度对应的曲线
plt.plot(Top_k, pred_len_96, marker='o', label='pred_len_96', color='blue')
plt.plot(Top_k, pred_len_192, marker='o', label='pred_len_192', color='orange')
plt.plot(Top_k, pred_len_336, marker='o', label='pred_len_336', color='green')
plt.plot(Top_k, pred_len_720, marker='o', label='pred_len_720', color='purple')

# 设置 x 轴刻度
plt.xticks(Top_k, [str(x) for x in Top_k])

# 坐标轴和标题设置
plt.xlabel('Top K ')
plt.ylabel('MSE')
plt.title('Exchange')
plt.grid(True, linestyle='--', alpha=0.5)

# 将图例放在右侧外部，避免遮挡折线
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 调整布局，留出右侧空间给图例
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()




