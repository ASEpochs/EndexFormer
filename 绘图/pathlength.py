# #这是数据集ETTh1
# import matplotlib.pyplot as plt

# # Patch Length (横坐标)
# patch_length = [8, 16, 24, 32]

# # 不同预测长度对应的 MSE 数据
# pred_len_96 = [0.0571, 0.0571, 0.0573, 0.0567]
# pred_len_192 = [0.0736, 0.0723, 0.0736, 0.0723]
# pred_len_336 = [0.0820, 0.0842, 0.0820, 0.0825]
# pred_len_720 = [0.0909, 0.0911, 0.0909, 0.0866]

# plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# # 绘制不同预测长度对应的曲线
# plt.plot(patch_length, pred_len_96, marker='o', label='pred_len_96', color='blue')
# plt.plot(patch_length, pred_len_192, marker='o', label='pred_len_192', color='orange')
# plt.plot(patch_length, pred_len_336, marker='o', label='pred_len_336', color='green')
# plt.plot(patch_length, pred_len_720, marker='o', label='pred_len_720', color='purple')

# # 设置 x 轴刻度
# plt.xticks(patch_length, [str(x) for x in patch_length])

# # 坐标轴和标题设置
# plt.xlabel('Patch Length')
# plt.ylabel('MSE')
# plt.title('ETTh1')
# plt.grid(True, linestyle='--', alpha=0.5)

# # 将图例放在右侧外部，避免遮挡折线
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # 调整布局，留出右侧空间给图例
# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()





##这是数据集ECL

# import matplotlib.pyplot as plt

# # Patch Length (横坐标)
# patch_length = [8, 16, 24, 32]

# # 不同预测长度对应的 MSE 数据
# pred_len_96 = [0.2777, 0.2779, 0.2874, 0.2716]
# pred_len_192 = [0.3156, 0.3108, 0.3156, 0.3121]
# pred_len_336 = [0.3667, 0.3710, 0.3717, 0.3628]
# pred_len_720 = [0.4446, 0.4349, 0.4510, 0.4782]

# plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# # 绘制不同预测长度对应的曲线
# plt.plot(patch_length, pred_len_96, marker='o', label='pred_len_96', color='blue')
# plt.plot(patch_length, pred_len_192, marker='o', label='pred_len_192', color='orange')
# plt.plot(patch_length, pred_len_336, marker='o', label='pred_len_336', color='green')
# plt.plot(patch_length, pred_len_720, marker='o', label='pred_len_720', color='purple')

# # 设置 x 轴刻度
# plt.xticks(patch_length, [str(x) for x in patch_length])

# # 坐标轴和标题设置
# plt.xlabel('Patch Length')
# plt.ylabel('MSE')
# plt.title('ECL')
# plt.grid(True, linestyle='--', alpha=0.5)

# # 将图例放在右侧外部，避免遮挡折线
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # 调整布局，留出右侧空间给图例
# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()




# #这是数据集ETTh2
# import matplotlib.pyplot as plt

# # Patch Length (横坐标)
# patch_length = [8, 16, 24, 32]

# # 不同预测长度对应的 MSE 数据
# pred_len_96 = [0.1311, 0.1325 , 0.1342, 0.1348 ]
# pred_len_192 = [0.1811, 0.1811, 0.1867 ,0.1834]
# pred_len_336 = [0.2221, 0.2251, 0.2221, 0.2291]
# pred_len_720 = [0.2403, 0.2467, 0.2288, 0.2353]

# plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# # 绘制不同预测长度对应的曲线
# plt.plot(patch_length, pred_len_96, marker='o', label='pred_len_96', color='blue')
# plt.plot(patch_length, pred_len_192, marker='o', label='pred_len_192', color='orange')
# plt.plot(patch_length, pred_len_336, marker='o', label='pred_len_336', color='green')
# plt.plot(patch_length, pred_len_720, marker='o', label='pred_len_720', color='purple')

# # 设置 x 轴刻度
# plt.xticks(patch_length, [str(x) for x in patch_length])

# # 坐标轴和标题设置
# plt.xlabel('Patch Length')
# plt.ylabel('MSE')
# plt.title('ETTh2')
# plt.grid(True, linestyle='--', alpha=0.5)

# # 将图例放在右侧外部，避免遮挡折线
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # 调整布局，留出右侧空间给图例
# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()




# #这是数据集Weather
# import matplotlib.pyplot as plt

# # Patch Length (横坐标)
# patch_length = [8, 16, 24, 32]

# # 不同预测长度对应的 MSE 数据
# pred_len_96 = [0.0013, 0.0012 ,0.0013, 0.0013 ]
# pred_len_192 = [0.0016, 0.0015, 0.0016 ,0.0015]
# pred_len_336 = [0.0018, 0.0016, 0.0018, 0.0017]
# pred_len_720 = [0.0022, 0.002, 0.0022, 0.0021]

# plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# # 绘制不同预测长度对应的曲线
# plt.plot(patch_length, pred_len_96, marker='o', label='pred_len_96', color='blue')
# plt.plot(patch_length, pred_len_192, marker='o', label='pred_len_192', color='orange')
# plt.plot(patch_length, pred_len_336, marker='o', label='pred_len_336', color='green')
# plt.plot(patch_length, pred_len_720, marker='o', label='pred_len_720', color='purple')

# # 设置 x 轴刻度
# plt.xticks(patch_length, [str(x) for x in patch_length])

# # 坐标轴和标题设置
# plt.xlabel('Patch Length')
# plt.ylabel('MSE')
# plt.title('Weather')
# plt.grid(True, linestyle='--', alpha=0.5)

# # 将图例放在右侧外部，避免遮挡折线
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # 调整布局，留出右侧空间给图例
# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()


# #这是数据集Exchange
# import matplotlib.pyplot as plt

# # Patch Length (横坐标)
# patch_length = [8, 16, 24, 32]

# # 不同预测长度对应的 MSE 数据
# pred_len_96 = [0.1167, 0.0995 ,0.0985, 0.1029 ]
# pred_len_192 = [0.2048, 0.2003, 0.2298 ,0.2189]
# pred_len_336 = [0.4456, 0.4187, 0.4537, 0.4202]
# pred_len_720 = [1.2681, 1.071, 1.2736, 1.1403]

# plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# # 绘制不同预测长度对应的曲线
# plt.plot(patch_length, pred_len_96, marker='o', label='pred_len_96', color='blue')
# plt.plot(patch_length, pred_len_192, marker='o', label='pred_len_192', color='orange')
# plt.plot(patch_length, pred_len_336, marker='o', label='pred_len_336', color='green')
# plt.plot(patch_length, pred_len_720, marker='o', label='pred_len_720', color='purple')

# # 设置 x 轴刻度
# plt.xticks(patch_length, [str(x) for x in patch_length])

# # 坐标轴和标题设置
# plt.xlabel('Patch Length')
# plt.ylabel('MSE')
# plt.title('Exchange')
# plt.grid(True, linestyle='--', alpha=0.5)

# # 将图例放在右侧外部，避免遮挡折线
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # 调整布局，留出右侧空间给图例
# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()





#这是数据集ETTm1
import matplotlib.pyplot as plt

# Patch Length (横坐标)
patch_length = [8, 16, 24, 32]

# 不同预测长度对应的 MSE 数据
pred_len_96 = [0.0280, 0.0281 ,0.0281, 0.0280 ]
pred_len_192 = [0.0429, 0.0431, 0.0430 ,0.0431]
pred_len_336 = [0.0567, 0.0566, 0.0566, 0.0565]
pred_len_720 = [0.0792, 0.0790, 0.0791, 0.0789]

plt.figure(figsize=(8, 4))  # 调整画布尺寸，为图例留出更多空间

# 绘制不同预测长度对应的曲线
plt.plot(patch_length, pred_len_96, marker='o', label='pred_len_96', color='blue')
plt.plot(patch_length, pred_len_192, marker='o', label='pred_len_192', color='orange')
plt.plot(patch_length, pred_len_336, marker='o', label='pred_len_336', color='green')
plt.plot(patch_length, pred_len_720, marker='o', label='pred_len_720', color='purple')

# 设置 x 轴刻度
plt.xticks(patch_length, [str(x) for x in patch_length])

# 坐标轴和标题设置
plt.xlabel('Patch Length')
plt.ylabel('MSE')
plt.title('ETTm1')
plt.grid(True, linestyle='--', alpha=0.5)

# 将图例放在右侧外部，避免遮挡折线
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 调整布局，留出右侧空间给图例
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()