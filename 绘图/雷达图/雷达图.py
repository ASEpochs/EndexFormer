import numpy as np
import matplotlib.pyplot as plt

# 定义分类名称
labels = np.array(["ETTh1", "ETTm1", "ECL", "Weather", "Traffic", "Exchange"])

# 定义不同方法的实验数据（数值越小越好）
original_data = {
    # labels = np.array(["ETTh1", "ETTm1", "ECL", "Weather", "Traffic", "Exchange"])
    "Endexformer": np.array([0.0762, 0.0517, 0.3487, 0.0016, 0.1637, 0.4474]),
    "TimeXer": np.array([0.0768, 0.0519, 0.3504, 0.0017, 0.1555, 0.5080]),
    "DLinear": np.array([0.1233, 0.0654, 0.3892, 0.0062, 0.2778, 0.3702]),
    "iTransformer": np.array([0.0767, 0.0524, 0.3835, 0.0017, 0.1837, 0.4682]),
    "TimesNet": np.array([0.0774, 0.0543, 0.4039, 0.0018, 0.1837, 0.4682]),
    "Crossformer": np.array([0.3154, 0.2126, 0.7850, 0.0043, 0.1773, 1.1992]),
}

# 找到每个数据集（每列）的最小值
min_vals = np.min(list(original_data.values()), axis=0)

# 归一化处理：让每列的最小值变成 1，其他值变成 min_val / value，确保最小值在外层
normalized_data = {
    method: min_vals / values for method, values in original_data.items()
}

# 颜色和线型
colors = {
    "Endexformer": "red",
    "TimeXer": "brown",
    "DLinear": "blue",
    "iTransformer": "purple",
    "TimesNet": "orange",
    "Crossformer": "green",
}
linestyles = {
    "Endexformer": "dashed",
    "TimeXer": "solid",
    "DLinear": "solid",
    "iTransformer": "solid",
    "TimesNet": "solid",
    "Crossformer": "solid",
}

# 计算雷达图的角度
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合多边形

# 绘制雷达图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for method, values in normalized_data.items():
    values = np.append(values, values[0])  # 闭合曲线
    ax.plot(angles, values, color=colors[method], linestyle=linestyles[method], linewidth=2, label=method)
    ax.fill(angles, values, color=colors[method], alpha=0.1)

# 添加标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')

# 优化图例
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)

# 显示图像
plt.show()