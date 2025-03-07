import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置科研绘图风格
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2

# 定义分类名称
labels = np.array(["ETTh1", "ETTm1", "ECL", "Weather", "Traffic", "Exchange"])

# 定义原始实验数据（数值越小越好）
original_data = {
    "Endexformer": np.array([0.0762, 0.0517, 0.3487, 0.0016, 0.1637, 0.4474]),
    "TimeXer": np.array([0.0768, 0.0519, 0.3504, 0.0017, 0.1555, 0.5080]),
    "DLinear": np.array([0.1233, 0.0654, 0.3892, 0.0062, 0.2778, 0.3702]),
    "iTransformer": np.array([0.0767, 0.0524, 0.3835, 0.0017, 0.1837, 0.4682]),
    "TimesNet": np.array([0.0774, 0.0543, 0.4039, 0.0018, 0.1837, 0.4682]),
    "Crossformer": np.array([0.3154, 0.2126, 0.7850, 0.0043, 0.1773, 1.1992]),
}

# 计算归一化数据（保持原始值比较关系）
min_vals = np.min(list(original_data.values()), axis=0)
normalized_data = {method: min_vals/values for method, values in original_data.items()}

# 可视化设置
colors = {
    "Endexformer": "#E41A1C",
    "TimeXer": "#377EB8",
    "DLinear": "#4DAF4A",
    "iTransformer": "#984EA3",
    "TimesNet": "#FF7F00",
    "Crossformer": "#A65628"
}
line_styles = {
    "Endexformer": (0, (5, 5)),
    "TimeXer": "solid",
    "DLinear": "solid",
    "iTransformer": "solid",
    "TimesNet": "solid",
    "Crossformer": "solid"
}

# 创建极坐标系统
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# 计算角度并闭合图形
num_vars = len(labels)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 绘制每个方法
for idx, method in enumerate(original_data):
    # 准备数据（闭合处理）
    values = np.append(normalized_data[method], normalized_data[method][0])
    
    # 绘制填充区域，alpha值控制透明度
    ax.fill(angles, values, color=colors[method], alpha=0.2)
    
    # 绘制主线条，并使用较大的数据点和白色填充以便区分
    ax.plot(angles, values, color=colors[method], 
            linewidth=2, linestyle=line_styles[method],
            label=method, marker='o', markersize=6,
            markerfacecolor='white', markeredgecolor=colors[method])
    
    # 添加数据标签，稍微偏移不同方法的标签以防重叠
    for i, (angle, norm_val, orig_val) in enumerate(zip(angles[:-1], normalized_data[method], original_data[method])):
        # 根据方法索引设置偏移量，减小同一位置标签的重叠
        offset = 0.05 + idx * 0.005
        text_radius = norm_val + 0.05 + offset
        
        # 转换角度为度数，并调整旋转方向以保持文本可读性
        rotation_angle = np.degrees(angle)
        if angle > np.pi/2 and angle < 3*np.pi/2:
            rotation_angle += 180
        
        ax.text(angle, text_radius, f"{orig_val:.4f}", 
                color=colors[method], fontsize=8,
                ha='center', va='center', 
                rotation=rotation_angle,
                rotation_mode='anchor')

# 坐标轴调整
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_ylim(0, 1.4)  # 扩展径向坐标范围

# 设置网格线和轴标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.tick_params(axis='both', which='major', pad=15)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

# 移除径向刻度标签
ax.set_yticklabels([])

# 添加图例
legend = ax.legend(loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15),
                   ncol=3, 
                   frameon=True,
                   fontsize=11,
                   handlelength=2,
                   columnspacing=1.5)
legend.get_frame().set_linewidth(1.2)
legend.get_frame().set_edgecolor("black")

# 添加标题
plt.title("Normalized Performance Comparison Across Datasets", 
          pad=35, fontsize=14, fontweight='bold')

# 优化布局并显示
plt.tight_layout()
plt.show()
