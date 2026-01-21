import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import pickle

# 读取原始数据
raw_data = pd.read_csv("4.0_PAH_Low.csv")
smiles = np.array(raw_data["smiles"])

# 转换序列数据为3152×128的二维数组
sequences = np.zeros((3152, 128))
for i in range(128):
    current_col = raw_data[str(i)]
    for x in range(3152):
        sequences[x][i] = current_col[x]


def not_organic(sm):
    """判断分子是否为无机分子（无C-H键）"""
    # 从SMILES创建分子对象
    molecule = Chem.MolFromSmiles(sm)
    if molecule is None:  # 处理无效SMILES
        return True

    # 为分子添加氢原子
    molecule = Chem.AddHs(molecule)

    # 检查是否存在C-H键
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 6:  # 碳原子
            for neb in atom.GetNeighbors():
                if neb.GetAtomicNum() == 1:  # 氢原子
                    return False  # 存在C-H键，是有机分子
    return True  # 无C-H键，不是有机分子


def has_charged_center(sm):
    """判断分子是否带有电荷中心"""
    return "+" in sm or "-" in sm


def too_big(sm):
    """判断分子中碳原子数量是否在50到100之间（包含边界）"""
    # 从SMILES创建分子对象
    molecule = Chem.MolFromSmiles(sm)
    if molecule is None:  # 处理无效SMILES
        return True  # 视为不符合条件

    # 统计碳原子数量（原子序数为6）
    carbon_count = sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6)

    # 判断碳原子数量是否在50到100之间
    return not (50 <= carbon_count <= 100)


# 筛选符合条件的分子
good_indices = []
num_charged, num_too_big, num_not_organic = 0, 0, 0

for sm_idx in tqdm(range(len(smiles)), desc="筛选分子"):
    sm = smiles[sm_idx]

    # 检查是否为无机分子
    if not_organic(sm):
        num_not_organic += 1
        continue

    # 检查是否带有电荷
    if has_charged_center(sm):
        num_charged += 1
        continue

    # 检查碳原子数量是否在50-100范围外
    if too_big(sm):
        num_too_big += 1
        continue

    # 所有条件都满足，保留该分子
    good_indices.append(sm_idx)

# 输出筛选结果统计
print(f"初始样本数: {len(smiles)}, 筛选后样本数: {len(good_indices)}")
print(f"被排除的无机分子数: {num_not_organic}")
print(f"被排除的带电分子数: {num_charged}")
print(f"被排除的碳数不在50-100范围的分子数: {num_too_big}")

# 保存筛选后的数据
data = {
    "smiles": smiles[good_indices],
    "sequences": sequences[good_indices]
}

with open("4.0_PAH_Low_DATA.pickle", "wb+") as f:
    pickle.dump(data, f)

print("筛选后的数据已保存为pickle文件")

