import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import cv2
import tqdm
import argparse
import random
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================
# 肺窗设置 (Lung Window): 适合观察肺结节
WINDOW_CENTER = -600
WINDOW_WIDTH = 1500

# 图片保存设置
OUTPUT_SIZE = (224, 224)   # 最终保存的图片大小
CROP_MODE = 'roi_center'   # 'roi_center': 正样本以结节为中心，负样本以该病人结节平均位置为中心
CROP_SIZE = 128            # 裁剪区域大小 (128x128) -> Resize到 OUTPUT_SIZE
NEGATIVE_RATIO = 1.0       # 正负样本比例，1.0 表示 1:1，即有多少正样本就采多少负样本

# Excel中列名的映射
COL_MAPPING = {
    'id': '影像号码',
    'volume': '结节体积',
    'mass': '结节质量',
    'dt_vol': '体积倍增时间',
    'dt_mass': '质量倍增时间',
    'surface_area': '结节表面积',
    'mean_ct': 'CT平均值',
    'spicular': '毛刺征',
    'lobulated': '分叶征',
}

# ================= 工具函数 =================

def apply_lung_window(img_data, center, width):
    """应用肺窗并归一化到 0-255"""
    min_value = center - width / 2
    max_value = center + width / 2
    img_data = np.clip(img_data, min_value, max_value)
    img_data = (img_data - min_value) / (max_value - min_value)
    img_data = (img_data * 255).astype(np.uint8)
    return img_data

def generate_description(row, label):
    """
    生成文本描述
    label=1: 包含结节特征
    label=0: 描述为正常区域，不包含结节特征
    """
    if label == 0:
        return "No nodule detected in this slice. Normal lung tissue structure."

    desc_parts = []
    # 仅当 label=1 时才把 Excel 里的结节特征写进去
    try:
        if pd.notna(row.get('结节体积')):
            desc_parts.append(f"The nodule volume is {row['结节体积']} mm3.")
        if pd.notna(row.get('结节质量')):
            desc_parts.append(f"The nodule mass is {row['结节质量']} mg.")
        if pd.notna(row.get('CT平均值')):
            desc_parts.append(f"The mean CT value is {row['CT平均值']} HU.")
        if pd.notna(row.get('体积倍增时间')):
            desc_parts.append(f"Volume doubling time is {row['体积倍增时间']} days.")
        if pd.notna(row.get('球形度')):
             desc_parts.append(f"Sphericity is {row['球形度']}.")
    except Exception:
        pass

    base_text = "Positive for lung nodule. "
    full_text = base_text + " ".join(desc_parts)
    return full_text[:500]

def process_single_case(case_path, output_img_root, row_data):
    """处理单个病例：生成正样本和负样本"""
    # 1. 寻找 volume 和 mask 文件
    vol_files = glob.glob(os.path.join(case_path, "**", "*.nii.gz"), recursive=True)
    vol_path = None
    mask_path = None
    
    for f in vol_files:
        fname = os.path.basename(f)
        if "mask_7LMS" in fname:
            mask_path = f
        elif "vol" in fname.lower() and "mask" not in fname.lower():
            vol_path = f
            
    if not vol_path or not mask_path:
        return []

    # 2. 加载数据
    try:
        vol_nii = nib.load(vol_path)
        mask_nii = nib.load(mask_path)
        vol_nii = nib.as_closest_canonical(vol_nii)
        mask_nii = nib.as_closest_canonical(mask_nii)
        vol_data = vol_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
    except Exception as e:
        print(f"Error loading {case_path}: {e}")
        return []

    # 3. 确定正样本层 (Mask > 0)
    # mask_data shape: (H, W, D)
    pos_z_indices = np.where(np.sum(mask_data, axis=(0, 1)) > 0)[0]
    if len(pos_z_indices) == 0:
        return []

    # 计算该病人结节的平均中心位置 (用于负样本裁剪，保证视野一致)
    # 简单的重心法
    all_pos_rows, all_pos_cols, _ = np.where(mask_data > 0)
    if len(all_pos_rows) > 0:
        avg_center_y = int(np.mean(all_pos_rows))
        avg_center_x = int(np.mean(all_pos_cols))
    else:
        avg_center_y, avg_center_x = vol_data.shape[0]//2, vol_data.shape[1]//2

    # 4. 确定负样本层 (Mask == 0)
    # 策略：在有 mask 的层上下扩展一定范围，或者是全肺范围。
    # 为了避免切到肚子或脖子，我们可以简单取 min_z - margin 到 max_z + margin
    all_z = np.arange(vol_data.shape[2])
    # 排除正样本层
    neg_z_candidates = np.setdiff1d(all_z, pos_z_indices)
    
    # 简单的范围筛选：只取正样本范围前后 20 层内的负样本 (Hard Negatives)
    # 如果想更泛化，可以取全范围
    if len(pos_z_indices) > 0:
        min_pos, max_pos = np.min(pos_z_indices), np.max(pos_z_indices)
        valid_min = max(0, min_pos - 30)
        valid_max = min(vol_data.shape[2], max_pos + 30)
        neg_z_candidates = neg_z_candidates[(neg_z_candidates >= valid_min) & (neg_z_candidates < valid_max)]

    # 随机采样负样本
    num_neg = int(len(pos_z_indices) * NEGATIVE_RATIO)
    if len(neg_z_candidates) > num_neg:
        neg_z_indices = np.random.choice(neg_z_candidates, size=num_neg, replace=False)
    else:
        neg_z_indices = neg_z_candidates

    generated_samples = []
    patient_id = str(row_data['影像号码'])
    save_dir = os.path.join(output_img_root, patient_id)
    os.makedirs(save_dir, exist_ok=True)

    # 5. 定义内部处理函数
    def save_slice(z_idx, label, center_y, center_x):
        slice_vol = vol_data[:, :, z_idx]
        img_uint8 = apply_lung_window(slice_vol, WINDOW_CENTER, WINDOW_WIDTH)
        
        # 裁剪逻辑
        if CROP_MODE == 'roi_center':
            y1 = max(0, center_y - CROP_SIZE // 2)
            y2 = min(img_uint8.shape[0], center_y + CROP_SIZE // 2)
            x1 = max(0, center_x - CROP_SIZE // 2)
            x2 = min(img_uint8.shape[1], center_x + CROP_SIZE // 2)
            crop_img = img_uint8[y1:y2, x1:x2]
            final_img = cv2.resize(crop_img, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        else:
            final_img = cv2.resize(img_uint8, OUTPUT_SIZE)

        img_filename = f"{patient_id}_z{z_idx}_L{label}.png"
        rel_path = os.path.join(patient_id, img_filename)
        abs_save_path = os.path.join(save_dir, img_filename)
        cv2.imwrite(abs_save_path, final_img)
        
        sample = {
            'image_path': rel_path,
            'id': f"{patient_id}_{z_idx}",
            'description': generate_description(row_data, label),
            'label_index': label,  # 0 或 1
            'original_id': patient_id,
            'slice_z': z_idx
        }
        return sample

    # 6. 处理正样本
    for z in pos_z_indices:
        # 正样本：每一层可能有微小的中心偏移，计算当前层的中心更准
        rows, cols = np.where(mask_data[:, :, z] > 0)
        if len(rows) > 0:
            cy, cx = int(np.mean(rows)), int(np.mean(cols))
        else:
            cy, cx = avg_center_y, avg_center_x # Fallback
        
        generated_samples.append(save_slice(z, 1, cy, cx))

    # 7. 处理负样本
    for z in neg_z_indices:
        # 负样本：没有结节，强制使用该病人的平均结节位置进行裁剪
        # 这样模型看到的是"原本应该有结节但现在没有"的区域
        generated_samples.append(save_slice(z, 0, avg_center_y, avg_center_x))
        
    return generated_samples

# ================= 主流程 =================

def main(excel_path, data_root, output_root):
    # 读取 Excel
    print(f"Loading Excel from {excel_path}...")
    try:
        df_meta = pd.read_excel(excel_path)
    except:
        df_meta = pd.read_csv(excel_path)
    print(f"Found {len(df_meta)} records in Excel.")
    
    output_img_root = os.path.join(output_root, "images")
    os.makedirs(output_img_root, exist_ok=True)
    
    all_samples = []
    
    # 遍历病例文件夹
    case_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    print(f"Found {len(case_folders)} folders in data root.")
    
    meta_dict = {}
    for idx, row in df_meta.iterrows():
        pid = str(row['影像号码']).strip()
        meta_dict[pid] = row

    success_count = 0
    
    for folder_name in tqdm.tqdm(case_folders):
        folder_path = os.path.join(data_root, folder_name)
        matched_pid = None
        for pid in meta_dict.keys():
            if pid in folder_name:
                matched_pid = pid
                break
        
        if not matched_pid:
            continue
            
        row = meta_dict[matched_pid]
        samples = process_single_case(folder_path, output_img_root, row)
        
        if samples:
            all_samples.extend(samples)
            success_count += 1
            
    print(f"Processed {success_count} patients, generated {len(all_samples)} slices.")
    
    out_df = pd.DataFrame(all_samples)
    if out_df.empty:
        print("Error: No samples generated.")
        return

    # 数据集划分
    unique_pids = out_df['original_id'].unique()
    train_ids, val_test_ids = train_test_split(unique_pids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=42)
    
    def get_split(pid):
        if pid in train_ids: return 'train'
        elif pid in val_ids: return 'val'
        else: return 'test'
        
    out_df['split'] = out_df['original_id'].apply(get_split)
    
    csv_save_path = os.path.join(output_root, "dataset.csv")
    out_df.to_csv(csv_save_path, index=False)
    print(f"Dataset CSV saved to {csv_save_path}")
    print("Class distribution:\n", out_df['label_index'].value_counts())
    print("Split distribution:\n", out_df['split'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.excel, args.data, args.output)




"""
    Example usage:
        python preprocess_ct_to_2d.py \
    --excel "/root/epfs/downloads/4823513924_月云飞yyf/已核对完的ROI数据/你的表格文件名.xlsx" \
    --data "/root/epfs/downloads/4823513924_月云飞yyf/已核对完的ROI数据/663volume20251111" \
    --output "/root/epfs/hhj/ZongShuJu/AiZheng/feixianai/2DCT"
"""