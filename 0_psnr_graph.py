import cv2, os, glob, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ==== 設定部分ここから ====
dir1 = "exp_src/"
dir2 = "48bit_alpha0.1/"
# ==== 設定部分ここまで ====

def calculate_psnr(image_path1, image_path2):
    #画像の読み込み
    original  = cv2.imread(image_path1) #元画像
    distorted = cv2.imread(image_path2)   #圧縮した画像

    #画素値の読み込み
    pixel_value_Ori = original.flatten().astype(float)
    pixel_value_Dis = distorted.flatten().astype(float)

    #画素情報の取得
    imageHeight, imageWidth, BPP = original.shape

    #画素数
    N = imageHeight * imageWidth 

    #1画素あたりRGB3つの情報がある. 
    addr = N * BPP

    #RGB画素値の差の2乗の総和
    sumR=0
    sumG=0
    sumB=0

    #差の2乗の総和を計算
    for i in range(addr):
        if(i%3==0):
            sumB += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 ) 
        elif(i%3==1):
            sumG += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 ) 
        else:
            sumR += pow ( (pixel_value_Ori[i]-pixel_value_Dis[i]), 2 ) 

    #PSNRを求める
    MSE =(sumR + sumG + sumB) / (3 * N )
    PSNR = 10 * math.log(255*255/MSE,10)
    # print('PSNR',PSNR)

    
    return PSNR

valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

def get_base_name(filename):
    return os.path.splitext(filename)[0]

files1 = [f for f in os.listdir(dir1) if os.path.splitext(f)[1].lower() in valid_extensions]
files2 = [f for f in os.listdir(dir2) if os.path.splitext(f)[1].lower() in valid_extensions]

base_dict_1 = {get_base_name(f): f for f in files1}
base_dict_2 = {get_base_name(f): f for f in files2}

# ソートを大文字小文字を区別しない形に修正
common_bases = sorted(set(base_dict_1.keys()) & set(base_dict_2.keys()), key=str.lower)

psnr_values = []
images_for_axis = []

for base in common_bases:
    img_path_1 = os.path.join(dir1, base_dict_1[base])
    img_path_2 = os.path.join(dir2, base_dict_2[base])

    # 画像読み込み
    img1 = np.array(Image.open(img_path_1).convert("RGB"))
    img2 = np.array(Image.open(img_path_2).convert("RGB"))

    # 画像サイズ揃え
    if img1.shape != img2.shape:
        img2 = np.array(Image.open(img_path_2).convert("RGB").resize((img1.shape[1], img1.shape[0])))

    # PSNR計算
    psnr_value = calculate_psnr(img_path_1, img_path_2)
    psnr_values.append(psnr_value)

    # X軸用に表示する画像(フォルダ1側の画像を使用)
    axis_img = Image.open(img_path_1).convert("RGB")
    images_for_axis.append(axis_img)

# 全画像の縦サイズを揃える(高さ固定)
desired_height = 50
resized_images_for_axis = []
for img in images_for_axis:
    aspect_ratio = img.width / img.height
    new_width = int(desired_height * aspect_ratio)
    img_resized = img.resize((new_width, desired_height))
    resized_images_for_axis.append(img_resized)

images_for_axis = resized_images_for_axis

# グラフ作成
fig, ax = plt.subplots(figsize=(len(psnr_values)*2, 6))

x_positions = np.arange(len(psnr_values))
ax.bar(x_positions, psnr_values, color='skyblue')

# 棒グラフ上部にpsnr値を表示
for i, val in enumerate(psnr_values):
    ax.text(i, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=12)

# X軸目盛りは空白にする
ax.set_xticks(x_positions)
ax.set_xticklabels(["" for _ in x_positions])

ax.set_ylabel("PSNR")

# グリッドとスパインを消す
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Y=0の位置にX軸線を描画
ax.axhline(y=0, color='black', linewidth=0.5)

# Y軸範囲設定：画像が表示されるよう、わずかに負の方向へ拡大
min_y = -0.1
max_y = max(psnr_values) + 0.1 if psnr_values else 1.0
ax.set_ylim(min_y, max_y)

# 画像をX軸とぴったりくっつけるように配置
# 上端がy=0になるように配置
for i, img in enumerate(images_for_axis):
    imagebox = OffsetImage(img, zoom=1.0)
    ab = AnnotationBbox(imagebox, (i, 0),
                        frameon=False,
                        box_alignment=(0.5, 1.0))
    ax.add_artist(ab)

plt.tight_layout(pad=2.0)
# plt.show()

# 画像を保存
output_path = f"{dir2.replace('/', '')}_psnr.png"
plt.savefig(output_path)
print(f"saved: {output_path}")

