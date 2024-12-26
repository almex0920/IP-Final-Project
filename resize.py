import os
import cv2

def resize_image(input_path, output_path, new_width, new_height, apply_binarization=False):
    # 讀取圖片
    img = cv2.imread(input_path)
    
    # 如果讀取失敗，顯示錯誤訊息
    if img is None:
        print(f"Failed to load image {input_path}")
        return

    # 調整大小
    resized_img = cv2.resize(img, (new_width, new_height))

    # 如果需要二值化處理
    if apply_binarization:
        # 轉換為灰度圖像
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        # 設定閾值進行二值化
        _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        
        # 保存二值化後的圖片
        cv2.imwrite(output_path, binary_img)
        print(f"Image has been resized and binarized, saved to {output_path}")
    else:
        # 如果不進行二值化，直接保存調整大小後的圖片
        cv2.imwrite(output_path, resized_img)
        print(f"Image has been resized and saved to {output_path}")

def resize_and_process_images_in_folder(input_folder, output_folder, new_width, new_height, apply_binarization=False):
    # 如果輸出資料夾不存在，則創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 讀取輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # 只處理圖片檔案
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path = os.path.join(output_folder, filename)
            # 調整大小並選擇是否進行二值化處理
            resize_image(input_path, output_path, new_width, new_height, apply_binarization)

# 範例使用
input_folder = 'training_dataset/image'  # 輸入資料夾
output_folder = 'training_dataset/res_imgs'  # 輸出資料夾
new_width = 256  # 新的寬度
new_height = 256  # 新的高度

# 設定是否需要二值化處理，True 表示需要二值化，False 表示不需要
apply_binarization = False  # 可以改為 False 來跳過二值化

resize_and_process_images_in_folder(input_folder, output_folder, new_width, new_height, apply_binarization)