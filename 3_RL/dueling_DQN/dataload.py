import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    """带进度条的下载函数"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(destination))
    
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("错误：下载过程中出现异常。")

def unzip_file(zip_path, extract_to):
    """解压并删除压缩包"""
    print(f"正在解压 {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path) # 节省空间
    print(f"解压完成并已删除压缩包。")

def main():
    # 设定存储目录
    base_dir = "./data/coco"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"创建目录: {base_dir}")

    # COCO 2017 官方下载地址
    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip"
    }

    # 执行下载与解压
    for key, url in urls.items():
        filename = os.path.basename(url)
        filepath = os.path.join(base_dir, filename)
        
        # 1. 下载
        if not os.path.exists(filepath.replace('.zip', '')): # 简单检查是否已存在解压后的目录
            print(f"\n准备下载 {key}...")
            if not os.path.exists(filepath):
                download_file(url, filepath)
            else:
                print(f"文件 {filename} 已存在，跳过下载。")
            
            # 2. 解压
            # 标注文件解压后会自动生成 annotations 文件夹
            # 图片文件解压后会自动生成 train2017 和 val2017 文件夹
            unzip_file(filepath, base_dir)
        else:
            print(f"\n{key} 数据已存在，跳过。")

    print("\n所有数据准备就绪！")
    print(f"目录结构：\n{base_dir}/ \n  ├── annotations/ \n  ├── train2017/ \n  └── val2017/")

if __name__ == "__main__":
    main()