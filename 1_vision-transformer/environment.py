# def generate_environment_report():
#     """生成完整的实验环境配置报告"""
    
#     report_lines = []
    
#     # 添加标题
#     report_lines.append("实验环境配置")
#     report_lines.append("=" * 50)
    
#     # 1. 系统信息
#     report_lines.append("\n一、系统信息")
#     report_lines.append("-" * 30)
    
#     import platform
#     import psutil
    
#     report_lines.append(f"• 操作系统: {platform.system()} {platform.release()}")
#     report_lines.append(f"• 系统版本: {platform.version()}")
#     report_lines.append(f"• 处理器: {platform.processor()}")
#     report_lines.append(f"• CPU核心数: {psutil.cpu_count(logical=False)} 物理核心 / {psutil.cpu_count(logical=True)} 逻辑核心")
    
#     # 内存信息
#     memory = psutil.virtual_memory()
#     report_lines.append(f"• 内存总量: {memory.total / (1024**3):.1f} GB")
#     report_lines.append(f"• 可用内存: {memory.available / (1024**3):.1f} GB")
    
#     # 2. Python环境
#     report_lines.append("\n二、Python环境")
#     report_lines.append("-" * 30)
    
#     import sys
#     report_lines.append(f"• Python版本: {sys.version.split()[0]}")
#     report_lines.append(f"• Python路径: {sys.executable}")
    
#     # 3. 深度学习框架
#     report_lines.append("\n三、深度学习框架")
#     report_lines.append("-" * 30)
    
#     # PyTorch信息
#     try:
#         import torch
#         report_lines.append(f"• PyTorch版本: {torch.__version__}")
#         report_lines.append(f"• CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        
#         if torch.cuda.is_available():
#             report_lines.append(f"• CUDA版本: {torch.version.cuda}")
#             report_lines.append(f"• GPU数量: {torch.cuda.device_count()}")
#             for i in range(torch.cuda.device_count()):
#                 gpu_props = torch.cuda.get_device_properties(i)
#                 report_lines.append(f"• GPU {i}: {gpu_props.name}")
#                 report_lines.append(f"  - 显存: {gpu_props.total_memory / 1024**3:.1f} GB")
#                 report_lines.append(f"  - 计算能力: {gpu_props.major}.{gpu_props.minor}")
#     except ImportError:
#         report_lines.append("• PyTorch: 未安装")
    
#     # TensorFlow信息
#     try:
#         import tensorflow as tf
#         report_lines.append(f"• TensorFlow版本: {tf.__version__}")
#         gpu_devices = tf.config.list_physical_devices('GPU')
#         report_lines.append(f"• GPU设备: {len(gpu_devices)} 个")
#     except ImportError:
#         report_lines.append("• TensorFlow: 未安装")
    
#     # 4. 关键数据科学库
#     report_lines.append("\n四、数据科学库")
#     report_lines.append("-" * 30)
    
#     libraries = [
#         ('numpy', 'NumPy'),
#         ('pandas', 'Pandas'),
#         ('PIL', 'Pillow'),
#         ('matplotlib', 'Matplotlib'),
#         ('sklearn', 'Scikit-learn'),
#         ('cv2', 'OpenCV'),
#         ('torchvision', 'TorchVision'),
#         ('tensorflow_datasets', 'TF Datasets')
#     ]
    
#     for lib_key, lib_name in libraries:
#         try:
#             module = __import__(lib_key)
#             version = getattr(module, '__version__', '未知版本')
#             report_lines.append(f"• {lib_name}: {version}")
#         except ImportError:
#             report_lines.append(f"• {lib_name}: 未安装")
    
#     # 5. 数据集相关环境
#     report_lines.append("\n五、数据集处理环境")
#     report_lines.append("-" * 30)
    
#     # 检查磁盘空间
#     import shutil
#     try:
#         total, used, free = shutil.disk_usage("/")
#         report_lines.append(f"• 磁盘空间: {free // (2**30)}GB 可用 / {total // (2**30)}GB 总量")
#     except:
#         report_lines.append("• 磁盘空间: 信息获取失败")
    
#     # 6. 开发环境
#     report_lines.append("\n六、开发环境")
#     report_lines.append("-" * 30)
    
#     try:
#         import notebook
#         report_lines.append(f"• Jupyter Notebook: {notebook.__version__}")
#     except:
#         report_lines.append("• Jupyter Notebook: 未安装")
    
#     # 返回格式化报告
#     return "\n".join(report_lines)

# def save_environment_report(filename="实验环境配置.txt"):
#     """生成并保存环境配置报告"""
    
#     report = generate_environment_report()
    
#     # 保存到文件
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write(report)
    
#     # 同时打印到控制台
#     print(report)
#     print(f"\n报告已保存到: {filename}")
    
#     return report

# # 高级版本：生成更详细的表格格式
# def generate_detailed_environment_table():
#     """生成更详细的表格格式环境配置"""
    
#     table_lines = []
#     table_lines.append("实验环境详细配置")
#     table_lines.append("=" * 60)
#     table_lines.append("\n| 类别 | 项目 | 规格/版本 | 备注 |")
#     table_lines.append("|------|------|-----------|------|")
    
#     import platform
#     import psutil
#     import sys
    
#     # 系统信息
#     table_lines.append(f"| 系统 | 操作系统 | {platform.system()} {platform.release()} | - |")
#     table_lines.append(f"| 系统 | 处理器 | {platform.processor()} | - |")
#     table_lines.append(f"| 系统 | 内存 | {psutil.virtual_memory().total / (1024**3):.1f} GB | - |")
#     table_lines.append(f"| 系统 | CPU核心 | {psutil.cpu_count(logical=False)}物理/{psutil.cpu_count(logical=True)}逻辑 | - |")
    
#     # Python环境
#     table_lines.append(f"| Python | 版本 | {sys.version.split()[0]} | - |")
    
#     # 深度学习框架
#     try:
#         import torch
#         cuda_status = "可用" if torch.cuda.is_available() else "不可用"
#         table_lines.append(f"| 深度学习 | PyTorch | {torch.__version__} | - |")
#         table_lines.append(f"| 深度学习 | CUDA | {cuda_status} | - |")
        
#         if torch.cuda.is_available():
#             for i in range(torch.cuda.device_count()):
#                 gpu_props = torch.cuda.get_device_properties(i)
#                 table_lines.append(f"| 硬件 | GPU {i} | {gpu_props.name} | {gpu_props.total_memory / 1024**3:.1f}GB |")
#     except ImportError:
#         table_lines.append("| 深度学习 | PyTorch | 未安装 | - |")
    
#     # 关键库版本
#     libraries = [
#         ('numpy', '数值计算'),
#         ('pandas', '数据处理'),
#         ('PIL', '图像处理'),
#         ('matplotlib', '可视化'),
#         ('sklearn', '机器学习'),
#         ('torchvision', '计算机视觉')
#     ]
    
#     for lib_key, lib_desc in libraries:
#         try:
#             module = __import__(lib_key)
#             version = getattr(module, '__version__', '未知')
#             table_lines.append(f"| 软件库 | {lib_desc} | {version} | - |")
#         except ImportError:
#             table_lines.append(f"| 软件库 | {lib_desc} | 未安装 | - |")
    
#     return "\n".join(table_lines)

# # 使用示例
# if __name__ == "__main__":
#     print("正在生成实验环境配置报告...")
#     print("\n" + "="*60)
    
#     # 生成标准报告
#     standard_report = generate_environment_report()
#     print(standard_report)
    
#     print("\n" + "="*60)
#     print("\n详细表格版本:")
#     print("="*60)
    
#     # 生成表格版本
#     table_report = generate_detailed_environment_table()
#     print(table_report)
    
#     # 保存报告到文件
#     save_environment_report("实验环境配置报告.txt")
    
#     print("\n" + "="*60)
#     print("使用说明:")
#     print("1. 将上述输出内容直接复制到Word文档中")
#     print("2. 表格版本可以复制到Word后转换为表格")
#     print("3. 报告文件已保存为 '实验环境配置报告.txt'")

import sys
import platform
import subprocess
import os
from datetime import datetime

def check_vit_environment():
    """检查是否在vit虚拟环境中运行"""
    current_env = os.path.basename(sys.prefix)
    if current_env != 'vit':
        print(f"警告: 当前不在 'vit' 虚拟环境中，而是在 '{current_env}' 环境中")
        return False
    return True

def get_gpu_info_detailed():
    """获取详细的GPU信息"""
    gpu_info = []
    try:
        # 尝试通过nvidia-smi获取准确信息
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,memory.total,count',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.strip():
                name, memory, count = line.split(', ')
                gpu_info.append({
                    'name': name.strip(),
                    'memory_gb': int(memory) / 1024,
                    'count': int(count)
                })
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 如果nvidia-smi不可用，尝试通过PyTorch获取
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'count': 1
                    })
        except ImportError:
            pass
    
    # 如果仍然没有获取到GPU信息
    if not gpu_info:
        gpu_info.append({
            'name': '未检测到GPU',
            'memory_gb': 0,
            'count': 0
        })
    
    return gpu_info

def get_cpu_info_detailed():
    """获取详细的CPU信息"""
    try:
        cpu_info = {
            'name': platform.processor(),
            'logical_cores': os.cpu_count() or 0
        }
        
        # 在Linux系统上获取更详细的CPU信息
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if "model name" in line:
                            cpu_info['name'] = line.split(':')[1].strip()
                            break
                        if "cpu cores" in line and 'physical_cores' not in cpu_info:
                            cpu_info['physical_cores'] = int(line.split(':')[1].strip())
            except:
                pass
        
        # 如果没有获取到物理核心数，使用逻辑核心数的一半作为估计
        if 'physical_cores' not in cpu_info:
            cpu_info['physical_cores'] = cpu_info['logical_cores'] // 2
            
        return cpu_info
    except Exception as e:
        return {'name': '未知', 'physical_cores': 0, 'logical_cores': 0}

def get_memory_info_detailed():
    """获取详细的内存信息"""
    try:
        # 尝试使用psutil
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 1)
        }
    except ImportError:
        # psutil不可用，尝试通过系统命令获取内存信息
        try:
            if platform.system() == "Linux":
                result = subprocess.run(
                    ['free', '-g'], 
                    capture_output=True, text=True, check=True
                )
                lines = result.stdout.split('\n')
                if len(lines) > 1:
                    memory_line = lines[1].split()
                    if len(memory_line) > 1:
                        return {'total_gb': int(memory_line[1])}
        except:
            pass
        
        return {'total_gb': '未知'}

def get_cuda_cudnn_info():
    """获取CUDA和cuDNN版本信息"""
    cuda_info = {'cuda': '未知', 'cudnn': '未知'}
    
    # 尝试通过PyTorch获取CUDA信息
    try:
        import torch
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            cuda_info['cuda'] = torch.version.cuda
    except ImportError:
        pass
    
    # 尝试通过系统命令获取CUDA信息
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release ')[1].split(',')[0]
                    cuda_info['cuda'] = cuda_version
                    break
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # 尝试获取cuDNN版本
    try:
        # 方法1: 通过Python包
        import cudnn
        cuda_info['cudnn'] = cudnn.__version__
    except ImportError:
        try:
            # 方法2: 通过查找cudnn头文件或库文件
            if platform.system() == "Linux":
                # 查找cudnn版本头文件
                result = subprocess.run([
                    'find', '/usr', '-name', 'cudnn_version.h', '-o', '-name', 'cudnn.h'
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    # 如果找到头文件，尝试读取版本信息
                    header_file = result.stdout.split('\n')[0].strip()
                    with open(header_file, 'r') as f:
                        content = f.read()
                        # 查找版本定义
                        if '#define CUDNN_MAJOR' in content:
                            import re
                            major = re.search(r'#define CUDNN_MAJOR\s+(\d+)', content)
                            minor = re.search(r'#define CUDNN_MINOR\s+(\d+)', content)
                            patch = re.search(r'#define CUDNN_PATCHLEVEL\s+(\d+)', content)
                            
                            if major and minor:
                                version_parts = [major.group(1), minor.group(1)]
                                if patch:
                                    version_parts.append(patch.group(1))
                                cuda_info['cudnn'] = '.'.join(version_parts)
        except:
            pass
    
    return cuda_info

def get_software_versions_detailed():
    """获取详细的软件版本信息"""
    versions = {}
    
    # 操作系统
    versions['os'] = f"{platform.system()} {platform.release()}"
    
    # Python版本
    versions['python'] = sys.version.split()[0]
    
    # 深度学习框架
    frameworks = {
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'torchvision': 'TorchVision',
        'tensorboard': 'TensorBoard'
    }
    
    for pkg, name in frameworks.items():
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, '__version__', '未知')
        except ImportError:
            versions[pkg] = '未安装'
    
    # CUDA和cuDNN
    cuda_cudnn = get_cuda_cudnn_info()
    versions.update(cuda_cudnn)
    
    return versions

def get_python_packages():
    """获取Python包依赖"""
    packages = {}
    
    # 关键的数据科学和深度学习包
    key_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'PIL',
        'opencv-python', 'scikit-learn', 'scipy', 'jupyter',
        'transformers', 'datasets', 'wandb', 'mlflow'
    ]
    
    for pkg in key_packages:
        try:
            # 处理包名中的破折号
            import_name = pkg.replace('-', '_')
            module = __import__(import_name)
            packages[pkg] = getattr(module, '__version__', '未知')
        except ImportError:
            packages[pkg] = '未安装'
    
    return packages

def get_environment_management_info():
    """获取环境管理信息"""
    env_info = {}
    
    # 检查环境类型
    env_info['environment_path'] = sys.prefix
    env_info['environment_name'] = os.path.basename(sys.prefix)
    
    # 检查包管理工具
    try:
        import conda
        env_info['package_manager'] = 'Conda'
        env_info['manager_version'] = conda.__version__
        
        # 尝试导出Conda环境
        try:
            result = subprocess.run(
                ['conda', 'env', 'export', '-n', 'vit'],
                capture_output=True, text=True, check=True
            )
            env_info['environment_file'] = 'environment.yml'
            with open('environment.yml', 'w') as f:
                f.write(result.stdout)
        except:
            env_info['environment_file'] = '无法导出'
            
    except ImportError:
        env_info['package_manager'] = 'Pip'
        try:
            import pip
            env_info['manager_version'] = pip.__version__
            
            # 导出requirements.txt
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'freeze'],
                    capture_output=True, text=True, check=True
                )
                env_info['environment_file'] = 'requirements.txt'
                with open('requirements.txt', 'w') as f:
                    f.write(result.stdout)
            except:
                env_info['environment_file'] = '无法导出'
                
        except ImportError:
            env_info['package_manager'] = '未知'
            env_info['manager_version'] = '未知'
    
    return env_info

def generate_vit_environment_report():
    """生成vit虚拟环境的详细实验环境配置报告"""
    
    # 检查是否在vit环境中
    if not check_vit_environment():
        print("请在 'vit' 虚拟环境中运行此脚本")
        return None
    
    report = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report.append("实验环境配置文档")
    report.append("=" * 60)
    report.append(f"生成时间: {timestamp}")
    report.append(f"虚拟环境: vit")
    report.append("")
    
    # 1. 硬件环境
    report.append("一、硬件环境")
    report.append("-" * 40)
    
    # GPU信息
    gpus = get_gpu_info_detailed()
    if gpus and gpus[0]['name'] != '未检测到GPU':
        gpu_summary = []
        for gpu in gpus:
            if gpu['count'] > 1:
                gpu_summary.append(f"{gpu['name']} {gpu['memory_gb']}GB * {gpu['count']}")
            else:
                gpu_summary.append(f"{gpu['name']} {gpu['memory_gb']}GB")
        report.append(f"• GPU: {', '.join(gpu_summary)}")
    else:
        report.append("• GPU: 未检测到GPU设备")
    
    # CPU信息
    cpu = get_cpu_info_detailed()
    report.append(f"• CPU: {cpu['name']}, {cpu['physical_cores']}核心/{cpu['logical_cores']}线程")
    
    # 内存信息
    memory = get_memory_info_detailed()
    report.append(f"• 内存: {memory['total_gb']}GB")
    
    # 2. 软件环境
    report.append("\n二、软件环境")
    report.append("-" * 40)
    
    software = get_software_versions_detailed()
    report.append(f"• 操作系统: {software['os']}")
    report.append(f"• Python版本: {software['python']}")
    
    # 深度学习框架
    dl_frameworks = []
    if software.get('torch', '未安装') != '未安装':
        dl_frameworks.append(f"PyTorch {software['torch']}")
    if software.get('tensorflow', '未安装') != '未安装':
        dl_frameworks.append(f"TensorFlow {software['tensorflow']}")
    
    if dl_frameworks:
        report.append(f"• 深度学习框架: {', '.join(dl_frameworks)}")
    else:
        report.append("• 深度学习框架: 未检测到主要框架")
    
    # CUDA和cuDNN
    report.append(f"• CUDA版本: {software['cuda']}")
    report.append(f"• cuDNN版本: {software['cudnn']}")
    
    # 3. Python包依赖
    report.append("\n三、Python包依赖")
    report.append("-" * 40)
    
    packages = get_python_packages()
    
    # 按类别分组显示
    categories = {
        '数据处理': ['numpy', 'pandas', 'scipy'],
        '计算机视觉': ['PIL', 'opencv-python'],
        '可视化': ['matplotlib', 'seaborn'],
        '机器学习': ['scikit-learn'],
        '实验工具': ['jupyter', 'wandb', 'mlflow', 'tensorboard'],
        'NLP工具': ['transformers', 'datasets']
    }
    
    for category, pkg_list in categories.items():
        installed = []
        for pkg in pkg_list:
            if pkg in packages and packages[pkg] != '未安装':
                installed.append(f"{pkg}=={packages[pkg]}")
        
        if installed:
            report.append(f"• {category}:")
            for pkg_info in installed:
                report.append(f"  - {pkg_info}")
    
    # 4. 环境管理
    report.append("\n四、环境管理")
    report.append("-" * 40)
    
    env_mgmt = get_environment_management_info()
    report.append(f"• 环境管理工具: {env_mgmt.get('package_manager', '未知')}")
    if 'manager_version' in env_mgmt:
        report.append(f"• 工具版本: {env_mgmt['manager_version']}")
    report.append(f"• 环境路径: {env_mgmt['environment_path']}")
    
    if 'environment_file' in env_mgmt and env_mgmt['environment_file'] != '无法导出':
        report.append(f"• 环境配置文件: {env_mgmt['environment_file']} (已生成)")
    else:
        report.append("• 环境配置文件: 无法自动生成")
    
    return "\n".join(report)

def save_vit_environment_report(filename="vit实验环境配置.md"):
    """保存vit环境的实验环境配置报告"""
    
    report = generate_vit_environment_report()
    if report is None:
        return None
    
    # 保存到文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 打印到控制台
    print(report)
    print(f"\n实验环境配置文档已保存到: {filename}")
    
    return report

def export_environment_files():
    """导出环境配置文件"""
    files_created = []
    
    # 尝试导出Conda环境
    try:
        result = subprocess.run(
            ['conda', 'env', 'export', '-n', 'vit'],
            capture_output=True, text=True, check=True
        )
        with open('environment.yml', 'w') as f:
            f.write(result.stdout)
        files_created.append('environment.yml')
    except:
        pass
    
    # 尝试导出pip requirements
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True, check=True
        )
        with open('requirements.txt', 'w') as f:
            f.write(result.stdout)
        files_created.append('requirements.txt')
    except:
        pass
    
    return files_created

# 主执行函数
def main():
    """主函数"""
    print("正在生成 'vit' 虚拟环境的实验环境配置文档...")
    print("=" * 70)
    
    # 生成并保存报告
    report = save_vit_environment_report()
    
    if report is None:
        return
    
    # 导出环境文件
    print("\n" + "=" * 70)
    print("正在导出环境配置文件...")
    exported_files = export_environment_files()
    
    if exported_files:
        print(f"已导出的环境文件: {', '.join(exported_files)}")
    else:
        print("未能自动导出环境配置文件")
    
    print("\n" + "=" * 70)
    print("文档生成完成！")
    print("请将生成的 'vit实验环境配置.md' 文件内容复制到您的实验文档中。")

if __name__ == "__main__":
    main()