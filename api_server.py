import os
import asyncio
import threading
import uuid
import shutil
from contextlib import asynccontextmanager
from typing import List, Dict

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException

from feature_extractor import ImageFeatureExtractor
from iceberg_manager import IcebergManager

# --- 配置 (建议从环境变量或配置文件中读取) ---
# 模型相关配置
MODEL_NAME = os.getenv("MODEL_NAME", "mobilenetv3_large_100")
PEFT_PATH = os.getenv("PEFT_PATH", "finetuned_model")
HEAD_WEIGHTS_PATH = os.getenv("HEAD_WEIGHTS_PATH", "finetuned_model/head_weights.pth")
# Iceberg相关配置
CATALOG_NAME = os.getenv("ICEBERG_CATALOG_NAME", "default")
TABLE_IDENTIFIER = os.getenv("ICEBERG_TABLE_IDENTIFIER", "feature_db.vectors")
# API服务相关配置
TEMP_UPLOAD_DIR = "temp_uploads"
# 批处理/缓冲相关配置
BUFFER_FLUSH_INTERVAL_SECONDS = 10  # 每10秒刷新一次缓冲区
BUFFER_FLUSH_SIZE_THRESHOLD = 100   # 或者当缓冲区达到100条记录时

# --- 全局变量 ---
# 用于在内存中缓冲待写入的数据
data_buffer: List[Dict] = []
# 线程锁，用于保证对缓冲区的并发访问安全
buffer_lock = threading.Lock()
# 后台任务，用于定时刷新缓冲区
background_task = None
# 服务关闭事件
shutdown_event = asyncio.Event()

# --- 服务组件 ---
# 在应用启动时，这些组件将被初始化
extractor: ImageFeatureExtractor = None
iceberg_manager: IcebergManager = None

# --- 后台任务与生命周期管理 ---

async def flush_buffer():
    """
    获取锁，将缓冲区数据写入Iceberg，然后清空缓冲区。
    这是一个核心函数，会被后台任务和关闭事件调用。
    """
    global data_buffer
    with buffer_lock:
        if not data_buffer:
            return

        # 复制缓冲区内容以进行处理，并立即清空原始缓冲区
        # 这样可以减少锁定的时间
        data_to_flush = data_buffer.copy()
        data_buffer.clear()

    print(f"检测到 {len(data_to_flush)} 条记录在缓冲区，正在刷新到Iceberg...")
    try:
        iceberg_manager.append(data_to_flush)
    except Exception as e:
        print(f"严重错误: 刷新缓冲区到Iceberg时失败: {e}")
        # 失败处理：可以将数据重新加回缓冲区，或存入一个临时的失败队列
        # 这里为了简化，我们选择重新加回去
        with buffer_lock:
            data_buffer.extend(data_to_flush)

async def periodic_buffer_flush():
    """
    一个在后台持续运行的异步任务，
    每隔指定时间就检查并刷新一次缓冲区。
    """
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(BUFFER_FLUSH_INTERVAL_SECONDS)
            await flush_buffer()
        except asyncio.CancelledError:
            print("后台刷新任务被取消。")
            break

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用的生命周期管理器。
    在应用启动时初始化资源，在关闭时清理资源。
    """
    global extractor, iceberg_manager, background_task

    print("--- 应用启动 ---")
    # 创建临时上传目录
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

    # 初始化模型提取器
    print("正在初始化特征提取器...")
    extractor = ImageFeatureExtractor(
        model_name=MODEL_NAME,
        peft_path=PEFT_PATH,
        head_weights_path=HEAD_WEIGHTS_PATH
    )

    # 初始化Iceberg管理器
    print("正在初始化Iceberg管理器...")
    iceberg_manager = IcebergManager(
        catalog_name=CATALOG_NAME,
        table_identifier=TABLE_IDENTIFIER
    )

    # 启动后台刷新任务
    print(f"正在启动后台刷新任务，每 {BUFFER_FLUSH_INTERVAL_SECONDS} 秒执行一次...")
    background_task = asyncio.create_task(periodic_buffer_flush())

    yield

    print("--- 应用关闭 ---")
    # 触发关闭事件
    shutdown_event.set()
    # 等待后台任务结束
    if background_task:
        background_task.cancel()
        await background_task

    # 在关闭前，执行最后一次缓冲区刷新
    print("正在执行最后一次缓冲区刷新...")
    await flush_buffer()

    # 清理临时上传目录
    print("正在清理临时上传目录...")
    shutil.rmtree(TEMP_UPLOAD_DIR)
    print("清理完成。")

# --- FastAPI应用实例 ---
app = FastAPI(lifespan=lifespan)

# --- API 端点 ---
@app.post("/extract-features/", status_code=202)
async def extract_features(file: UploadFile = File(...)):
    """
    接收一张图片，提取特征向量，并将其加入处理队列。
    这是一个异步接收的端点，它会立即返回一个"Accepted"响应，
    实际的写入操作由后台任务完成。
    """
    # 检查文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传的文件不是图片格式。")

    # 为临时文件生成一个唯一的文件名
    temp_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    temp_filepath = os.path.join(TEMP_UPLOAD_DIR, temp_filename)

    try:
        # 将上传的文件保存到临时位置
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 提取特征 (这是一个CPU/GPU密集型操作)
        # 在实际高并发应用中，可以考虑用Celery等任务队列将其移出主事件循环
        features_tensor = extractor.extract(temp_filepath)

        if features_tensor is None:
            raise HTTPException(status_code=500, detail="特征提取失败，无法处理图片。")

        # 准备要存入Iceberg的数据
        image_id = os.path.splitext(file.filename)[0]
        # 将torch.Tensor转换为Python列表
        features_list = features_tensor.cpu().numpy().flatten().tolist()

        record = {"id": image_id, "vec": features_list}

        # 将记录添加到缓冲区
        with buffer_lock:
            data_buffer.append(record)
            # 检查是否达到阈值需要立即刷新 (可选，但可以提高响应性)
            if len(data_buffer) >= BUFFER_FLUSH_SIZE_THRESHOLD:
                # 注意：这里我们不直接调用flush_buffer，
                # 而是让后台的循环来处理，避免阻塞当前请求。
                # 如果需要更强的实时性，可以考虑在这里触发一个事件。
                pass

    except Exception as e:
        # 更详细的错误日志
        print(f"处理文件 {file.filename} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理图片时发生内部错误: {str(e)}")
    finally:
        # 确保临时文件被删除
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    return {"message": "图片已接收，特征提取请求已加入处理队列。", "image_id": image_id}

# --- 用于本地运行的入口 ---
if __name__ == "__main__":
    import uvicorn
    print("--- 启动FastAPI服务 ---")
    print("访问 http://127.0.0.1:8000/docs 查看API文档。")
    print("要运行此服务，请确保已设置好Iceberg和模型相关的环境变量。")
    uvicorn.run(app, host="0.0.0.0", port=8000)
