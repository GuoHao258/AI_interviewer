import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tempfile
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

# 导入CosyVoiceWithAzure类
from cosyvoice_with_azure import CosyVoiceWithAzure

app = FastAPI()

# 设置跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 临时目录，用于存储上传的音频文件
temp_dir = tempfile.mkdtemp()
print(f"临时目录: {temp_dir}")
print(f"[TEMP DIRECTORY] Created temporary directory: {os.path.abspath(temp_dir)}")

def generate_data(model_output):
    """生成音频数据流"""
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

@app.get("/")
async def root():
    """API根路径，返回欢迎信息"""
    return {
        "message": "CosyVoice with Azure API",
        "version": "1.0",
        "endpoints": [
            "/zero_shot",
            "/instruct",
            "/cross_lingual"
        ]
    }

@app.get("/favicon.ico")
async def favicon():
    """处理favicon请求"""
    return {"status": "no favicon"}

@app.get("/zero_shot")
@app.post("/zero_shot")
async def zero_shot(target_text: str = Form(...), prompt_audio: UploadFile = File(...)):
    """Zero-shot语音合成API"""
    # 保存上传的音频文件
    prompt_path = os.path.join(temp_dir, f"prompt_{prompt_audio.filename}")
    with open(prompt_path, "wb") as f:
        f.write(await prompt_audio.read())
    
    # 使用CosyVoiceWithAzure进行zero-shot合成
    output_paths = cosyvoice.zero_shot_with_auto_prompt(target_text, prompt_path)
    
    # 如果合成成功，返回第一个合成结果
    if output_paths:
        # 读取合成的音频文件
        with open(output_paths[0], "rb") as f:
            audio_data = f.read()
        
        # 返回音频文件
        return StreamingResponse(iter([audio_data]), media_type="audio/wav")
    else:
        return {"error": "语音合成失败"}

@app.get("/instruct")
@app.post("/instruct")
async def instruct(target_text: str = Form(...), prompt_audio: UploadFile = File(...), instruct_text: str = Form(...)):
    """Instruct语音合成API"""
    # 保存上传的音频文件
    prompt_path = os.path.join(temp_dir, f"prompt_{prompt_audio.filename}")
    with open(prompt_path, "wb") as f:
        f.write(await prompt_audio.read())
    
    # 使用CosyVoiceWithAzure进行instruct合成
    output_paths = cosyvoice.instruct_with_auto_prompt(target_text, prompt_path, instruct_text)
    
    # 如果合成成功，返回第一个合成结果
    if output_paths:
        # 读取合成的音频文件
        with open(output_paths[0], "rb") as f:
            audio_data = f.read()
        
        # 返回音频文件
        return StreamingResponse(iter([audio_data]), media_type="audio/wav")
    else:
        return {"error": "语音合成失败"}

@app.get("/cross_lingual")
@app.post("/cross_lingual")
async def cross_lingual(target_text: str = Form(...), prompt_audio: UploadFile = File(...)):
    """Cross-lingual语音合成API"""
    # 保存上传的音频文件
    prompt_path = os.path.join(temp_dir, f"prompt_{prompt_audio.filename}")
    with open(prompt_path, "wb") as f:
        f.write(await prompt_audio.read())
    
    # 使用CosyVoiceWithAzure进行cross-lingual合成
    output_paths = cosyvoice.cross_lingual_with_auto_prompt(target_text, prompt_path)
    
    # 如果合成成功，返回第一个合成结果
    if output_paths:
        # 读取合成的音频文件
        with open(output_paths[0], "rb") as f:
            audio_data = f.read()
        
        # 返回音频文件
        return StreamingResponse(iter([audio_data]), media_type="audio/wav")
    else:
        return {"error": "语音合成失败"}

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CosyVoice with Azure FastAPI Server")
    parser.add_argument("--port", type=int, default=50000, help="服务器端口")
    parser.add_argument("--azure_key", default="2e031a64874b4625b4d50b58f9006bab", help="Azure语音服务的密钥")
    parser.add_argument("--azure_region", default="eastasia", help="Azure语音服务的区域")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice2-0.5B", help="CosyVoice模型路径")
    parser.add_argument("--language", default="auto", help="语言代码 (auto: 自动检测)")
    parser.add_argument("--output_dir", default="outputs/api", help="输出目录")
    parser.add_argument("--fp16", action="store_true", help="使用半精度浮点数(FP16)加速")
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"[OUTPUT DIRECTORY] Created output directory: {os.path.abspath(args.output_dir)}")
    
    # 初始化CosyVoiceWithAzure
    print("正在初始化CosyVoiceWithAzure...")
    try:
        cosyvoice = CosyVoiceWithAzure(
            azure_key=args.azure_key,
            azure_region=args.azure_region,
            tts_model_path=args.model_dir,
            language=args.language,
            output_dir=args.output_dir,
            use_fp16=args.fp16
        )
        print("CosyVoiceWithAzure初始化成功！")
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        sys.exit(1)
    
    # 启动FastAPI服务器
    print(f"启动FastAPI服务器，端口: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port) 