import os
import sys
import time
import gradio as gr
import tempfile
import torchaudio
import torch
import threading
import argparse

# 导入我们自定义的CosyVoiceWithAzure集成类
from cosyvoice_with_azure import CosyVoiceWithAzure

class CosyVoiceWithAzureWebUI:
    """
    集成Azure语音识别的CosyVoice的Web界面类
    
    该类提供了一个基于Gradio的Web界面，使用户可以通过浏览器
    使用集成了Azure语音识别功能的CosyVoice语音合成系统。
    支持Zero-shot、Instruct和Cross-lingual三种语音合成模式。
    """
    
    def __init__(self):
        """
        初始化Web界面
        
        创建临时目录用于存储合成的音频文件，并初始化CosyVoice实例为None，
        等待用户在界面上提供配置参数后再进行初始化。
        """
        # CosyVoice实例初始为None，延迟初始化
        self.cosyvoice = None
        # 创建临时目录用于存储合成的音频文件
        self.temp_dir = tempfile.mkdtemp()
        print(f"临时目录: {self.temp_dir}")
        # 初始化标志，用于跟踪模型是否正在加载
        self.is_initializing = False
        # 默认参数
        self.default_azure_key = "2e031a64874b4625b4d50b58f9006bab"
        self.default_azure_region = "eastasia"
        self.default_tts_model_path = "pretrained_models/CosyVoice2-0.5B"
        self.default_language = "auto"
    
    def preload_model(self):
        """
        预加载模型
        
        在后台线程中预加载模型，这样用户在界面加载完成后就可以立即使用，
        而不需要等待第一次请求时的模型加载。
        """
        if not self.is_initializing and self.cosyvoice is None:
            self.is_initializing = True
            print("正在后台预加载CosyVoice模型...")
            
            # 在后台线程中初始化模型
            def init_model():
                try:
                    self.cosyvoice = CosyVoiceWithAzure(
                        azure_key=self.default_azure_key,
                        azure_region=self.default_azure_region,
                        tts_model_path=self.default_tts_model_path,
                        language=self.default_language,
                        output_dir=self.temp_dir,
                        optimize_for_first_run=True,
                        use_fp16=False
                    )
                    print("模型预加载完成！系统已准备就绪。")
                except Exception as e:
                    print(f"模型预加载失败: {str(e)}")
                    print("将在首次使用时再次尝试加载模型。")
                finally:
                    self.is_initializing = False
            
            # 启动后台线程
            threading.Thread(target=init_model, daemon=True).start()
    
    def ensure_initialized(self, azure_key, azure_region, tts_model_path, language):
        """
        确保CosyVoice已初始化
        
        如果CosyVoice实例尚未初始化，则使用提供的参数进行初始化。
        这种延迟初始化的方式可以让用户在界面上配置参数后再初始化模型，
        避免在启动时就加载大型模型导致启动缓慢。
        
        参数:
            azure_key (str): Azure语音服务的密钥
            azure_region (str): Azure语音服务的区域
            tts_model_path (str): CosyVoice模型路径
            language (str): 语言代码
            
        返回:
            bool: 初始化是否成功
        """
        # 如果CosyVoice实例尚未初始化且不在初始化过程中
        if self.cosyvoice is None and not self.is_initializing:
            self.is_initializing = True
            try:
                print("正在初始化CosyVoice...")
                # 创建CosyVoiceWithAzure实例
                self.cosyvoice = CosyVoiceWithAzure(
                    azure_key=azure_key,
                    azure_region=azure_region,
                    tts_model_path=tts_model_path,
                    language=language,
                    output_dir=self.temp_dir,
                    optimize_for_first_run=True,
                    use_fp16=False
                )
                print("初始化成功！")
                return True
            except Exception as e:
                # 如果初始化过程中发生异常，打印错误信息并返回False
                print(f"初始化失败: {str(e)}")
                return False
            finally:
                self.is_initializing = False
        # 如果正在初始化中，等待初始化完成
        elif self.is_initializing:
            print("模型正在初始化中，请稍候...")
            # 等待初始化完成
            while self.is_initializing:
                time.sleep(0.5)
            return self.cosyvoice is not None
        # 如果CosyVoice实例已经初始化，直接返回True
        return True
    
    def zero_shot_synthesis(self, azure_key, azure_region, tts_model_path, language, target_text, prompt_audio):
        """
        Zero-shot语音合成
        
        使用zero-shot模式进行语音合成，只需要提供目标文本和提示音频，
        不需要额外的指令文本。系统会自动识别提示音频内容，并使用提示音频的声音特征
        合成目标文本的语音。
        
        参数:
            azure_key (str): Azure语音服务的密钥
            azure_region (str): Azure语音服务的区域
            tts_model_path (str): CosyVoice模型路径
            language (str): 语言代码
            target_text (str): 要合成的目标文本
            prompt_audio (str): 提示音频文件路径
            
        返回:
            tuple: (合成结果消息, 合成音频文件路径)
        """
        # 确保CosyVoice已初始化
        if not self.ensure_initialized(azure_key, azure_region, tts_model_path, language):
            return "错误: 初始化CosyVoice失败", None
        
        # 检查是否提供了提示音频
        if prompt_audio is None:
            return "错误: 请上传提示音频", None
        
        try:
            # 使用zero-shot模式合成
            output_paths = self.cosyvoice.zero_shot_with_auto_prompt(
                target_text,
                prompt_audio
            )
            
            if not output_paths:
                return "语音合成失败", None
            
            return "语音合成成功", output_paths[0]
        except Exception as e:
            return f"语音合成出错: {str(e)}", None
    
    def instruct_synthesis(self, azure_key, azure_region, tts_model_path, language, target_text, prompt_audio, instruct_text):
        """
        Instruct语音合成
        
        使用instruct模式进行语音合成，除了目标文本和提示音频外，
        还需要提供指令文本，用于控制合成语音的风格、情感等特性。
        
        参数:
            azure_key (str): Azure语音服务的密钥
            azure_region (str): Azure语音服务的区域
            tts_model_path (str): CosyVoice模型路径
            language (str): 语言代码
            target_text (str): 要合成的目标文本
            prompt_audio (str): 提示音频文件路径
            instruct_text (str): 指令文本，如"用四川话说这句话"
            
        返回:
            tuple: (合成结果消息, 合成音频文件路径)
        """
        # 确保CosyVoice已初始化
        if not self.ensure_initialized(azure_key, azure_region, tts_model_path, language):
            return "错误: 初始化CosyVoice失败", None
        
        # 检查是否提供了提示音频
        if prompt_audio is None:
            return "错误: 请上传提示音频", None
        
        # 检查是否提供了指令文本
        if not instruct_text:
            return "错误: 请输入指令文本", None
        
        try:
            # 使用instruct模式合成
            output_paths = self.cosyvoice.instruct_with_auto_prompt(
                target_text,
                prompt_audio,
                instruct_text
            )
            
            if not output_paths:
                return "语音合成失败", None
            
            return "语音合成成功", output_paths[0]
        except Exception as e:
            return f"语音合成出错: {str(e)}", None
    
    def cross_lingual_synthesis(self, azure_key, azure_region, tts_model_path, language, target_text, prompt_audio):
        """
        Cross-lingual语音合成
        
        使用cross-lingual模式进行语音合成，可以使用一种语言的提示音频
        合成另一种语言的语音，保持声音特征相似。
        
        参数:
            azure_key (str): Azure语音服务的密钥
            azure_region (str): Azure语音服务的区域
            tts_model_path (str): CosyVoice模型路径
            language (str): 语言代码
            target_text (str): 要合成的目标文本，可以是与提示音频不同的语言
            prompt_audio (str): 提示音频文件路径
            
        返回:
            tuple: (合成结果消息, 合成音频文件路径)
        """
        # 确保CosyVoice已初始化
        if not self.ensure_initialized(azure_key, azure_region, tts_model_path, language):
            return "错误: 初始化CosyVoice失败", None
        
        # 检查是否提供了提示音频
        if prompt_audio is None:
            return "错误: 请上传提示音频", None
        
        try:
            # 使用cross-lingual模式合成
            output_paths = self.cosyvoice.cross_lingual_with_auto_prompt(
                target_text,
                prompt_audio
            )
            
            if not output_paths:
                return "语音合成失败", None
            
            return "语音合成成功", output_paths[0]
        except Exception as e:
            return f"语音合成出错: {str(e)}", None
    
    def launch_ui(self):
        """
        启动Web界面
        
        使用Gradio创建并启动Web界面，包括配置参数面板和三个语音合成模式的标签页。
        每个标签页包含输入区域和输出区域，用户可以在输入区域提供必要的参数，
        点击合成按钮后在输出区域查看结果。
        """
        # 开始预加载模型
        self.preload_model()
        
        # 使用Gradio的Blocks创建Web界面
        with gr.Blocks(title="集成Azure语音识别的CosyVoice") as demo:
            # 添加标题和说明
            gr.Markdown("# 集成Azure语音识别的CosyVoice")
            gr.Markdown("这个应用集成了Azure语音识别功能，可以自动识别提示音频内容，无需手动输入prompt文本。")
            
            # 添加模型加载状态提示
            with gr.Row():
                loading_status = gr.Textbox(label="模型加载状态", value="正在后台加载模型，首次使用可能需要等待...", interactive=False)
                refresh_btn = gr.Button("刷新状态")
            
            # 创建配置参数面板（默认折叠）
            with gr.Accordion("配置参数", open=False):
                # Azure密钥输入框，预填默认值
                azure_key = gr.Textbox(label="Azure密钥", placeholder="输入你的Azure语音服务密钥", value=self.default_azure_key)
                # Azure区域输入框，预填默认值
                azure_region = gr.Textbox(label="Azure区域", placeholder="输入你的Azure语音服务区域", value=self.default_azure_region)
                # CosyVoice模型路径输入框，预填默认值
                tts_model_path = gr.Textbox(label="CosyVoice模型路径", placeholder="输入CosyVoice模型路径", value=self.default_tts_model_path)
                # 语言选择下拉框，支持自动检测和多种语言
                language = gr.Dropdown(
                    label="语言", 
                    choices=[
                        "auto", 
                        "zh-CN", "en-US", 
                        "ja-JP", "ko-KR", 
                        "fr-FR", "de-DE", 
                        "es-ES", "ru-RU", 
                        "it-IT", "pt-BR"
                    ], 
                    value=self.default_language,
                    info="auto: 自动检测, zh-CN: 中文, en-US: 英文, ja-JP: 日语, ko-KR: 韩语, fr-FR: 法语, de-DE: 德语, es-ES: 西班牙语, ru-RU: 俄语, it-IT: 意大利语, pt-BR: 葡萄牙语(巴西)"
                )
            
            # Cross-lingual语音合成标签页（移到最前面作为默认标签页）
            with gr.Tab("Cross-lingual语音合成（跨语言）"):
                with gr.Row():
                    # 左侧输入区域
                    with gr.Column():
                        # 目标文本输入框
                        cross_lingual_target_text = gr.Textbox(label="目标文本", placeholder="输入要合成的文本", lines=5)
                        # 提示音频上传控件
                        cross_lingual_prompt_audio = gr.Audio(label="提示音频", type="filepath")
                        # 合成按钮
                        cross_lingual_btn = gr.Button("合成")
                    # 右侧输出区域
                    with gr.Column():
                        # 合成结果文本框
                        cross_lingual_output = gr.Textbox(label="合成结果")
                        # 合成音频播放控件
                        cross_lingual_audio = gr.Audio(label="合成音频")
            
            # Zero-shot语音合成标签页
            with gr.Tab("Zero-shot语音合成"):
                with gr.Row():
                    # 左侧输入区域
                    with gr.Column():
                        # 目标文本输入框
                        zero_shot_target_text = gr.Textbox(label="目标文本", placeholder="输入要合成的文本", lines=5)
                        # 提示音频上传控件
                        zero_shot_prompt_audio = gr.Audio(label="提示音频", type="filepath")
                        # 合成按钮
                        zero_shot_btn = gr.Button("合成")
                    # 右侧输出区域
                    with gr.Column():
                        # 合成结果文本框
                        zero_shot_output = gr.Textbox(label="合成结果")
                        # 合成音频播放控件
                        zero_shot_audio = gr.Audio(label="合成音频")
            
            # Instruct语音合成标签页
            with gr.Tab("Instruct语音合成"):
                with gr.Row():
                    # 左侧输入区域
                    with gr.Column():
                        # 目标文本输入框
                        instruct_target_text = gr.Textbox(label="目标文本", placeholder="输入要合成的文本", lines=5)
                        # 提示音频上传控件
                        instruct_prompt_audio = gr.Audio(label="提示音频", type="filepath")
                        # 指令文本输入框
                        instruct_text = gr.Textbox(label="指令文本", placeholder="输入指令，如：用四川话说这句话")
                        # 合成按钮
                        instruct_btn = gr.Button("合成")
                    # 右侧输出区域
                    with gr.Column():
                        # 合成结果文本框
                        instruct_output = gr.Textbox(label="合成结果")
                        # 合成音频播放控件
                        instruct_audio = gr.Audio(label="合成音频")
            
            # 设置事件处理
            # 当点击Zero-shot合成按钮时，调用zero_shot_synthesis方法
            zero_shot_btn.click(self.zero_shot_synthesis, 
                               inputs=[azure_key, azure_region, tts_model_path, language, zero_shot_target_text, zero_shot_prompt_audio], 
                               outputs=[zero_shot_output, zero_shot_audio])
            
            # 当点击Instruct合成按钮时，调用instruct_synthesis方法
            instruct_btn.click(self.instruct_synthesis, 
                              inputs=[azure_key, azure_region, tts_model_path, language, instruct_target_text, instruct_prompt_audio, instruct_text], 
                              outputs=[instruct_output, instruct_audio])
            
            # 当点击Cross-lingual合成按钮时，调用cross_lingual_synthesis方法
            cross_lingual_btn.click(self.cross_lingual_synthesis, 
                                   inputs=[azure_key, azure_region, tts_model_path, language, cross_lingual_target_text, cross_lingual_prompt_audio], 
                                   outputs=[cross_lingual_output, cross_lingual_audio])
            
            # 定义更新加载状态的函数
            def update_loading_status():
                if self.cosyvoice is not None:
                    return "模型已加载完成，可以开始使用"
                elif self.is_initializing:
                    return "正在加载模型，请稍候..."
                else:
                    return "模型尚未加载，首次使用时将自动加载"
            
            # 使用刷新按钮更新状态，而不是使用every参数
            refresh_btn.click(update_loading_status, outputs=[loading_status])
            
            # 在页面加载时更新一次状态
            demo.load(update_loading_status, outputs=[loading_status])
        
        # 启动Gradio界面，share=True表示生成一个可公开访问的链接
        demo.queue()  # 启用队列处理，避免并发请求导致的问题
        demo.launch(share=True)

if __name__ == "__main__":
    # 检查是否安装了gradio库
    try:
        import gradio
    except ImportError:
        # 如果未安装gradio，自动安装
        print("错误: 未安装gradio库，正在安装...")
        import subprocess
        # 使用pip安装gradio
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        print("gradio安装完成，请重新运行程序")
        # 退出程序，让用户重新运行
        sys.exit(1)
    
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description="集成Azure语音识别的CosyVoice Web界面")
    parser.add_argument("--preload", action="store_true", help="启动时预加载模型，减少首次生成的等待时间")
    parser.add_argument("--fp16", action="store_true", help="使用半精度浮点数(FP16)加速，可能会略微降低质量")
    parser.add_argument("--no-optimize", action="store_true", help="禁用首次运行优化")
    parser.add_argument("--azure-key", type=str, default="2e031a64874b4625b4d50b58f9006bab", help="Azure语音服务的密钥")
    parser.add_argument("--azure-region", type=str, default="eastasia", help="Azure语音服务的区域")
    parser.add_argument("--model-path", type=str, default="pretrained_models/CosyVoice2-0.5B", help="CosyVoice模型路径")
    parser.add_argument("--language", type=str, default="auto", help="语言代码，auto为自动检测")
    parser.add_argument("--output-dir", type=str, default="outputs", help="指定输出音频文件的保存目录")
    args = parser.parse_args()
    
    # 创建Web界面实例
    webui = CosyVoiceWithAzureWebUI()
    
    # 设置默认参数
    webui.default_azure_key = args.azure_key
    webui.default_azure_region = args.azure_region
    webui.default_tts_model_path = args.model_path
    webui.default_language = args.language
    
    # 使用命令行指定的输出目录替代临时目录
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    webui.temp_dir = output_dir
    print(f"Output files will be saved to: {os.path.abspath(output_dir)}")
    
    # 如果指定了预加载，则在启动前预加载模型
    if args.preload:
        print("启用模型预加载，这将减少首次生成的等待时间")
        # 创建CosyVoiceWithAzure实例，使用命令行参数
        try:
            webui.cosyvoice = CosyVoiceWithAzure(
                azure_key=args.azure_key,
                azure_region=args.azure_region,
                tts_model_path=args.model_path,
                language=args.language,
                output_dir=webui.temp_dir,
                optimize_for_first_run=not args.no_optimize,
                use_fp16=args.fp16
            )
            print("模型预加载完成！系统已准备就绪。")
        except Exception as e:
            print(f"模型预加载失败: {str(e)}")
            print("将在首次使用时再次尝试加载模型。")
    
    # 启动Web界面
    webui.launch_ui() 