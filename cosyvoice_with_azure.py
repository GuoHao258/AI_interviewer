import os
import sys
import time
import argparse
import torchaudio
import torch
import azure.cognitiveservices.speech as speechsdk

# 添加CosyVoice路径到系统路径，确保可以导入CosyVoice相关模块
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class CosyVoiceWithAzure:
    """
    集成Azure语音识别的CosyVoice类
    
    该类将Azure的语音识别服务与CosyVoice语音合成系统集成在一起，
    实现自动识别提示音频内容并用于语音合成的功能。
    """
    
    def __init__(self, 
                 azure_key="2e031a64874b4625b4d50b58f9006bab",
                 azure_region="eastasia",
                 tts_model_path='pretrained_models/CosyVoice2-0.5B',
                 language='auto',
                 output_dir='outputs',
                 optimize_for_first_run=True,
                 use_fp16=False):
        """
        初始化集成Azure语音识别的CosyVoice
        
        参数:
            azure_key (str): Azure语音服务的密钥，用于认证Azure服务
            azure_region (str): Azure语音服务的区域，如eastasia、westus等
            tts_model_path (str): CosyVoice模型路径，指向预训练模型文件夹
            language (str): 语言代码，'zh-CN'为中文, 'en-US'为英文, 'auto'为自动检测，影响语音识别的语言设置
            output_dir (str): 合成音频的输出目录，如不存在会自动创建
            optimize_for_first_run (bool): 是否优化首次运行速度，默认为True
            use_fp16 (bool): 是否使用半精度浮点数(FP16)加速，默认为False
        """
        print("正在初始化集成Azure语音识别的CosyVoice...")
        
        # 存储Azure语音服务配置参数
        self.azure_key = azure_key
        self.azure_region = azure_region
        self.language = language
        
        # 设置并创建输出目录（如果不存在）
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
            print(f"[OUTPUT DIRECTORY] Created output directory: {os.path.abspath(output_dir)}")
        else:
            print(f"[OUTPUT DIRECTORY] Using output directory: {os.path.abspath(output_dir)}")
        
        # 初始化Azure语音识别配置
        print("正在配置Azure语音服务...")
        # 创建SpeechConfig对象，设置订阅密钥和区域
        self.speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
        
        # 如果指定了具体语言（非auto），则设置语音识别语言
        if language != 'auto':
            self.speech_config.speech_recognition_language = language
            print(f"已设置语音识别语言: {language}")
        else:
            print("已设置为自动语言检测模式")
        
        # 初始化CosyVoice模型
        # 首先尝试加载CosyVoice2模型，如果失败则尝试加载CosyVoice模型
        print("正在加载CosyVoice模型...")
        
        # 优化首次运行的参数
        load_jit = False  # JIT编译可能会增加初始化时间
        load_trt = False  # TensorRT加速需要额外的编译时间
        fp16 = use_fp16    # 是否使用半精度浮点数
        
        if optimize_for_first_run:
            print("已启用首次运行优化")
            # 预热CUDA，减少首次推理延迟
            if torch.cuda.is_available():
                print("正在预热CUDA...")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # 创建一个小的张量并在GPU上运行一个简单操作，预热GPU
                dummy = torch.ones(1, 1).cuda()
                dummy = dummy + dummy
                torch.cuda.synchronize()
                print("CUDA预热完成")
        
        try:
            print(f"尝试加载CosyVoice2模型: {tts_model_path}")
            # 尝试加载CosyVoice2模型（更新版本）
            # load_jit=False: 不使用JIT编译
            # load_trt=False: 不使用TensorRT加速
            # fp16: 是否使用半精度浮点数（FP16）
            start_time = time.time()
            self.tts = CosyVoice2(tts_model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
            load_time = time.time() - start_time
            print(f"CosyVoice2模型加载成功，耗时: {load_time:.2f}秒")
        except Exception as e:
            print(f"加载CosyVoice2模型失败: {str(e)}")
            try:
                print(f"尝试加载CosyVoice模型: {tts_model_path}")
                # 如果CosyVoice2加载失败，尝试加载原始CosyVoice模型
                start_time = time.time()
                self.tts = CosyVoice(tts_model_path)
                load_time = time.time() - start_time
                print(f"CosyVoice模型加载成功，耗时: {load_time:.2f}秒")
            except Exception as e:
                # 如果两种模型都加载失败，则抛出异常
                print(f"错误: 无法初始化CosyVoice模型: {str(e)}")
                raise
        
        print("初始化完成! 系统已准备就绪。")
    
    def recognize_from_file(self, audio_file_path):
        """
        从音频文件识别语音内容
        
        使用Azure语音识别服务从提供的音频文件中识别语音内容，
        并返回识别出的文本。如果识别失败，则返回None。
        
        参数:
            audio_file_path (str): 要识别的音频文件路径
            
        返回:
            str或None: 识别出的文本，如果识别失败则返回None
        """
        # 检查音频文件是否存在
        if not os.path.exists(audio_file_path):
            print(f"错误: 音频文件 {audio_file_path} 不存在")
            return None
        
        print(f"开始从音频文件识别语音: {audio_file_path}")
        print(f"文件大小: {os.path.getsize(audio_file_path)} 字节")
        
        try:
            # 创建音频配置，指定要识别的音频文件
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
            
            # 根据language设置创建不同的识别器
            if self.language == 'auto':
                # 使用自动语言检测
                print("使用自动语言检测...")
                auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=["zh-CN", "en-US", "ja-JP", "ko-KR", "fr-FR", "de-DE", "es-ES", "ru-RU", "it-IT", "pt-BR"]  # 支持自动检测的语言列表
                )
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, 
                    audio_config=audio_config,
                    auto_detect_source_language_config=auto_detect_source_language_config
                )
            else:
                # 使用指定的语言
                print(f"使用指定的语言: {self.language}")
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, 
                    audio_config=audio_config
                )
            
            print("开始识别...")
            # 异步识别音频内容并等待结果
            result = speech_recognizer.recognize_once_async().get()
            print("识别完成，处理结果...")
            
            # 处理识别结果
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # 成功识别到语音内容
                print(f"识别到文本: {result.text}")
                
                # 如果使用自动语言检测，打印检测到的语言
                if self.language == 'auto' and hasattr(result, 'properties') and result.properties.get(speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult):
                    detected_language = result.properties[speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
                    print(f"检测到的语言: {detected_language}")
                
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                # 未能识别到语音内容
                print("未能识别语音")
                print(f"NoMatch原因: {result.no_match_details}")
                return None
            elif result.reason == speechsdk.ResultReason.Canceled:
                # 识别过程被取消
                cancellation_details = result.cancellation_details
                print(f"语音识别被取消: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    # 如果是因为错误而取消，打印错误详情
                    print(f"错误详情: {cancellation_details.error_details}")
                return None
        except Exception as e:
            # 捕获并打印任何异常
            print(f"文件识别出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def zero_shot_with_auto_prompt(self, target_text, prompt_audio_path, output_prefix=None):
        """
        使用自动识别的prompt文本进行zero-shot语音合成
        
        该方法首先识别提示音频的内容，然后使用识别出的文本和音频
        进行zero-shot语音合成，生成与提示音频声音相似的目标文本语音。
        
        参数:
            target_text (str): 要合成的目标文本
            prompt_audio_path (str): 提示音频文件路径，用于提取声音特征
            output_prefix (str, 可选): 输出文件前缀，默认为None（使用时间戳）
        
        返回:
            list: 输出音频文件路径列表
        """
        # 1. 加载提示音频
        print(f"加载提示音频: {prompt_audio_path}")
        # 使用CosyVoice的load_wav函数加载音频，采样率为16000Hz
        prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # 2. 识别提示音频内容
        print("识别提示音频内容...")
        # 调用之前定义的recognize_from_file方法识别音频内容
        prompt_text = self.recognize_from_file(prompt_audio_path)
        
        # 如果识别失败，使用默认提示文本
        if prompt_text is None:
            print("警告: 无法识别提示音频内容，将使用默认提示文本")
            prompt_text = "以自然的语气说这句话"
        
        # 3. 使用识别的文本和音频进行zero-shot合成
        print(f"使用识别的提示文本进行zero-shot合成: {prompt_text}")
        output_paths = []
        
        # 如果未指定输出前缀，使用时间戳作为前缀
        if output_prefix is None:
            output_prefix = f"output_{int(time.time())}"
        
        # 调用CosyVoice的inference_zero_shot方法进行合成
        # stream=False表示不使用流式合成，一次性返回所有结果
        for i, result in enumerate(self.tts.inference_zero_shot(target_text, prompt_text, prompt_speech, stream=False)):
            # 构建输出文件路径
            output_path = os.path.join(self.output_dir, f"{output_prefix}_{i}.wav")
            # 保存合成的音频到文件
            torchaudio.save(output_path, result['tts_speech'], self.tts.sample_rate)
            print(f"语音已保存到: {output_path}")
            print(f"[OUTPUT FILE] Audio saved to: {os.path.abspath(output_path)}")
            output_paths.append(output_path)
        
        return output_paths
    
    def instruct_with_auto_prompt(self, target_text, prompt_audio_path, instruct_text, output_prefix=None):
        """
        使用自动识别的prompt文本进行instruct语音合成
        
        该方法使用提示音频和指令文本进行instruct语音合成，
        可以通过指令控制合成语音的风格、情感等特性。
        
        参数:
            target_text (str): 要合成的目标文本
            prompt_audio_path (str): 提示音频文件路径，用于提取声音特征
            instruct_text (str): 指令文本，如"用四川话说这句话"，控制合成风格
            output_prefix (str, 可选): 输出文件前缀，默认为None（使用时间戳）
        
        返回:
            list: 输出音频文件路径列表
        """
        # 1. 加载提示音频
        print(f"加载提示音频: {prompt_audio_path}")
        # 使用CosyVoice的load_wav函数加载音频，采样率为16000Hz
        prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # 2. 使用instruct模式合成
        print(f"使用instruct模式合成，指令: {instruct_text}")
        output_paths = []
        
        # 如果未指定输出前缀，使用时间戳作为前缀
        if output_prefix is None:
            output_prefix = f"output_{int(time.time())}"
        
        try:
            # 首先尝试使用inference_instruct2方法（更新版本）
            for i, result in enumerate(self.tts.inference_instruct2(target_text, instruct_text, prompt_speech, stream=False)):
                # 构建输出文件路径
                output_path = os.path.join(self.output_dir, f"{output_prefix}_{i}.wav")
                # 保存合成的音频到文件
                torchaudio.save(output_path, result['tts_speech'], self.tts.sample_rate)
                print(f"语音已保存到: {output_path}")
                print(f"[OUTPUT FILE] Audio saved to: {os.path.abspath(output_path)}")
                output_paths.append(output_path)
        except AttributeError:
            # 如果inference_instruct2不可用（可能是旧版本CosyVoice），尝试使用inference_instruct
            print("inference_instruct2不可用，尝试使用inference_instruct...")
            # 使用第一个可用的说话人ID和指令文本进行合成
            for i, result in enumerate(self.tts.inference_instruct(target_text, self.tts.list_available_spks()[0], instruct_text, stream=False)):
                # 构建输出文件路径
                output_path = os.path.join(self.output_dir, f"{output_prefix}_{i}.wav")
                # 保存合成的音频到文件
                torchaudio.save(output_path, result['tts_speech'], self.tts.sample_rate)
                print(f"语音已保存到: {output_path}")
                print(f"[OUTPUT FILE] Audio saved to: {os.path.abspath(output_path)}")
                output_paths.append(output_path)
        
        return output_paths
    
    def cross_lingual_with_auto_prompt(self, target_text, prompt_audio_path, output_prefix=None):
        """
        使用自动识别的prompt文本进行cross-lingual语音合成
        
        该方法使用提示音频进行跨语言语音合成，可以使用一种语言的提示音频
        合成另一种语言的语音，保持声音特征相似。
        
        参数:
            target_text (str): 要合成的目标文本，可以是与提示音频不同的语言
            prompt_audio_path (str): 提示音频文件路径，用于提取声音特征
            output_prefix (str, 可选): 输出文件前缀，默认为None（使用时间戳）
        
        返回:
            list: 输出音频文件路径列表
        """
        # 1. 加载提示音频
        print(f"加载提示音频: {prompt_audio_path}")
        # 使用CosyVoice的load_wav函数加载音频，采样率为16000Hz
        prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # 2. 使用cross-lingual模式合成
        print("使用cross-lingual模式合成")
        output_paths = []
        
        # 如果未指定输出前缀，使用时间戳作为前缀
        if output_prefix is None:
            output_prefix = f"output_{int(time.time())}"
        
        # 调用CosyVoice的inference_cross_lingual方法进行跨语言合成
        for i, result in enumerate(self.tts.inference_cross_lingual(target_text, prompt_speech, stream=False)):
            # 构建输出文件路径
            output_path = os.path.join(self.output_dir, f"{output_prefix}_{i}.wav")
            # 保存合成的音频到文件
            torchaudio.save(output_path, result['tts_speech'], self.tts.sample_rate)
            print(f"语音已保存到: {output_path}")
            print(f"[OUTPUT FILE] Audio saved to: {os.path.abspath(output_path)}")
            output_paths.append(output_path)
        
        return output_paths

def main():
    """
    主函数，处理命令行参数并执行相应的语音合成操作
    
    该函数解析命令行参数，创建CosyVoiceWithAzure实例，
    并根据指定的模式执行相应的语音合成操作。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="集成Azure语音识别的CosyVoice")
    # 添加各种命令行参数
    parser.add_argument("--azure_key", default="2e031a64874b4625b4d50b58f9006bab", help="Azure语音服务的密钥")
    parser.add_argument("--azure_region", default="eastasia", help="Azure语音服务的区域")
    parser.add_argument("--tts_model", default="pretrained_models/CosyVoice2-0.5B", help="CosyVoice模型路径")
    parser.add_argument("--language", default="auto", help="语言代码 (auto: 自动检测, zh-CN: 中文, en-US: 英文, ja-JP: 日语, ko-KR: 韩语, fr-FR: 法语, de-DE: 德语, es-ES: 西班牙语, ru-RU: 俄语, it-IT: 意大利语, pt-BR: 葡萄牙语)")
    parser.add_argument("--output_dir", default="outputs", help="输出目录")
    parser.add_argument("--mode", choices=["zero_shot", "instruct", "cross_lingual"], default="cross_lingual", help="合成模式")
    parser.add_argument("--target_text", required=True, help="要合成的目标文本")
    parser.add_argument("--prompt_audio", required=True, help="提示音频文件路径")
    parser.add_argument("--instruct_text", help="指令文本（仅在instruct模式下使用）")
    parser.add_argument("--output_prefix", help="输出文件前缀")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建CosyVoiceWithAzure实例
    cosyvoice_with_azure = CosyVoiceWithAzure(
        azure_key=args.azure_key,
        azure_region=args.azure_region,
        tts_model_path=args.tts_model,
        language=args.language,
        output_dir=args.output_dir
    )
    
    # 根据指定的模式执行相应的语音合成操作
    if args.mode == "zero_shot":
        # 执行zero-shot语音合成
        cosyvoice_with_azure.zero_shot_with_auto_prompt(
            args.target_text,
            args.prompt_audio,
            args.output_prefix
        )
    elif args.mode == "instruct":
        # 检查instruct模式是否提供了指令文本
        if not args.instruct_text:
            print("错误: 在instruct模式下必须指定指令文本")
            return
        
        # 执行instruct语音合成
        cosyvoice_with_azure.instruct_with_auto_prompt(
            args.target_text,
            args.prompt_audio,
            args.instruct_text,
            args.output_prefix
        )
    elif args.mode == "cross_lingual":
        # 执行cross-lingual语音合成
        cosyvoice_with_azure.cross_lingual_with_auto_prompt(
            args.target_text,
            args.prompt_audio,
            args.output_prefix
        )

# 当脚本直接运行时执行main函数
if __name__ == "__main__":
    main() 