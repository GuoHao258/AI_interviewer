@echo off
REM Turn off command echo for cleaner interface

echo ======================================================
echo Starting CosyVoice with Azure (High Performance Mode)...
echo ======================================================
echo The system will preload the model and use FP16 precision
echo This will reduce memory usage and increase speed
echo Output files will be saved to: %CD%\..\outputs\audio_highperf
echo Please wait, the browser interface will open automatically
echo ======================================================

REM Activate Conda environment
call conda activate Cosyvoice1

REM Set environment variables to optimize CUDA performance
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Create output directory if it doesn't exist
if not exist ..\outputs\audio_highperf mkdir ..\outputs\audio_highperf

REM Run Python script with preloading, FP16 acceleration and specific output directory
cd ..
python cosyvoice_with_azure_webui.py --preload --fp16 --output-dir outputs\audio_highperf

REM Pause to let user see output
pause 