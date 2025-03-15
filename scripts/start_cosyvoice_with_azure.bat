@echo off
REM Turn off command echo for cleaner interface

echo ======================================================
echo Starting CosyVoice with Azure...
echo ======================================================
echo The system will preload the model to reduce first-run delay
echo Output files will be saved to: %CD%\..\outputs\audio
echo Please wait, the browser interface will open automatically
echo ======================================================

REM Activate Conda environment
call conda activate Cosyvoice1

REM Set environment variables to optimize CUDA performance
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Create output directory if it doesn't exist
if not exist ..\outputs\audio mkdir ..\outputs\audio

REM Run Python script with preloading and specific output directory
cd ..
python cosyvoice_with_azure_webui.py --preload --output-dir outputs\audio

REM Pause to let user see output
pause 