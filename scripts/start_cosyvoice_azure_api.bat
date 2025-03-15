@echo off
REM Turn off command echo for cleaner interface

echo ======================================================
echo Starting CosyVoice with Azure FastAPI Server...
echo ======================================================
echo The system will start a FastAPI server on port 50000
echo You can access the API at http://localhost:50000
echo Output files will be saved to: %CD%\..\outputs\api
echo ======================================================

REM Activate Conda environment
call conda activate Cosyvoice1

REM Set environment variables to optimize CUDA performance
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Create output directory if it doesn't exist
if not exist ..\outputs\api mkdir ..\outputs\api

REM Run FastAPI server with Azure integration
cd ..
python runtime/python/fastapi/server_with_azure.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B --output_dir outputs/api

REM Pause to let user see output
pause 