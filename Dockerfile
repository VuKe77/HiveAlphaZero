FROM python:3.10.12

WORKDIR /app

# copy and install requirements
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# install torch with CUDA support
# cu118 appears to be the latest; cu119, cu120 and so on don't work (access denied)
RUN pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir 

COPY ChessAlphaZero ChessAlphaZero

# -u flag for unbuffered output, otherwise python output gets buffered
# inside docker container and only gets flushed when the script exits
CMD ["pip", "list"]
#CMD ["python", "-u", "ChessAlphaZero/AlphaZeroDP.py"]