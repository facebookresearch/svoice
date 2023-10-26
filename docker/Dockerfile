from pytorch/pytorch

COPY svoice/requirements.txt .

RUN apt-get update -y
RUN pip install numpy tqdm hydra_core hydra_colorlog pystoi torchaudio librosa
RUN pip install numba
RUN apt-get install -y gcc
RUN pip install pesq