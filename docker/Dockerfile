from pytorch/pytorch

COPY requirements.txt .

RUN apt-get update -y
RUN pip install numpy tqdm hydra_core hydra_colorlog pystoi torchaudio librosa
RUN pip install numba
RUN apt-get install -y gcc
RUN pip install pesq

RUN pip install soundfile
RUN apt-get update -y
RUN apt-get install -y g++ git
RUN git clone https://github.com/ty274/rir-generator.git
RUN cd rir-generator && pip install -e ./nprirgen
RUN pip install pydub