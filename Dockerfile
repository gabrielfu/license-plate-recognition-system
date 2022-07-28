FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN mkdir -p /app
WORKDIR /app

# libraries required by opencv
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

# Set up the program in the image
COPY . .
ENV PATH="/app:${PATH}"

CMD ["python3", "./run_lprs.py"]