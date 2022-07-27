FROM python:3.9-slim

RUN mkdir -p /app
WORKDIR /app

RUN apt clean \
    && apt -y update \
    && apt install -y libgl1 libopencv-dev python3-opencv \
    && python -m pip install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Set up the program in the image
COPY . .
ENV PATH="/app:${PATH}"

CMD ["python3", "./run_lprs.py"]