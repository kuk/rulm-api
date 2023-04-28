FROM python:3.10.8-slim

RUN apt-get update && apt-get install -y \
    cmake \
    gcc g++

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
ENTRYPOINT ["python", "main.py"]
