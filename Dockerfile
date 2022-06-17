FROM python:3.8

COPY requirements.txt .

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install -r requirements.txt

COPY ./emodetection ./emodetection

WORKDIR /emodetection

CMD ["python","app.py"]
