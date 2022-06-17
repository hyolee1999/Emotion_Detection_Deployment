FROM python:3.6-alpine

COPY requirements.txt .

# RUN apt-get update && apt-get install -y python3-opencv

RUN pip install --no-cache-dir -r requirements.txt

COPY ./emodetection ./emodetection

WORKDIR /emodetection

CMD ["python","app.py"]
