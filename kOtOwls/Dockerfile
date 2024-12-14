FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    wget \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/local/lib/python3.11/site-packages/pix2tex/model/checkpoints && \
    wget -O /usr/local/lib/python3.11/site-packages/pix2tex/model/checkpoints/weights.pth \
        https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/weights.pth && \
    wget -O /usr/local/lib/python3.11/site-packages/pix2tex/model/checkpoints/image_resizer.pth \
        https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1/image_resizer.pth


COPY . /app/
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]