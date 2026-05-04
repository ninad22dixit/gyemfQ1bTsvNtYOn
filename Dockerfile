FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY requirements.txt requirements.txt
COPY requirements requirements
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "trading_pipeline.pipeline"]
