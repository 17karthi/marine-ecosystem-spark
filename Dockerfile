FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_LOCAL_HOSTNAME=localhost
ENV PYSPARK_PYTHON=python3
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.zip .
RUN python -m zipfile -e models.zip models

COPY app.py .

EXPOSE 10000
CMD ["python", "app.py"]
