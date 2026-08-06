FROM apache/spark:3.5.1

USER root
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stream_processor.py .
CMD ["python", "stream_processor.py"]