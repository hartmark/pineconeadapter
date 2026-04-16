FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=11434
ENV EMBED_MODEL=llama-text-embed-v2
ENV EMBED_DIMENSIONS=2048

EXPOSE 11434

CMD ["python", "app.py"]
