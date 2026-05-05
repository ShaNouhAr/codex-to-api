FROM python:3.12-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir "httpx>=0.27,<1" "uvicorn[standard]>=0.30,<1" "fastapi>=0.115,<1"
COPY main.py .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
