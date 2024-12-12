FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python vector_db/create_faiss_db.py
CMD ["python", "main.py"]