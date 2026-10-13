FROM python:3.12-slim
# ffmpeg only repackages downloads (no re-encoding); the browser does all the video reading
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY static ./static
ENV SHADOWBOARD_DATA=/data
VOLUME /data
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
