# 1. Base Image kiválasztása
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
# 2. Munkakönyvtár beállítása a konténeren belül
WORKDIR /app
# 3. Rendszerszintű függőségek telepítése
RUN apt-get update && apt-get install -y --no-install-recommends \
 git \
 libgl1-mesa-glx \
 libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
# 4. Python függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Create directories (dev container will mount over these)
RUN mkdir -p /app/src /app/notebooks /app/output /data
# 6. Copy code (only used for standalone docker run, not dev container)
COPY ./src ./src
# 7. Script permissions
RUN if [ -f ./src/run.sh ]; then chmod +x ./src/run.sh; fi
# 8. Default command (not used in dev container)
CMD ["bash"]