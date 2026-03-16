FROM python:3.11-slim

WORKDIR /app

# Install dependensi sistem (Sudah oke di log kamu)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- PERBAIKAN DI SINI ---
# Copy requirements duluan agar di-cache
COPY requirements.txt ./

# Install library (Gunakan --no-cache-dir untuk menghemat RAM)
RUN pip3 install --no-cache-dir -r requirements.txt

# Baru copy source code
COPY src/ ./src/

EXPOSE 8501

# Gunakan python -m streamlit agar tidak kena error "executable not found" lagi
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]