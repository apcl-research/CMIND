FROM ubuntu:24.04

# Update package list and install Python + pip
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for Joern
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk-headless \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Joern using the official installer script
WORKDIR /opt
RUN curl -L https://github.com/joernio/joern/releases/latest/download/joern-install.sh -o joern-install.sh \
    && chmod +x joern-install.sh \
    && ./joern-install.sh --install-dir /opt/joern \
    && rm joern-install.sh

# Add Joern to PATH
#ENV PATH="/opt/joern:$PATH"
#ENV PATH="/opt/joern/joern-cli/bin:$PATH"
# Set working directory inside container
WORKDIR /app

# Install Doxygen + graphviz
RUN apt-get update && apt-get install -y \
    doxygen \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install ngrok
#RUN curl -s https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -o ngrok.tgz \
#    && tar -xzf ngrok.tgz \
#    && mv ngrok /usr/local/bin/ \
#    && rm ngrok.tgz
COPY openai.key .
# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment
RUN python3 -m venv /opt/venv

# Activate venv and install dependencies
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Ensure the venv is used by default
ENV PATH="/opt/venv/bin:$PATH"
#ENV PATH="/opt/venv/bin:/opt/joern/joern-cli/bin:$PATH"
#RUN ngrok authtoken 31bvqUuEvlRYEdDBtcgLvTqF494_3z8qMDh69ZmSdPtxYvXwg
# Copy your code into the container
COPY . .
#CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
# Expose Flask port
#EXPOSE 5000 4040
EXPOSE 5000 
#COPY start.sh /app/start.sh
#CMD ["/app/start.sh"]
# Run your Python app
#CMD ["python3", "app.py"]
CMD ["bash"]
