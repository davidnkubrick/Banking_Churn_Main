FROM ubuntu

RUN apt update && apt install -y \
    python3-dev \
    python3-pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY README.md /
COPY src/ /src/
COPY resources /resources/

WORKDIR /src

CMD ["python3", "main.py"]