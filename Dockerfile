FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ 

WORKDIR /niviz
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e . --use-deprecated=legacy-resolver

ENTRYPOINT ["python", "/niviz/niviz/make_svgs.py"]
