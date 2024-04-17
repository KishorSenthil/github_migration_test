

FROM python:3.9

WORKDIR /mle_training

RUN apt-get update

COPY . .

RUN pip install build

RUN python -m build

RUN mkdir -p /mle_training/logs

RUN  pip install -r requirements.txt


CMD ["sh", "-c", "python ./src/housing/run_script.py && pytest && mlflow ui "]