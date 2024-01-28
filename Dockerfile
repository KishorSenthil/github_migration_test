FROM python:3.9

RUN apt-get update && apt-get install -y git

RUN git clone https://REMOVED_GITHUB_TOKEN0AAxcO0O5LNLgC7yXjoyyPmS5Pxh5P3MYJgp@github.com/KishorSenthil/mle_training.git

COPY requirements.txt .

COPY ./src ./src

RUN pip install build

RUN pip install -r requirements.txt

CMD ["python", "./src/housing/run_script.py"]