FROM python:3.12.2-alpine

ENV PYTHONUNBUFFERED 1

RUN mkdir /rnn_api

WORKDIR /rnn_api

COPY requirements.txt .

RUN apk add --no-cache gcc musl-dev && \
    pip install -r requirements.txt && \
    apk del gcc musl-dev

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:5000", "--workers", \
     "3", "--timeout", "300"]
