
# how to set envs?
# hnswlib 
# delete previous chromadb - alter in code


FROM python:3.12-slim-bookworm

LABEL maintainer="Dora Schuller <dschuller@pamgene.com>"

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN apt-get update --fix-missing && apt-get install python3.12-dev && apt-get install -y --fix-missing build-essential


# ENV GRADIO_SERVER_NAME="0.0.0.0"
# ARG PYTHON_ENV=my_env
# ENV PYTHON_ENV=$PYTHON_ENV


EXPOSE 6060

CMD ["python", "app.py"]