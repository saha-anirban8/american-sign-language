FROM python:3.8

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "--server.maxUploadSize=5"]
CMD ["src/st_asl.py"]