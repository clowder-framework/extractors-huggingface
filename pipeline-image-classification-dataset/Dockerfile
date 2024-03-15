FROM python:3.10

WORKDIR /extractor
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY image_classification_dataset.py extractor_info.json ./
CMD python3 -u image_classification_dataset.py