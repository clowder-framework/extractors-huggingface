FROM python:3.10

WORKDIR /extractor
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY segment_anything_file.py segment_anything_file_extractor.py extractor_info.json ./
CMD python3 -u segment_anything_file_extractor.py