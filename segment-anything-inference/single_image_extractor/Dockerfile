FROM python:3.10

WORKDIR /extractor
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY segment_anything_file.py segment_anything_file_extractor.py extractor_info.json ./

# Download VIT-H model
RUN curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -o sam_vit_h_4b8939.pth


CMD python3 -u segment_anything_file_extractor.py