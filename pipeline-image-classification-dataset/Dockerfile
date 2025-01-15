FROM minio-mounted-clowder-extractor

WORKDIR /extractor
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY image_classification_dataset.py extractor_info.json ./

# Clone pyclowder, install specific branch, and return to working directory
RUN git clone https://github.com/clowder-framework/pyclowder.git && \
    cd pyclowder && \
    git checkout 116-modify-download-apis-for-minio-mounted-fs && \
    python setup.py install && \
    cd .. && \
    rm -rf pyclowder

CMD ["python3", "-u", "image_classification_dataset.py"]