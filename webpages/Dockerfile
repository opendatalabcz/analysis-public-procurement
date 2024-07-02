FROM python:3.11.3

# Install Quarto
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    curl \
    gdebi-core \
    && rm -rf /var/lib/apt/lists/*

ARG QUARTO_VERSION="1.4.553"
RUN curl -o quarto-linux-amd64.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb
RUN gdebi --non-interactive quarto-linux-amd64.deb

# create a directory for the webpages
RUN mkdir "./webpages"
RUN mkdir "./webpages/www"
RUN mkdir "./webpages/data"

COPY preprocessing.py /webpages/preprocessing.py
COPY public_procurements.ipynb /webpages/public_procurements.ipynb
COPY requirements.txt /webpages/requirements.txt
COPY www/images /webpages/www/images
COPY generate.sh /generate.sh
COPY catboost_model_production /webpages/catboost_model_production

# install the requirements
RUN pip3.11 install -r /webpages/requirements.txt

RUN chmod +x /generate.sh

CMD /generate.sh


