# our base image
FROM python:3.9


# upgrade pip
RUN pip install --upgrade pip


# copy files required for the app to run
RUN mkdir -p /opt/query_autocomplete
COPY data /opt/query_autocomplete/data/
COPY model /opt/query_autocomplete/model/
COPY api.py /opt/query_autocomplete
COPY query_autocomplete.py /opt/query_autocomplete
COPY requirements.txt /opt/query_autocomplete


# install Python modules needed by the Python app
RUN pip install --no-cache-dir -r /opt/query_autocomplete/requirements.txt

# download 'punkt' from 'nltk'
RUN python -c "import nltk ; nltk.download('punkt')"


WORKDIR /opt/query_autocomplete


# tell the port number the container should expose
EXPOSE 5000
