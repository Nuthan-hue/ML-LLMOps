# base image
FROM python:3.11-slim-buster

# workind directory
WORKDIR dev

# copy
COPY . dev

# run
RUN pip install -r requirements.txt

# commands
CMD ["python3", "dev.py"]


# docker build -t(tag) nuthan/myapp:latest .
# docker run -d (for continuous running of container, even when application is stopped)
