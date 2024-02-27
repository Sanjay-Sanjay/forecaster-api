FROM python:3.9
USER root
RUN mkdir /code
RUN mkdir /code/logs
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
RUN chmod 777 -R /code/logs && chgrp -R 0 /code/logs
RUN chown -R 1000:1000 /code/logs
EXPOSE 9008
CMD ["python","./app.py"]