FROM python:3.10

RUN pip3 install numpy matplotlib

RUN mkdir /code

COPY gradient_descent.py /code

WORKDIR /code