FROM python:3.10

RUN pip3 install numpy matplotlib

COPY gradient_descent.py /code

WORKDIR /code