FROM python:3
RUN pip3 install jupyter
EXPOSE 80 8888 5000
ADD code /
CMD ["python", "/code/exploration/EDA-Mojgan_Teifouri.ipynb"]
