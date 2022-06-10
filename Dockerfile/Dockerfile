FROM pytorch/pytorch

RUN apt update
RUN apt install -y git

RUN conda install -c conda-forge --name base --update-deps --force-reinstall ipykernel

RUN pip3 install pandas matplotlib scikit-learn streamlit regex