FROM jupyter/minimal-notebook:hub-4.0.2
USER root
RUN apt-get update
USER ${NB_USER}
RUN conda install --quiet --yes nb_conda_kernels
COPY --chown=${NB_USER} . /codegeex-extension 
RUN pip install -ve /codegeex-extension