conda create -n turbo-genius -y python=3.13
conda run -n turbo-genius conda config --add channels nvidia
conda run -n turbo-genius conda install -y cudatoolkit