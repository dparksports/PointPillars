#Instruction

echo "# PointPillars" >> README.md

```Bash
    7  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    8  sudo dpkg -i Downloads/cuda-repo-ubuntu1604_10.1.168-1_amd64.deb 
    9  sudo apt-get update;sudo apt-get upgrade
   10  sudo apt-get install cuda
   14  tar -xf Downloads/Python-3.7.3.tar.xz 
   18  sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
   19  ./configure --enable-optimizations
   20  make -j 12
   21  sudo make altinstall

   26  md5sum Downloads/Miniconda3-latest-Linux-x86_64.sh 
   27  sudo bash Downloads/Miniconda3-latest-Linux-x86_64.sh 
   29  conda create -n pointpillars python=3.7 anaconda
   30  sudo apt-get install libboost-all-dev
   
   31  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
   38  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
   70  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip
   
   42  conda create -n pointpillars python=3.7 anaconda
   43  conda activate pointpillars
   44  conda install shapely pybind11 protobuf scikit-image numba pillow
   45  conda install pytorch torchvision -c pytorch
   46  conda install google-sparsehash -c bioconda
   47  pip install --upgrade pip
   48  pip install fire tensorboardX
   
   51  git clone git@github.com:facebookresearch/SparseConvNet.git
   52  git clone https://github.com/facebookresearch/SparseConvNet.git
   53  cd SparseConvNet/
   56  bash build.sh
   58  conda install scikit-image scipy numba pillow matplotlib
   
   61  tar -xf Downloads/cmake-3.14.5.tar.gz 
   65  ./bootstrap 
   66  make -j 12
   67  sudo make install
   68  cmake --version
  
  288  sudo cp cuda/include/cudnn.h /usr/local/cuda/include
  289  sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
  290  sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

```
