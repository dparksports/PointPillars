# What is
- These steps are to setup the KITTI dataset and/or the NUScene dataset, train and evaluate a PointPillar network from scratch.

# Performance
- Based on Ubuntu 16 with CUDA 10.1 on GTX 1060 Mobile.
- Training takes about 4 hours per epoch.

# Instruction

- You can use my bash history as shown below.  
- It takes about 1700 steps to get there.
- I will come back and edit them for a simple and clean instruction.


```plain
    1  sudo ufw enable
    2  sudo apt-get update;sudo apt-get upgrade
    3  history
    4  sudo apt-get update;sudo apt-get upgrade
    5  sudo apt autoremove
    6  sudo apt-get update;sudo apt-get upgrade
    7  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    8  sudo dpkg -i Downloads/cuda-repo-ubuntu1604_10.1.168-1_amd64.deb 
    9  sudo apt-get update;sudo apt-get upgrade
   10  sudo apt-get install cuda
   11  history
   12  sudo apt-get update;sudo apt-get upgrade
   13  sudo dpkg -i Downloads/opera-stable_62.0.3331.18_amd64.deb 
   14  tar -xf Downloads/Python-3.7.3.tar.xz 
   15  cd Python-3.7.3/
   16  history
   17  sudo apt-get update;sudo apt-get upgrade
   18  sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
   19  ./configure --enable-optimizations
   20  make -j 12
   21  sudo make altinstall
   22  python --version
   23  python3 --version
   24  python3.7 --version
   25  cd
   26  md5sum Downloads/Miniconda3-latest-Linux-x86_64.sh 
   27  sudo bash Downloads/Miniconda3-latest-Linux-x86_64.sh 
   28  gedit conda.sh
   29  conda create -n pointpillars python=3.7 anaconda
   30  sudo apt-get install libboost-all-dev
   31  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
   32  fg
   33  ll
   34  ls
   35  ll *.zip
   36  df .
   37  sudo apt-get install chromium-browser
   38  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
   39  fg
   40  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
   41  source conda.sh 
   42  conda create -n pointpillars python=3.7 anaconda
   43  conda activate pointpillars
   44  conda install shapely pybind11 protobuf scikit-image numba pillow
   45  conda install pytorch torchvision -c pytorch
   46  conda install google-sparsehash -c bioconda
   47  pip install --upgrade pip
   48  pip install fire tensorboardX
   49  git clone git@github.com:facebookresearch/SparseConvNet.git
   50  sudo apt install git
   51  git clone git@github.com:facebookresearch/SparseConvNet.git
   52  git clone https://github.com/facebookresearch/SparseConvNet.git
   53  cd SparseConvNet/
   54  bash build.sh
   55  gedit build.sh
   56  bash build.sh
   57  gedit ~/.bashrc
   58  conda install scikit-image scipy numba pillow matplotlib
   59  gedit ~/.bashrc
   60  cd ..
   61  tar -xf Downloads/cmake-3.14.5.tar.gz 
   62  cd cmake-3.14.5/
   63  ls
   64  gedit README.rst &
   65  ./bootstrap 
   66  make -j 12
   67  sudo make install
   68  cmake --version
   69  cd ..
   70  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip
   71  fg
   72  gsettings set org.gnome.nautilus.preferences default-folder-viewer 'list-view'
   73  cmake --version
   74  history|grep apt-get
   75  sudo apt-get update;sudo apt-get upgrade
   76  cd /etc/modprobe.d/
   77  ls
   78  sudo tee blacklist-ideapad.conf <<< "blacklist ideapad-laptop"
   79  cat blacklist-ideapad.conf 
   80  sudo rfkill list
   81  history|grep apt
   82  sudo apt-get update;sudo apt-get upgrade
   83  blid
   84  blkid
   85  gedit .bashrc
   86  cd Downloads/
   87  ls
   88  cd
   89  gedit .bashrc
   90  sudo apt-get update
   91  sudo apt-get install libboost-all-dev
   92  cmake --version
   93  nvcc
   94  /usr/local/cuda/bin/nvcc
   95  /usr/local/cuda/bin/nvcc --version
   96  $ ssh-keygen -t rsa -b 4096 -C "dpark.sports@gmail.com"
   97  ssh-keygen -t rsa -b 4096 -C "dpark.sports@gmail.com"
   98  eval "$(ssh-agent -s)"
   99  ssh-add ~/.ssh/id_rsa
  100  cat  ~/.ssh/id_rsa.pub
  101  cp -r /usr/src/cudnn_samples_v7/ $HOME
  102  gedit .bashrc
  103  ll /usr/local/cuda/lib64/libcudnn*
  104  gedit .bashrc
  105  cd Downloads/
  106  df 
  107  df .
  108  ls
  109  cd 
  110  tar -xf Downloads/CLion-2019.1.4.tar.gz 
  111  ls
  112  cd clion-2019.1.4/
  113  ls
  114  bash bin/clion.sh 
  115  df .
  116  cd
  117  history > history.log
  118  gedit history.log 
  119  gedit history2.log 
  120  df .
  121  cd Downloads/
  122  ll
  123  rm CLion-2019.1.4.tar.gz 
  124  ls
  125  ll
  126  rm opera-stable_62.0.3331.18_amd64.deb Python-3.7.3.tar.xz Miniconda3-latest-Linux-x86_64.sh cmake-3.14.5.tar.gz 
  127  ll
  128  rm cudnn-10.1-linux-x64-v7.6.1.34.tgz cudnn-10.1-linux-x64-v7.6.1.34.tgz.part 
  129  ll
  130  rm libcudnn7-d*
  131  ll
  132  df .
  133  cd
  134  ls
  135  df .
  136  sudo dpkg -i Downloads/code_1.35.1-1560350270_amd64.deb 
  137  code &
  138  g++ --version
  139  sudo apt install build-essential
  140  sudo apt install g++
  141  df .
  142  pwd
  143  df .
  144  ls
  145  cd data3-kitti/
  146  cd training/
  147  ls
  148  cd velodyne/
  149  ls 000000.bin
  150  gedit ~/.bashrc
  151  bg
  152  cd ..
  153  mkdir velodyne_reduced
  154  cd
  155  find . -iname "create_data.py" 
  156  find . -iname "create_data.py" 2> /dev/null
  157  shasum -a 256 Downloads/gcc-9.1.0.tar.xz 
  158  df .
  159  tar -xf Downloads/gcc-9.1.0.tar.xz 
  160  cd gcc-9.1.0/
  161  mkdir build && cd build
  162  ../configure 
  163  sudo iotop
  164  sudo iotops
  165  sudo apt-get install iotops
  166  sudo apt-get install iotop
  167  sudo iotop
  168  gedit ~/Documents/bash_history 
  169  mv ~/.bash_history ~/.bash_history_bak
  170  mv ~/Documents/bash_history ~/.bash_history
  171  history > ~/Documents/bash_history-numerated



```
