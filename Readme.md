#Instruction

- You can use my bash history as shown below.  
- You can see that it takes about 1700 steps to get there.
- I will come back and edit them for a better exploration.
- These steps are to setup the KITTI dataset or the NUScene dataset, train and evaluate a PointPillar network from scratch.
- Based on Ubuntu 16 with CUDA 10.1 on GTX 1060 Mobile.
- Training takes about 4 hours per epoch.


```plain
    sudo ufw enable
cd Downloads/
sudo dpkg -i cuda-repo-ubuntu1604_10.1.168-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
cd /etc/modprobe.d/
ls
sudo tee blacklist-ideapad.conf <<< "blacklist ideapad_laptop"
sudo tail blacklist-ideapad.conf 
sudo rfkill list
git
sudo apt-get update
sudo apt-get upgrade
cd Downloads/
cd ../Python-3.7.3/
make -j12
sudo make altinstall
python --version
python3 --version
python3.7 --version
cd
ls
exports
export
export|grep data
cd Downloads/
ll
sudo dpkg -i code_1.35.1-1560350270_amd64.deb 
/usr/local/bin/python3.7 -m pip install -U pylint --user
ls
ll
ls
mv nuscenes-devkit nutonomy/
l
cd nutonomy/
ll
cd ..
mv SparseConvNet nutonomy/
cd PointPillars nutonomy/
cd ..
mv PointPillars nutonomy/
ll
ls
cd nutonomy/
ll
cd nuscenes-devkit/
code .
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo docker pull nvidia/cuda:10.2-base
sudo docker pull nvidia/cuda:10.2-latest
sudo docker pull nvidia/cuda:10.1-base
sudo docker run --runtime=nvidia -rm nvidia/cuda:10.1-base nvidia-smi
sudo docker run --runtime=nvidia --rm nvidia/cuda:10.1-base nvidia-smi
sudo apt-get update
sudo apt-get install     apt-transport-https     ca-certificates     curl     gnupg-agent     software-properties-common
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo docker run hello-world
docker pull scrin/second-pytorch 
sudo docker pull scrin/second-pytorch 
ls
cd pretrained_models/
ls
cd ..
cd pretain-model-kitti/
ls
cd ..
cd pretain-model-kitti/
mv * ../pretrained_models/
mv pretrained_models_v1.5 ../pretrained_models/
mv pretrained_models_v1.5.zip ../pretrained_models/
cd ..
ll
mv pretain-model-kitti pretrained-kitti
cd pretrained-kitti/
ll
code .
mv pretrained_models_v1.5 ../
mv pretrained_models_v1.5.zip ../Downloads/
cd ..
rmdir pretrained-kitti/
ll
cd pretrained_models
ls
cd ..
rmdir pretrained_models
ll
history
ls -a
cat .bash_profile 
source .bash_profile 
export|grep data
ls
git cloen https://github.com/fferroni/PointPillars.git
git clone https://github.com/fferroni/PointPillars.git
cd Downloads/
ls
mv v1.0-mini.tgz ../data-nuscenes/
cd ../data-nuscenes/
tar -xf v1.0-mini.tgz 
ls
cd
git clone git@github.com:facebookresearch/SparseConvNet.git
git clone https://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet/
ll
gedit build.sh 
code .
bash build.sh 
ccccccevjkfkuhtjjfjrkuvkehfukulkcefvdddbkbdl
ccccccevjkfkveublrteetkichbblhkfeeduncvrteee
ccccccevjkfknvlvrcnkningnruktddljbvdhbbfghud
ccccccevjkfkkcnhjulhdjgvftjjijdbdrfunjctlnfk
cd
cd nutonomy/
ll
cd
ls 0a
ls -a
gedit .bash_profile 
cd nutonomy/second.pytorch/
pwd.
pwd
gedit ~/.bash_profile 
cd ..
ll
mv second.pytorch/ ../
mv SparseConvNet/ ../
mv nuscenes-devkit/ ../
mv PointPillars/ ../
cd ..
ll
ls
rmdir nutonomy
rm -fr PointPillars
cd second.pytorch/
cd second/
conda create -n second python=3.6
ls -a ~
source ~/activate_conda.sh 
conda create -n second python=3.6
conda activate second
conda install scikit-image scipy numba pillow matplotlib
pip install fire tensorboardX protobuf opencv-python
cd ..
cd
git clone https://github.com/traveller59/spconv.git --recursive
sudo apt-get install libboostall-dev
history|grep boost
sudo apt-get install libboost-all-dev
cmake --version
sudo apt-get install cmake
sudo apt-get remove cmake
cd Downloads/
ls
tar -xf cmake-3.14.5.tar.gz 
ll
mv cmake-3.14.5 ../
cd 
cd cmake-3.14.5/
ll
ls
gedit README.rst 
bg
./bootstrap 
ls
less README.rst 
make -j12
less README.rst 
sudo make install
cmake --version
conda list
conda list |less
conda install pytorch
history
conda install pytorch torchvision -c pytorch
conda install -c anaconda flash
conda install -c anaconda flask
cd
nvidia-docker run -it --rm -v /media/yy/960evo/datasets/:/root/data -v $HOME/pretrained_models:/root/model --ipc=host second-pytorch:latest
sudo nvidia-docker run -it --rm -v /media/yy/960evo/datasets/:/root/data -v $HOME/pretrained_models:/root/model --ipc=host second-pytorch:latest
history
sudo nvidia-docker run -it --rm -v /media/yy/960evo/datasets/:/root/data -v $HOME/pretrained_models:/root/model --ipc=host scrin/second-pytorch:latest
pwd
ls
sudo nvidia-docker run -it --rm -v ~/data-nuscenes/:/root/data -v $HOME/pretrained_models:/root/model --ipc=host scrin/second-pytorch:latest
sudo nvidia-docker run -it --rm -v ~/data-nuscenes/:/root/data -v $HOME/pretrained_models_v1.5:/root/model --ipc=host scrin/second-pytorch:latest
history
sudo nvidia-docker run -it --rm -v ~/data-nuscenes/:/root/data -v $HOME/pretrained_models_v1.5:/root/model --ipc=host scrin/second-pytorch:latest
pwd
ll
ls
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
cd second.pytorch/
ls
cd second/
ls
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
ls
ll configs/
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
ls
ls ..
code ..
nvcc -arch=sm_61
cd Downloads/
ls
tar -xf cudnn-10.1-linux-x64-v7.6.1.34.tgz 
ls
mv cuda cudnn
cd cudnn/
ls
code NVIDIA_SLA_cuDNN_Support.txt 
cd ..
ls
ll *.deb
ls *.deb
sudo dpkg -i libcudnn7-dev_7.6.1.34-1+cuda10.1_amd64.deb 
sudo dpkg -i libcudnn7_7.6.1.34-1+cuda10.1_amd64.deb 
sudo dpkg -i libcudnn7-doc_7.6.1.34-1+cuda10.1_amd64.deb 
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd  $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
ls
make
ll
cd ..
ls
ll
cd mnistCUDNN/
ll
make 
cd ..
ls
ll
code .
cd mnistCUDNN/
ls
ll
ake
make
ll
code .
cd
set
set |grep DNN
source activate_conda.sh 
conda activate pointpillars
cd spconv/
ls
python setup.py bdist_wheel
cd ~/Downloads/
ls
mv cudnn cuda
ll
ls
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
history
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
gedit ~/.bash_profile
bg
cd /usr/local/cuda/lib64/
ls
ls *cudnn*
cd
cd spconv/
ls
python setup.py bdist_wheel
ls
cd ./dist/
ls
ll
history|grep pip
pip install spconv-1.1-cp37-cp37m-linux_x86_64.whl --user
history
history|grep evaluate
history|grep python
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
cd ../../nutonomy/second.pytorch/second/
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
cd ..
ll
python ./second/pytorch/train.py evaluate --config_path=./second/configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
nvcc --version
sudo apt install nvidia-cuda-toolkit
nvidia-smi
cd Downloads/
ll
sudo apt-get install linux-headers-$(uname -r)
sudo dpkg -l nvidia
dpkg -l|grep nvidia
dpkg -l|grep cuda
cd /usr/local/cuda
ls
find . -iname "nvcc"
./bin/nvcc --version
echo $PATH
cd bin
ls nvcc
cd
ls
cd second.pytorch/
ls
nvcc -std=c++11 -c -o ../cc/nms/nms_kernel.cu.o ../cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
nvidia-smi
nvcc
/usr/local/cuda/bin/nvcc --help
/usr/local/cuda/bin/nvcc --resource-usage
nvclock
/usr/local/cuda/bin/nvclock
find /usr/local/cuda nvclock
lspci
nvidia-smi --help
nvidia-smi --help |grep compute
nvidia-smi clocks
nvidia-smi clocks -i
nvidia-smi -q
nvidia-smi -q |grep compute
nvidia-settings
find . -iname "nms*"
cd second/core
nvcc -std=c++11 -c -o ../cc/nms/nms_kernel.cu.o ../cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ../cc/nms/nms_kernel.cu.o ../cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
cd cc
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ../cc/nms/nms_kernel.cu.o ../cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
ls
ll
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ../cc/nms/nms_kernel.cu.o ../cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I~/anaconda3/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart 
cat ~/.bash_profile 
history
cat ~/.bash_profile 
history|grep evaluate
sudo apt-get update
sudo apt-get upgrade
sudo apt-get -f install
ls
cd data-nuscenes/
ll
code .
pwd
cd data-kitti/
ls
cd ~/Downloads/
ll *.zip
mv *.zip ~/data-kitti/
cd ~/data-kitti/
ls
ll
mkdir image2
mv data_object_image_2.zip image2/
cd image2/
ll
unzip data_object_image_2.zip 
ll
ll ..
mv testing/ ..
mv training/ ..
mv data_object_image_2.zip ~/Downloads/
cd ..
rmdir image2/
ll
unzip data_object_image_3.zip 
ls
mv data_object_image_3.zip ~/Downloads/
mkdir ~/kitti-originals
mv ~/*.zip ~/kitti-originals/
mv ~/Downloads/*.zip ~/kitti-originals/
ll ~/kitti-originals/
ll
history |grep unzip
mkdir velodyne
mv data_object_velodyne.zip velodyne/
cd velodyne/
unzip data_object_velodyne.zip 
ll
mv data_object_velodyne.zip ..
ll
cd ..
ll
mkdir det2
mv data_object_det_2.zip det2/
cd det2/
ll
unzip data_object_det_2.zip 
ll
mv data_object_det_2.zip ../
cd ..
rm -r det2 &
ls
unzip data_object_det_2.zip 
ls
mv data_object_det_2.zip ~/kitti-originals/
unzip data_object_calib.zip 
mv data_object_calib.zip ../kitti-originals/
ls
cd Downloads/
wget -c https://www.nuscenes.org/download#
cd Downloads/
ll
mv 1812.05784.pdf ../Documents/
ll *.tgz
rm *.tgz
ll
ll *.tgz
ll
mv pretrained_models_v1.5.zip ../nutonomy/
ll
cd
ls
mkdir pretain-model-kitti
cd pretain-model-kitti/
mv ../nutonomy/pretrained_models_v1.5.zip .
unzip pretrained_models_v1.5.zip 
ls
cd pretrained_models_v1.5/
ll
code .
cd
ls
source activate_conda.sh 
conda env list
conda activate second
ls
cd second.pytorch/
ls
cd second/
ls
python ./kittiviewer/backend/main.py main --port=8888
python ./kittiviewer/backend.py main --port=8888
code ..
python ./kittiviewer/backend/main.py main --port=8888
code ..
python ./kittiviewer/backend.py main --port=8888
cat ~/.bash_profile 
source ~/.bash_profile 
python ./kittiviewer/backend.py main --port=8888
history
code ..
cd
find . -iname "*.fhd.*"
find . -iname "*.fhd.config" 2>
find . -iname "*.fhd.*"
find . -iname "car*.config"
ls
code second.pytorch
conda list
cd second.pytorch/second/
ls
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
cd core/cc
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ../cc/nms/nms_kernel.cu.o ../cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I~/anaconda3/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
conda list
conda install shapely pybind11 protobuf scikit-image numba pillow
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ../cc/nms/nms_kernel.cu.o ../cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I~/anaconda3/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ../cc/nms/nms_kernel.cu.o ../cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ../cc/nms/nms_kernel.cu.o ../cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I~/anaconda3/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
python -c 'import pybind11; print(pybind11.get_include())
'
python -c 'import pybind11; print(pybind11.get_include())'
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ../cc/nms/nms_kernel.cu.o ../cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I~/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ../cc/nms/nms_kernel.cu.o ../cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
python -c 'import pybind11; print(pybind11.get_include())'
history
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
cd ../../
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
find . -iname "nms*"
/usr/local/cuda/bin/nvcc -arch=sm_61
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ../cc/nms/nms_kernel.cu.o ../cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
ll ./core/cc/nms/nms_kernel.cu.o 
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
/usr/local/cuda/bin/nvcc -arch=sm_61
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
ls
cd ..
ll
ls
ll /usr/local/cuda/lib64/
pwd
history|grep python
cd seon
cd second/
ls
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
history|grep python
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
cd
cd spconv/
ls
cd dist/
ls
pip install spconv-1.1-cp37-cp37m-linux_x86_64.whl --user
pip install spconv-1.1-cp37-cp37m-linux_x86_64.whl -user
cd ..
python setup.py bdist_wheel
cd dist/
ll
pip install spconv-1.1-cp36-cp36m-linux_x86_64.whl --user
cd
ls
cd second.pytorch/second/
ls
history|grep python
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
history|grep g++
history|grep nvcc
history|grep g++
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
cd ~/.conda/envs/second/include/
ls
cd ../../
ls
cd pointpillars/include/python3.7m/
cd ~/.local/
ls
find . -iname "python*
"
find . -iname "python*"
history|grep bind
python -c 'import pybind11; print(pybind11.get_include())
'
python -c 'import pybind11; print(pybind11.get_include())'
history|grep bind
cd
find . *.pkl
find . -iname "*.pkl"
ls
ll
ls
find second.pytorch -iname "*.pkl"
find second.pytorch/ -iname "*.pkl"
find data-nuscenes/ -iname "*.pkl"
find nutonomy/ -iname "*.pkl"
ls -R |grep pkl
ls -R |grep . pkl
ls -R |grep .pkl
ll -R |grep .pkl
find . -iname "*.pkl"
grep -r -n -e "xyres_16.proto" *
ls
code nuscenes-devkit
cd nuscenes-devkit/
find . -iname "*.pkl"
grep -r -n -e "kitti_info_val.pkl" *
grep -r -n -e ".pkl" *
grep -r -n -e "pkl" *
grep -r -n -e "pkl" * |less
grep -r -n -e "kitti" * 
code .
cd
ls
cd data-kitti/
ls
ll
unzip models_lsvm.zip 
ls
ll
mkdir models_lsvm
mv *.mat models_lsvm/
mv orientation.pdf readme.txt models_lsvm/
ll
mkdir devkit_object
mv devkit_object.zip devkit_object/
cd devkit_object/
ls
ll
unzip devkit_object.zip 
ll
gedit readme.txt 
cd ..
ls
ll
cd models_lsvm/
ll
gedit readme.txt 
open orientation.pdf 
view orientation.pdf 
code orientation.pdf 
cd ..
ls
ll
cd
ls
cd nuscenes-devkit/
cd
cd second.pytorch-nutonomy/
ls
cd second/
ls
python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
cd ..
find . -iname "ImageSets"
cd second/data/ImageSets/
ls
cd
cd second.pytorch-nutonomy/second/
ls ~
ls -a ~
gedit ~/.bash_profile 
cd
ls
code nuscenes-devkit
cd nuscenes-devkit/
ls
cd python-sdk/nuscenes/scripts/
ls
code README.md 
python --version
- python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/nusc_kitti
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-nuscenes
cd
ls
cd data-nuscenes/
ls
ll
pwd
ls
ll
code .
cp mini_train training
ls
rsync -ax mini_train training
ll
ls
cd training/
ll
cd ..
rm -r training
mv mini_train training
cd training/velodyne/
ls
ll
ls
code .
cd ~/data-kitti/
ll *.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_prev_3.zip
ls
unzip data_object_prev_3.zip 
mv data_object_prev_3.zip ~/kitti-originals/
ll *.zip
cd data-kitti/
ls
ll *.zip
ll ~/Downloads/*.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_det_2.zip
ll *.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_prev_2.zip
ls
mv data_object_prev_2.zip 
mv data_object_prev_2.zip ../kitti-originals/
cd Downloads/
ll *.tgz
wget -c https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval07_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=C2s09uTMLpgzdHHXaw6qZXPFkcg%3D&Expires=1561998206
df .
cd data-kitti/
ls
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
unzip data_object_velodyne.zip 
mv data_object_velodyne.zip ~/kitti-originals/
ls *.zip
ls
cd training/
ls
cd ..
find .. -iname "label_2"
cd ../data-nuscenes/training/label_2/
ls
cd ..
ls
cd ..
ls
cd kitti-originals/
ll
cd ../data-kitti/
wget -c http://cvlibs.net/download.php?file=data_object_label_2.zip
ls
ll
ls
rm download.php\?file\=data_object_label_2.zip 
ll
mv Geiger2011NIPS.pdf ~/Documents/
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
unzip data_object_label_2.zip 
mv data_object_label_2.zip ../kitti-originals/
sudo fdisk -l
sudo blkid
ll /media/computer/
sudo chgrp adm /media/computer/d1d53a2d-7bae-4372-9fb7-c454f107330e/
ll /media/computer/
sudo chmod g+w /media/computer/d1d53a2d-7bae-4372-9fb7-c454f107330e/
ll /media/computer/
df .
ls
ls nutonomy/
mv nutonomy/nuscenes-devkit/ .
ll
ls
mv nutonomy/second.pytorch second.pytorch-nutonomy
ls
ls nutonomy/
rmdir nutonomy
cd data-nuscenes/
ls
ll
ls
pwd
cd
ll
ls
cd nuscenes-devkit/
ls
source ../activate_conda.sh 
conda env list
conda activate pointpillars
ls
jupyter notebook 
jobs
ls
ls -a
source activate_conda.sh 
conda env list
conda activate nuscenes
sudo apt install python-pip
ls
cd nuscenes-devkit/
pip install -r setup/requirements.txt 
conda install pandas
conda install jupyter notebook
cd ..
ls
cd
ls
git clone https://github.com/nutonomy/second.pytorch.git
mkdir nutonomy
mv second.pytorch nutonomy/
cd nutonomy/
ls
cd second.pytorch/
ls
gedit README.md e
cd
conda create -n pointpillars python=3.7 anaconda
conda activate pointpillars
conda install shapely pybind11 protobuf scikit-image numba pillow
conda install pytorch torchvision -c pytorch
conda install google-sparsehash -c bioconda
pip install --upgrade pip
pip install fire tensorboardX
cd nutonomy/SparseConvNet/
ls
./build.sh
bash build.sh 
history
sudo apt-get install libboost-all-dev
history
bash build.sh
sudo apt-get install libboost-all-dev
cat ~/.bash_profile 
cd
cd SparseConvNet/
ls
cd
ls
codde nuscenes-devkit/
code nuscenes-devkit/
ls
mkdir nutonomy
cd nutonomy/
git clone https://github.com/nutonomy/second.pytorch.git
mv ../nuscenes-devkit/ .
ll
cd nuscenes-devkit/
conda list
cd ../second.pytorch/
ls
cd second/
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
g++ -std=c++11 -shared -o ~/second.pytorch/second/core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
cd
ls
cd spconv/
ls
cmake --version
python --version
ls
python setup.py bdist_wheel
ls
rm -fr build;mkdir build;cd build
cmake ..
lsb_release -a
cmake --version
cd ..
ls
code CMakeLists.txt 
rm -fr build;mkdir build;cd build
cmake ..
history
cd ..
python setup.py bdist_wheel
gedit ~/.bash_profile 
source ~/.bash_profile 
python setup.py bdist_wheel
which nvcc
python setup.py bdist_wheel
history
cd
ls
cd nutonomy/second.pytorch/second/
ls
history|grep evaluate
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
python --version
history|grep nvcc
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
history|grep g++
g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
ls
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
history|grep python
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
cd
ll
cd spconv/
ls
cd dist/
ls
pip install spconv-1.1-cp36-cp36m-linux_x86_64.whl --user
pip install spconv-1.1-cp37-cp37m-linux_x86_64.whl --user
history|grep g++
g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
cd ~/nutonomy/second.pytorch/second/
g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/pointpillars/include/python3.7m -I~/.local/lib/python3.7 -L/usr/local/cuda/lib64 -lcublas -lcudart
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
find . -iname "box*"
find . -iname "*.so"
find . -iname "*.o"
history|grep nvcc
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/box_ops.o ./core/cc/box_ops.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
python -c 'import pybind11; print(pybind11.get_include())'
history|grep bind
conda install shapely pybind11 protobuf scikit-image numba pillow
python -c 'import pybind11; print(pybind11.get_include())'
/usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/box_ops.o ./core/cc/box_ops.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
g++ -std=c++11 -shared -o ./core/cc/box_ops.so ./core/cc/box_ops.o ./core/cc/box_ops.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/pointpillars/include/python3.7m -I~/.local/lib/python3.7 -L/usr/local/cuda/lib64 -lcublas -lcudart
g++ -std=c++11 -shared -o ./core/cc/box_ops.so ./core/cc/box_ops.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/pointpillars/include/python3.7m -I~/.local/lib/python3.7 -L/usr/local/cuda/lib64 -lcublas -lcudart
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
grep -r -n -e "kitti_infos_val.pkl" *
history
grep -r -n -e "kitti_infos_val.pkl" *
ls
code .
code ..
df .
grep -r -n -e "kitti_infos_val.pkl" *
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
history
grep -r -n -e "kitti_infos_val.pkl" *
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
grep -r -n -e "kitti_infos_val.pkl" *
conda install cv2
conda install opencv2
conda install --verbose -c conda-forge opencv==3.4.6
conda install --verbose -c conda-forge opencv==3.4
conda install --verbose -c anaconda opencv==3.4.6
conda install --verbose -c anaconda opencv==3.4
conda install --verbose -c anaconda opencv==3.4.1
python --version
python
conda install pyquaternion
pip install pyquaternion
python
conda install cachetools
pip install cachetools Pillow
pip install scikit-learn scipy
pip install Shapely tqdm
pip install opencv-python numpy
pip install matplotlib jupyter 
pip install pyquaternion>=0.9.5
code '/home/computer/nuscenes-devkit/python-sdk/nuscenes/eval/detection/configs/cvpr_2019.json' 
python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
cd ..
python second/create_data.py create_kitti_info_file --data_path=~/data-nuscenes
source ~/.bash_profile 
cd second/
python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
ls
cd ..
python --version
python
source ~/.bash_profile 
python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
cd second/
python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
cd
cd nuscenes-devkit/
cd python-sdk/nuscenes/scripts/
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-nuscenes
grep -r -n -e "/data/sets/nuscenes/v1.0-mini" ~/nuscenes-devkit/
grep -r -n -e "v1.0-mini" ~/nuscenes-devkit/
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-nuscenes
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-kitti
ls
history
cd
ls
cd second.pytorch-nutonomy/second/
ls
history
python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
python create_data.py create_kitti_info_file --data_path=~/data-nuscenes/mini_train/
python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
ls
grep -r -n -e "000000" ../*
code data/ImageSets/
code data
ls
cd data/
ls
cd ..
cd configs/
ls
cd pointpillars/
ls
ll
cat README.md 
cd car/
ls
ll
gedit xyres_16.proto 
cd ../../../data/ImageSets/
ls
ll
cp train.txt train.txt-bk
ls ~/data-nuscenes/training/velodyne/ >> train.txt
tail train.txt-bk 
tail train.txt
gedit train.txt
ls
ll
ls ~/data-nuscenes/training/velodyne/ > train.txt
gedit train.txt
ls
history
cd ../../
python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
cd data/ImageSets/
ll
mv train.txt train.txt-mini
mv train.txt-bk train.txt
ll
cd ..
python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
history
python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
history
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-kitti
cd ~/nuscenes-devkit/python-sdk/nuscenes/scripts/
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-kitti
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-nuscenes/
cd ~/second.pytorch-nutonomy/second/
ls
history
python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
python create_data.py create_kitti_info_file --data_path=/home/computer/data-kitti/
python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
conda install -c numba/label/dev llvmlite
conda list|grep lite
conda install -c numba/label/dev llvmlite
python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
conda list|grep numba
python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
conda install numba=0.39
python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
history
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
history|grep evaluate
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
ls
rm "=0.9.5" 
ls
ls
df .
mkdir data2-kitti
cd data-kitti
ls
ll
mv *.zip ~/kitti-originals/
ll
cd ..
ls
rm -r data-kitti
cd data2-kitti/
mv ~/kitti-originals/*.zip .
ls
unzip *.zip
ls
unzip data_object_calib.zip
ls
unzip data_object_label_2.zip 
ls
unzip data_object_det_2.zip 
ls
unzip data_object_prev_2.zip 
ls
unzip data_object_prev_3.zip 
ls
history|grep unzip
ls
unzip data_object_velodyne.zip data_object_image_2.zip data_object_image_3.zip 
unzip data_object_velodyne.zip
unzip data_object_image_2.zip
unzip data_object_image_3.zip
ls
history|grep unzip
ls
df .
cd 
ls
git clone https://github.com/ApolloAuto/apollo.git
cd Downloads/
ls
ls *.tgz
ll *.tgz
df .
cd
ls
cd kitti-originals/
ll
mv ~/data2-kitti/*.zip .
ll
cd ..
pwd
cd /media/computer/
ls
ll
ls
cd 887a2604-a468-4ad6-9614-09ff6a9b1fab/
ls
cd
fdisk
fdisk list
sudo apt-get install gparted
sudo apt-get -f install
sudo apt-get install gparted
mount
mount |grep kitti
cd /media/computer/
ls
cd
sudo fdisk -l
sudo mkdir /media/data
sudo mount -t ext4 /dev/sdb /media/data -o uid=1000
sudo mkfs.ext4 /dev/sdb1
mount
cd /media/computer/
ls
cd d1d53a2d-7bae-4372-9fb7-c454f107330e/
ls
echo "" > hello
cd
history
sudo mount -t ext4 /dev/sdb /media/data -o uid=1000
ls
cd /media/data/
ls
echo "" > hello.txt
cd ../
ls
ll
cd computer/
ls
history
sudo fdisk -l
sudo mount -t ext4 /dev/sdb /data -o uid=1000
history 
mkdir /data
mdkir /media/computer/data
mkdir /media/computer/data
sudo mkdir /data
sudo mount -t ext4 /dev/sdb /data -o uid=1000
sudo mount -t ext4 /dev/sdb1 /data -o uid=1000
cd /data
ls
echo "" > hello.txt
rsync -ax /home/computer/kitti-originals/. .
rsync -ax /home/computer/kitti-originals/ .
mkdir kitti
cd kitti/
rsync -ax /home/computer/kitti-originals/ .
rsync -axv --process /home/computer/kitti-originals/ .
rsync -axv --progress /home/computer/kitti-originals/ .
cd ../nusc
ls
rsync -axv --progress /home/computer/Downloads/*.tgz .
cd
unmount
mount
umount /data
umount /ddev/sdb1
umount /dev/sdb1
mount
cd
mount
cd Downloads/
cd clion-2019.1.4/
cd bin
ls
./clion.sh 
cd
cd Downloads/
ls
shasum -a 256 CLion-2019.1.4.tar.gz 
history|grep cuda
history|grep cp |grep cuda
sudo cp cuda/include/cudnn.h /usr/local/cuda/include;sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
history|grep chmod
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
cd
ls
rm -r apollo &
fg
rm -fr apollo
git clone https://github.com/ApolloAuto/apollo.git
cd Downloads/
ls
sudo dpkg -i opera-stable_60.0.3255.170_amd64.deb 
sudo apt-get install chromium-browser
sudo apt-get purge firefox
sudo apt-get install firefox
ls *.tgz
ll *.tgz
wget -c https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval03_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2%2FWe9F9tkzhWjiFZQo%3D&Expires=1562049707
curl
curl --help
curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU%2BPgWr5zvjOkU%3D&Expires=1562016186
curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
ls
ll
sudo apt-get install axel
axel https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
wget https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
gedit
bg
jobs
curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E+3A+USljS2pZ9s=&Expires=1562049676
jobs
wget https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
wget https%3A//s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz%3FAWSAccessKeyId%3DAKIA6RIK4RRMFUKM7AM2%26Signature%3DWqUUUYtRT8X0E%252B3A%252BUSljS2pZ9s%253D%26Expires%3D1562049676%0A%0A
wget https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz%3FAWSAccessKeyId%3DAKIA6RIK4RRMFUKM7AM2%26Signature%3DWqUUUYtRT8X0E%252B3A%252BUSljS2pZ9s%253D%26Expires%3D1562049676%0A%0A
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676"
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU+PgWr5zvjOkU=&Expires=1562016186"
jobs
fg %2
jobs
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU%2BPgWr5zvjOkU%3D&Expires=1562016186"
ls *.t*
ll *.t*
history
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E+3A+USljS2pZ9s=&Expires=1562049676" -o v1.0-trainval04_blobs.tgz
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E+3A+USljS2pZ9s=&Expires=1562049676" 
history
wget "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676"
cd Downloads/
ll
ll *.tgz
mv v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676 v1.0-trainval04_blobs.tgz
ll *.tgz
ll S*
ll 
ls
cd
ls
cd /
ls
cd
ls
cd Downloads/
ls
history
ll *.t*
mv "v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676" v1.0-trainval04_blobs.tgz
ll *.t*
rm "v1.0-trainval04_blobs.tgz?AWSAccessKeyId=A*"
rm v1.0-trainval04_blobs.tgz?AWSAccessKeyId=A*
ll *.t*
mv "v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU%2BPgWr5zvjOkU%3D&Expires=1562016186" v1.0-trainval05_blobs.tgz
ll *.t*
rm "v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2" 
ll *.t*
ls
cd data-nuscenes/
ls
ll
cd
cd Downloads/
ll *.tgz
cd Downloads/
wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
df .
wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
ls
cd /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e
ls
mkdir kitii
cd kitii/
rsync -ax /home/computer/kitti-originals/ .
rsync -ax --progress /home/computer/kitti-originals/ .
ls
cd ..
mkdir nuscenes
cd nuscenes/
rsync -ax --progress /home/computer/data-nuscenes/v1.0-mini.tgz .
ls
rsync -ax --progress /home/computer/Downloads/*.tgz .
df .
ll
cd ..
ls
ll
cd kitii/
ls
cd ..
rm -r kitii/
cd nuscenes/
ll
ls
rsync -ax --progress /home/computer/Downloads/*.tgz .
rsync -ax --progress /home/computer/data-nuscenes/*.tgz .
df ~
df .
jbos
jobs
nvidia-settings
nvidia-smi
jobs
ll
jobs
rsync -ax --progress /home/computer/data-nuscenes/*.tgz .
rsync -ax --progress /home/computer/Downloads/*.tgz .
df .
ll
df .
df ~
nvidia-smi
df ~
ll
ls
rm 2.tgz 
rm 3.tgz 
rm v1.0-trainval04_blobs.tgz 
ll
ls
df .
ll
ls
df .
rsync -ax --progress /home/computer/data-nuscenes/v1.0-trainval04_blobs.tgz .
df .
ll
cd /media/computer/d1d53a2d-7bae-4372-9fb7-c454f107330e/
ls
cd nusc/
ls
ll
ls
rsync -ax --progress ../../f4e42898-25a5-4741-b235-9d3597483a8e/nuscenes/*.tgz .
rsync -ax --progress ~/Downloads/*.tgz .
ls
ll
cd ..
rsync -ax --progress /home/computer/epoch1_train_pointpillars .
ls
rsync -ax --progress /home/computer/epoch1_train_pointpillars .
ll
rm hello.txt 
rm -fr .Trash-1000/
ll
cd nusc/
ls
ll
ls
exit
ls
cd pretrained_models_v1.5/
ls
ll
code .
cd
ls
cd data2-kitti/
ls
cd
find . -iname "*.pkl"
ls
cd data2-kitti/
ls
cd ../second.pytorch-nutonomy/
ls
cd second/
ls
ll
cd data/
ls
cd ImageSets/
ls
cd ..
cd ImageSets/
ls
less train.txt
less train.txt-mini 
cd ..
ls
cat fire-trace.log 
ld
ls
tail 2nd-pass.log 
at 2nd-pass.log 
cat 2nd-pass.log 
ls
find ~ -iname "result*.pkl"
cd \~/pretrained_models_v1.5/
ls
ll
cd eval_results/
ls
cd step_0/
ls
python -mpickle result.pkl |less
python -mpickle result.pkl 
ll
unzip ~/Downloads/pickleViewer.zip 
ls
mv 445a2dc921fb23ecdff0-268e3aab9a6efd74fb0bc62e226e0df270a6050c pickleviewer
ls
mv pickleviewer/pickleViewer.py .
rmdir pickleviewer/
pythonn pickleViewer.py result.pkl 
python pickleViewer.py result.pkl 
less pickleViewer.py 
gedit pickleViewer.py 
python pickleViewer.py result.pkl 
code pickleViewer.py 
python pickleViewer.py result.pkl 
ls
cd ..
mv eval_results/step_0/pickleViewer.py ../../
cd ../..
ls
mv 2nd-pass.log pass2.log
mv first-pass.log pass1.log
ls
cd ..
find . -iname "*catter*"
grep -r -n -e "scatter" *
grep -r -n -e "PointPillarScatter" *
ls
code .
cd
ls
cd pretrained_models_v1.5/
ls
ll
cd eval_results/
ls
cd step_0/
ls
ll
cd
cd empty
ls
cd eval_results/step_0/
ls
ll
less 007480.txt 
cd
mkdir train_pointpillars
cd data2-kitti/
ll
cd training/
ls
mv velodyne_reduced velodyne
cd ..
ll
cd training/
ls
mv velodyne velodyne_reduced
ls
cd
ls
cd train_pointpillars/
ls
ll
ls eval_checkpoints/
ls summary/
ll
cd
find . -iname "*.tckpt"
cd train_pointpillars/
ll
ls summary/ results/ eval_checkpoints/
ll summary/ results/ eval_checkpoints/
ls
ll
df .
cd
cd Downloads/
ls
ll
cd ../Documents/
ll
ls
cd ..
ls
ll
ls
cd Videos/
ls
cd ..
ls
cd data-nuscenes/
ls
ll
rm v1.0-trainval10_blobs.tgz 
rm v1.0-trainval09_blobs.tgz 
rm v1.0-trainval08_blobs.tgz 
rm v1.0-trainval07_blobs.tgz 
ll
df .
cd
cd pretrained_pointpillars_nusc/
ll
cd eval_results/
ll
cd step_0/
ll
cd ../../
ls
ll
ll predict_test/
ll predict_test/step_0/
ls
ll
cd
ls
cd epoch1_train_pointpillars/
ll
ll -t
cd
ls
cd pretrained_pointpillars_nusc/
ls
ll
code .
ll
nvidia_smi
nvidia-smi
exit
cd Downloads/
wget -O 1.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval01_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=cDQZCIAhrgr/RzB/B1s7nN2MZgA=&Expires=1562137463"
df .
wget -O 1.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval01_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=cDQZCIAhrgr/RzB/B1s7nN2MZgA=&Expires=1562137463"
history|grep tgz
df .
ll
ll *.tgz
df .
history|grep tgz
ll *.tgz
df .
ll *.tgz
df .
exit
history
wget "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676"
ll *.t*
mv v1.0-trainval04_blobs.tgz\?AWSAccessKeyId\=AKIA6RIK4RRMFUKM7AM2\&Signature\=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D\&Expires\=1562049676 Downloads/
ll *.t*
cd Downloads/
ll *.t*
history
ll *.t*
wget --help
wget -O 3.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval03_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2/We9F9tkzhWjiFZQo=&Expires=1562049707"
wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
wget -O test.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-test_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HHkgw0OT5z6kYHhEFFRfQgd9EbE=&Expires=1562140025"
wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
ll *.tgz
wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
bg
fg
wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
fg
history|grep tgz
wget -O 3.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval03_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2/We9F9tkzhWjiFZQo=&Expires=1562049707"
history|grep tgz
exit
jobs
ll *.tgz
kill %1
jobs
ll *.tgz
rm 6.tgz 
nautilus .
ll *.tgz
df . .
df .
ll
ll *.tgz
history|grep tgz
wget -O 3.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2/We9F9tkzhWjiFZQo=&Expires=1562049707"
wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
find . -iname "*.pkl"
find . -iname "kitti*.pkl"
cd data2-kitti/
ls
cd training/
ls
cd ..
code .
cd ..
find . -iname "config_tool.py"
find . -iname "config*tool.py"
find . -iname "*tool.py"
find . -iname "config.py"
find . -iname "config_.py"
find . -iname "config_*.py"
ls
cd second.pytorch-nutonomy/
find . -iname "config_*.py"
find . -iname "*tool.py"
find . -iname "config*tool.py"
cd
ls
cd nuscenes-devkit/
find . -iname "*tool.py"
cd
data2-kitti/
cd data2-kitti/
ls
cd training/
ls
ll
mv velodyne velodyne_reduced
nvidia-smi
firefox 
nvidia-smi
gparted
sudo gparted
history|grep chgrp
ls /media/computer/
sudo chgrpu adm /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
sudo chgrp adm /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
history |less
history|grep chgrp
history --help
history -d 700
history
history |less
sudo chmod g+w /media/computer/d1d53a2d-7bae-4372-9fb7-c454f107330e/
sudo chmod g+w /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
ls
cd ..
ls
cd
ls
cd kitti-originals/
ls
unzip models_lsvm.zip 
ls
mkdir models_lsvm
mv *.mat models_lsvm/
mv readme.txt orientation.pdf models_lsvm/
ls
cd models_lsvm/
ls
code .
ls
source activate_conda.sh 
conda env list
conda activate pointpillars
ls -a
cat .bash_profile 
source .bash_profile 
gedit .bash_profile 
source .bash_profile 
cd second.pytorch-nutonomy/
ls
cd second/
ls
history|grep python
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
python --version
pip --help
pip list
pip install -U protobuf
python --version
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
ls
code .
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
ls
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --batch_size=1
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --batch_size=1 > first-pass.log
gedit fire-trace.log
ls
code fire-trace.log first-pass.log 
tail first-pass.log 
cat first-pass.log 
ls
history
cd
ll
ls
history
ls
cd data-nuscenes/
ls
cd v1.0-mini/
code .
cd
mkdir mini-data-nuscenes
cd data-nuscenes/v1.0-mini/
ls
mv * ~/mini-data-nuscenes/
ls
cd ..
ls
rmdir v1.0-mini
cd v1.0-mini/
ll
cat .v1.0-mini.txt 
cd ..
rm -r v1.0-mini
ls
mv v1.0-mini.tgz ~/mini-data-nuscenes/
cd ../mini-data-nuscenes/
ls
cd sweeps/
ls
cd ..
ll
code .
cd 
cd data-nuscenes/
ls
ll ~/Downloads/*.tgz
mv ~/Downloads/*.tgz .
ll
tar -xf v1.0-trainval_meta.tgz 
ls
code . 
ll
df .
tar -xf v1.0-trainval10_blobs.tgz 
ls
jobs
history|grep python
cd ../second.pytorch-nutonomy/second/
ls
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --batch_size=1 > first-pass.log
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/  2> 2nd-pass.log
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/  > 2nd-pass.log
jobs
fg
ls
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_models_v1.5/
ls
jobs
fg
df .
ls
cd
ls
pp
ll
ls
ls data2-kitti/
ls kitti-originals/
cd kitti-originals/models_lsvm/
ls
code .
cd ..
mv models_lsvm ../models_lsvm-kitti
mv models_lsvm.zip ../models_lsvm-kitti/
cd
ls
cd models_lsvm-kitti/
ls
ll
ls
history|grep python
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_models_v1.5/
cd 
cd second.pytorch-nutonomy/second/
history|grep eval
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/models_lsvm-kitti/
cd
ls
mkdir empty
cd second.pytorch-nutonomy/second/
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/empty/
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/empty/ --pickle_result=False
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/train_pointpillars
history|grep create
python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
python create_data.py create_groundtruth_database --data_path=/home/computer/data2-kitti/
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/train_pointpillars
history|grep python
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/empty/ --pickle_result=False
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False --predict_test=True
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False --ckpt_path=/home/computer/pretrained_pointpillars_nusc/
python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False --ckpt_path=/home/computer/pretrained_pointpillars_nusc/voxelnet-296960.tckpt 
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/
fg
ls
find . -iname "*.tckpt"
mkdir pretrained_pointpillars_nusc
cp pretrained_models_v1.5/pp_model_for_nuscenes_pretrain/voxelnet-296960.tckpt pretrained_pointpillars_nusc/
cd pretrained_pointpillars_nusc/
ll
cd
find . -iname "*.tckpt"
df .
cd train_pointpillars/
ls
ll
ll eval_checkpoints/
cat pipeline.config 
ll
ll summary/
ll summary/loss/
ll summary/loss/loc_elem/
ll summary/loss/loc_elem/0/
ll
ll results/
ll results/step_18560/
cat checkpoints.json 
ll
ll -tl
ls
cd ..
mv train_pointpillars epoch1_train_pointpillars
ls
cd epoch1_train_pointpillars/
ll
cd ..
ls
cd second.pytorch-nutonomy/
find . -iname "*.config"
cd ../second.pytorch
find . -iname "*.config"
cd ../nuscenes-devkit/
find . -iname "*.config"
find . -iname "*.proto"
cd ../second.pytorch-nutonomy/
find . -iname "*.proto"
cd ../second.pytorch
find . -iname "*.proto"
cd 
find . -iname "*.proto"
ls
cd /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
ll
cd nuscenes/
ll
ls
df .
rsync -ax --progress /home/computer/Downloads/1.tgz .
df .
ll
ls
lls
ls
df .
rsync v1.0-trainval04_blobs.tgz /home/computer/Downloads/
rsync -ax --progress v1.0-trainval04_blobs.tgz /home/computer/Downloads/
ls
rm v1.0-trainval04_blobs.tgz 
df .
rsync /home/computer/Downloads/v1.0-trainval06_blobs.tgz .
rsync -ax --progress /home/computer/Downloads/v1.0-trainval06_blobs.tgz .
ls
rsync -ax v1.0-trainval05_blobs.tgz /home/computer/Downloads/
rsync -ax --progress v1.0-trainval05_blobs.tgz /home/computer/Downloads/
ls
ll
cd Downloads/
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
wget -O 6.tgz -c https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628
wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
history|grep wget
history|grep trainval06
wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=%2BwaO%2FfrI%2B8TxCB5wOfg9F12uVRk%3D&Expires=1562204679" -O 62.tgz


```
