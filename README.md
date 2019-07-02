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
    820  python setup.py bdist_wheel
  821  which nvcc
  822  python setup.py bdist_wheel
  823  history
  824  cd
  825  ls
  826  cd nutonomy/second.pytorch/second/
  827  ls
  828  history|grep evaluate
  829  python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  830  python --version
  831  history|grep nvcc
  832  /usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
  833  history|grep g++
  834  g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
  835  ls
  836  python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  837  history|grep python
  838  python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  839  cd
  840  ll
  841  cd spconv/
  842  ls
  843  cd dist/
  844  ls
  845  pip install spconv-1.1-cp36-cp36m-linux_x86_64.whl --user
  846  pip install spconv-1.1-cp37-cp37m-linux_x86_64.whl --user
  847  history|grep g++
  848  g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
  849  cd ~/nutonomy/second.pytorch/second/
  850  g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/second/include/python3.6m -I~/.local/include/python3.6m -L/usr/local/cuda/lib64 -lcublas -lcudart
  851  g++ -std=c++11 -shared -o ./core/non_max_suppression/nms.so ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/pointpillars/include/python3.7m -I~/.local/lib/python3.7 -L/usr/local/cuda/lib64 -lcublas -lcudart
  852  python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  853  find . -iname "box*"
  854  find . -iname "*.so"
  855  find . -iname "*.o"
  856  history|grep nvcc
  857  /usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/nms/nms_kernel.cu.o ./core/cc/nms/nms_kernel.cu.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
  858  /usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/box_ops.o ./core/cc/box_ops.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
  859  python -c 'import pybind11; print(pybind11.get_include())'
  860  history|grep bind
  861  conda install shapely pybind11 protobuf scikit-image numba pillow
  862  python -c 'import pybind11; print(pybind11.get_include())'
  863  /usr/local/cuda/bin/nvcc -std=c++11 -c -o ./core/cc/box_ops.o ./core/cc/box_ops.cc -I/usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_61 --expt-relaxed-constexpr 
  864  g++ -std=c++11 -shared -o ./core/cc/box_ops.so ./core/cc/box_ops.o ./core/cc/box_ops.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/pointpillars/include/python3.7m -I~/.local/lib/python3.7 -L/usr/local/cuda/lib64 -lcublas -lcudart
  865  g++ -std=c++11 -shared -o ./core/cc/box_ops.so ./core/cc/box_ops.cc -I/usr/local/cuda/include -fPIC -O3 -I$HOME/.conda/envs/pointpillars/include/python3.7m -I~/.local/lib/python3.7 -L/usr/local/cuda/lib64 -lcublas -lcudart
  866  python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  867  grep -r -n -e "kitti_infos_val.pkl" *
  868  history
  869  grep -r -n -e "kitti_infos_val.pkl" *
  870  ls
  871  code .
  872  code ..
  873  df .
  874  grep -r -n -e "kitti_infos_val.pkl" *
  875  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  876  history
  877  grep -r -n -e "kitti_infos_val.pkl" *
  878  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  879  grep -r -n -e "kitti_infos_val.pkl" *
  880  conda install cv2
  881  conda install opencv2
  882  conda install --verbose -c conda-forge opencv==3.4.6
  883  conda install --verbose -c conda-forge opencv==3.4
  884  conda install --verbose -c anaconda opencv==3.4.6
  885  conda install --verbose -c anaconda opencv==3.4
  886  conda install --verbose -c anaconda opencv==3.4.1
  887  python --version
  888  python
  889  conda install pyquaternion
  890  pip install pyquaternion
  891  python
  892  conda install cachetools
  893  pip install cachetools Pillow
  894  pip install scikit-learn scipy
  895  pip install Shapely tqdm
  896  pip install opencv-python numpy
  897  pip install matplotlib jupyter 
  898  pip install pyquaternion>=0.9.5
  899  code '/home/computer/nuscenes-devkit/python-sdk/nuscenes/eval/detection/configs/cvpr_2019.json' 
  900  python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
  901  cd ..
  902  python second/create_data.py create_kitti_info_file --data_path=~/data-nuscenes
  903  source ~/.bash_profile 
  904  cd second/
  905  python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
  906  ls
  907  cd ..
  908  python --version
  909  python
  910  source ~/.bash_profile 
  911  python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
  912  cd second/
  913  python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
  914  cd
  915  cd nuscenes-devkit/
  916  cd python-sdk/nuscenes/scripts/
  917  python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-nuscenes
  918  grep -r -n -e "/data/sets/nuscenes/v1.0-mini" ~/nuscenes-devkit/
  919  grep -r -n -e "v1.0-mini" ~/nuscenes-devkit/
  920  python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-nuscenes
  921  python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-kitti
  922  ls
  923  history
  924  cd
  925  ls
  926  cd second.pytorch-nutonomy/second/
  927  ls
  928  history
  929  python create_data.py create_kitti_info_file --data_path=~/data-nuscenes
  930  python create_data.py create_kitti_info_file --data_path=~/data-nuscenes/mini_train/
  931  python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
  932  ls
  933  grep -r -n -e "000000" ../*
  934  code data/ImageSets/
  935  code data
  936  ls
  937  cd data/
  938  ls
  939  cd ..
  940  cd configs/
  941  ls
  942  cd pointpillars/
  943  ls
  944  ll
  945  cat README.md 
  946  cd car/
  947  ls
  948  ll
  949  gedit xyres_16.proto 
  950  cd ../../../data/ImageSets/
  951  ls
  952  ll
  953  cp train.txt train.txt-bk
  954  ls ~/data-nuscenes/training/velodyne/ >> train.txt
  955  tail train.txt-bk 
  956  tail train.txt
  957  gedit train.txt
  958  ls
  959  ll
  960  ls ~/data-nuscenes/training/velodyne/ > train.txt
  961  gedit train.txt
  962  ls
  963  history
  964  cd ../../
  965  python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
  966  cd data/ImageSets/
  967  ll
  968  mv train.txt train.txt-mini
  969  mv train.txt-bk train.txt
  970  ll
  971  cd ..
  972  python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
  973  history
  974  python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
  975  history
  976  python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-kitti
  977  cd ~/nuscenes-devkit/python-sdk/nuscenes/scripts/
  978  python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-kitti
  979  python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/data-nuscenes/
  980  cd ~/second.pytorch-nutonomy/second/
  981  ls
  982  history
  983  python create_data.py create_kitti_info_file --data_path=/home/computer/data-nuscenes/
  984  python create_data.py create_kitti_info_file --data_path=/home/computer/data-kitti/
  985  python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
  986  conda install -c numba/label/dev llvmlite
  987  conda list|grep lite
  988  conda install -c numba/label/dev llvmlite
  989  python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
  990  conda list|grep numba
  991  python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
  992  conda install numba=0.39
  993  python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
  994  history
  995  python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  996  history|grep evaluate
  997  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
  998  ls
  999  rm "=0.9.5" 
 1000  ls
 1001  ls
 1002  df .
 1003  mkdir data2-kitti
 1004  cd data-kitti
 1005  ls
 1006  ll
 1007  mv *.zip ~/kitti-originals/
 1008  ll
 1009  cd ..
 1010  ls
 1011  rm -r data-kitti
 1012  cd data2-kitti/
 1013  mv ~/kitti-originals/*.zip .
 1014  ls
 1015  unzip *.zip
 1016  ls
 1017  unzip data_object_calib.zip
 1018  ls
 1019  unzip data_object_label_2.zip 
 1020  ls
 1021  unzip data_object_det_2.zip 
 1022  ls
 1023  unzip data_object_prev_2.zip 
 1024  ls
 1025  unzip data_object_prev_3.zip 
 1026  ls
 1027  history|grep unzip
 1028  ls
 1029  unzip data_object_velodyne.zip data_object_image_2.zip data_object_image_3.zip 
 1030  unzip data_object_velodyne.zip
 1031  unzip data_object_image_2.zip
 1032  unzip data_object_image_3.zip
 1033  ls
 1034  history|grep unzip
 1035  ls
 1036  df .
 1037  cd 
 1038  ls
 1039  git clone https://github.com/ApolloAuto/apollo.git
 1040  cd Downloads/
 1041  ls
 1042  ls *.tgz
 1043  ll *.tgz
 1044  df .
 1045  cd
 1046  ls
 1047  cd kitti-originals/
 1048  ll
 1049  mv ~/data2-kitti/*.zip .
 1050  ll
 1051  cd ..
 1052  pwd
 1053  cd /media/computer/
 1054  ls
 1055  ll
 1056  ls
 1057  cd 887a2604-a468-4ad6-9614-09ff6a9b1fab/
 1058  ls
 1059  cd
 1060  fdisk
 1061  fdisk list
 1062  sudo apt-get install gparted
 1063  sudo apt-get -f install
 1064  sudo apt-get install gparted
 1065  mount
 1066  mount |grep kitti
 1067  cd /media/computer/
 1068  ls
 1069  cd
 1070  sudo fdisk -l
 1071  sudo mkdir /media/data
 1072  sudo mount -t ext4 /dev/sdb /media/data -o uid=1000
 1073  sudo mkfs.ext4 /dev/sdb1
 1074  mount
 1075  cd /media/computer/
 1076  ls
 1077  cd d1d53a2d-7bae-4372-9fb7-c454f107330e/
 1078  ls
 1079  echo "" > hello
 1080  cd
 1081  history
 1082  sudo mount -t ext4 /dev/sdb /media/data -o uid=1000
 1083  ls
 1084  cd /media/data/
 1085  ls
 1086  echo "" > hello.txt
 1087  cd ../
 1088  ls
 1089  ll
 1090  cd computer/
 1091  ls
 1092  history
 1093  sudo fdisk -l
 1094  sudo mount -t ext4 /dev/sdb /data -o uid=1000
 1095  history 
 1096  mkdir /data
 1097  mdkir /media/computer/data
 1098  mkdir /media/computer/data
 1099  sudo mkdir /data
 1100  sudo mount -t ext4 /dev/sdb /data -o uid=1000
 1101  sudo mount -t ext4 /dev/sdb1 /data -o uid=1000
 1102  cd /data
 1103  ls
 1104  echo "" > hello.txt
 1105  rsync -ax /home/computer/kitti-originals/. .
 1106  rsync -ax /home/computer/kitti-originals/ .
 1107  mkdir kitti
 1108  cd kitti/
 1109  rsync -ax /home/computer/kitti-originals/ .
 1110  rsync -axv --process /home/computer/kitti-originals/ .
 1111  rsync -axv --progress /home/computer/kitti-originals/ .
 1112  cd ../nusc
 1113  ls
 1114  rsync -axv --progress /home/computer/Downloads/*.tgz .
 1115  cd
 1116  unmount
 1117  mount
 1118  umount /data
 1119  umount /ddev/sdb1
 1120  umount /dev/sdb1
 1121  mount
 1122  cd
 1123  mount
 1124  cd Downloads/
 1125  cd clion-2019.1.4/
 1126  cd bin
 1127  ls
 1128  ./clion.sh 
 1129  cd
 1130  cd Downloads/
 1131  ls
 1132  shasum -a 256 CLion-2019.1.4.tar.gz 
 1133  history|grep cuda
 1134  history|grep cp |grep cuda
 1135  sudo cp cuda/include/cudnn.h /usr/local/cuda/include;sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
 1136  history|grep chmod
 1137  sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
 1138  cd
 1139  ls
 1140  rm -r apollo &
 1141  fg
 1142  rm -fr apollo
 1143  git clone https://github.com/ApolloAuto/apollo.git
 1144  cd Downloads/
 1145  ls
 1146  sudo dpkg -i opera-stable_60.0.3255.170_amd64.deb 
 1147  sudo apt-get install chromium-browser
 1148  sudo apt-get purge firefox
 1149  sudo apt-get install firefox
 1150  ls *.tgz
 1151  ll *.tgz
 1152  wget -c https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval03_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2%2FWe9F9tkzhWjiFZQo%3D&Expires=1562049707
 1153  curl
 1154  curl --help
 1155  curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU%2BPgWr5zvjOkU%3D&Expires=1562016186
 1156  curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
 1157  ls
 1158  ll
 1159  sudo apt-get install axel
 1160  axel https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
 1161  curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
 1162  wget https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
 1163  gedit
 1164  bg
 1165  jobs
 1166  curl -O https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E+3A+USljS2pZ9s=&Expires=1562049676
 1167  jobs
 1168  wget https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676
 1169  wget https%3A//s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz%3FAWSAccessKeyId%3DAKIA6RIK4RRMFUKM7AM2%26Signature%3DWqUUUYtRT8X0E%252B3A%252BUSljS2pZ9s%253D%26Expires%3D1562049676%0A%0A
 1170  wget https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz%3FAWSAccessKeyId%3DAKIA6RIK4RRMFUKM7AM2%26Signature%3DWqUUUYtRT8X0E%252B3A%252BUSljS2pZ9s%253D%26Expires%3D1562049676%0A%0A
 1171  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676"
 1172  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU+PgWr5zvjOkU=&Expires=1562016186"
 1173  jobs
 1174  fg %2
 1175  jobs
 1176  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU%2BPgWr5zvjOkU%3D&Expires=1562016186"
 1177  ls *.t*
 1178  ll *.t*
 1179  history
 1180  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E+3A+USljS2pZ9s=&Expires=1562049676" -o v1.0-trainval04_blobs.tgz
 1181  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E+3A+USljS2pZ9s=&Expires=1562049676" 
 1182  history
 1183  wget "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676"
 1184  cd Downloads/
 1185  ll
 1186  ll *.tgz
 1187  mv v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676 v1.0-trainval04_blobs.tgz
 1188  ll *.tgz
 1189  ll S*
 1190  ll 
 1191  ls
 1192  cd
 1193  ls
 1194  cd /
 1195  ls
 1196  cd
 1197  ls
 1198  cd Downloads/
 1199  ls
 1200  history
 1201  ll *.t*
 1202  mv "v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676" v1.0-trainval04_blobs.tgz
 1203  ll *.t*
 1204  rm "v1.0-trainval04_blobs.tgz?AWSAccessKeyId=A*"
 1205  rm v1.0-trainval04_blobs.tgz?AWSAccessKeyId=A*
 1206  ll *.t*
 1207  mv "v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=MYKzIlYNu1BJ6aU%2BPgWr5zvjOkU%3D&Expires=1562016186" v1.0-trainval05_blobs.tgz
 1208  ll *.t*
 1209  rm "v1.0-trainval05_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2" 
 1210  ll *.t*
 1211  ls
 1212  cd data-nuscenes/
 1213  ls
 1214  ll
 1215  cd
 1216  cd Downloads/
 1217  ll *.tgz
 1218  cd Downloads/
 1219  wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1220  df .
 1221  wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1222  ls
 1223  cd /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e
 1224  ls
 1225  mkdir kitii
 1226  cd kitii/
 1227  rsync -ax /home/computer/kitti-originals/ .
 1228  rsync -ax --progress /home/computer/kitti-originals/ .
 1229  ls
 1230  cd ..
 1231  mkdir nuscenes
 1232  cd nuscenes/
 1233  rsync -ax --progress /home/computer/data-nuscenes/v1.0-mini.tgz .
 1234  ls
 1235  rsync -ax --progress /home/computer/Downloads/*.tgz .
 1236  df .
 1237  ll
 1238  cd ..
 1239  ls
 1240  ll
 1241  cd kitii/
 1242  ls
 1243  cd ..
 1244  rm -r kitii/
 1245  cd nuscenes/
 1246  ll
 1247  ls
 1248  rsync -ax --progress /home/computer/Downloads/*.tgz .
 1249  rsync -ax --progress /home/computer/data-nuscenes/*.tgz .
 1250  df ~
 1251  df .
 1252  jbos
 1253  jobs
 1254  nvidia-settings
 1255  nvidia-smi
 1256  jobs
 1257  ll
 1258  jobs
 1259  rsync -ax --progress /home/computer/data-nuscenes/*.tgz .
 1260  rsync -ax --progress /home/computer/Downloads/*.tgz .
 1261  df .
 1262  ll
 1263  df .
 1264  df ~
 1265  nvidia-smi
 1266  df ~
 1267  ll
 1268  ls
 1269  rm 2.tgz 
 1270  rm 3.tgz 
 1271  rm v1.0-trainval04_blobs.tgz 
 1272  ll
 1273  ls
 1274  df .
 1275  ll
 1276  ls
 1277  df .
 1278  rsync -ax --progress /home/computer/data-nuscenes/v1.0-trainval04_blobs.tgz .
 1279  df .
 1280  ll
 1281  cd /media/computer/d1d53a2d-7bae-4372-9fb7-c454f107330e/
 1282  ls
 1283  cd nusc/
 1284  ls
 1285  ll
 1286  ls
 1287  rsync -ax --progress ../../f4e42898-25a5-4741-b235-9d3597483a8e/nuscenes/*.tgz .
 1288  rsync -ax --progress ~/Downloads/*.tgz .
 1289  ls
 1290  ll
 1291  cd ..
 1292  rsync -ax --progress /home/computer/epoch1_train_pointpillars .
 1293  ls
 1294  rsync -ax --progress /home/computer/epoch1_train_pointpillars .
 1295  ll
 1296  rm hello.txt 
 1297  rm -fr .Trash-1000/
 1298  ll
 1299  cd nusc/
 1300  ls
 1301  ll
 1302  ls
 1303  exit
 1304  ls
 1305  cd pretrained_models_v1.5/
 1306  ls
 1307  ll
 1308  code .
 1309  cd
 1310  ls
 1311  cd data2-kitti/
 1312  ls
 1313  cd
 1314  find . -iname "*.pkl"
 1315  ls
 1316  cd data2-kitti/
 1317  ls
 1318  cd ../second.pytorch-nutonomy/
 1319  ls
 1320  cd second/
 1321  ls
 1322  ll
 1323  cd data/
 1324  ls
 1325  cd ImageSets/
 1326  ls
 1327  cd ..
 1328  cd ImageSets/
 1329  ls
 1330  less train.txt
 1331  less train.txt-mini 
 1332  cd ..
 1333  ls
 1334  cat fire-trace.log 
 1335  ld
 1336  ls
 1337  tail 2nd-pass.log 
 1338  at 2nd-pass.log 
 1339  cat 2nd-pass.log 
 1340  ls
 1341  find ~ -iname "result*.pkl"
 1342  cd \~/pretrained_models_v1.5/
 1343  ls
 1344  ll
 1345  cd eval_results/
 1346  ls
 1347  cd step_0/
 1348  ls
 1349  python -mpickle result.pkl |less
 1350  python -mpickle result.pkl 
 1351  ll
 1352  unzip ~/Downloads/pickleViewer.zip 
 1353  ls
 1354  mv 445a2dc921fb23ecdff0-268e3aab9a6efd74fb0bc62e226e0df270a6050c pickleviewer
 1355  ls
 1356  mv pickleviewer/pickleViewer.py .
 1357  rmdir pickleviewer/
 1358  pythonn pickleViewer.py result.pkl 
 1359  python pickleViewer.py result.pkl 
 1360  less pickleViewer.py 
 1361  gedit pickleViewer.py 
 1362  python pickleViewer.py result.pkl 
 1363  code pickleViewer.py 
 1364  python pickleViewer.py result.pkl 
 1365  ls
 1366  cd ..
 1367  mv eval_results/step_0/pickleViewer.py ../../
 1368  cd ../..
 1369  ls
 1370  mv 2nd-pass.log pass2.log
 1371  mv first-pass.log pass1.log
 1372  ls
 1373  cd ..
 1374  find . -iname "*catter*"
 1375  grep -r -n -e "scatter" *
 1376  grep -r -n -e "PointPillarScatter" *
 1377  ls
 1378  code .
 1379  cd
 1380  ls
 1381  cd pretrained_models_v1.5/
 1382  ls
 1383  ll
 1384  cd eval_results/
 1385  ls
 1386  cd step_0/
 1387  ls
 1388  ll
 1389  cd
 1390  cd empty
 1391  ls
 1392  cd eval_results/step_0/
 1393  ls
 1394  ll
 1395  less 007480.txt 
 1396  cd
 1397  mkdir train_pointpillars
 1398  cd data2-kitti/
 1399  ll
 1400  cd training/
 1401  ls
 1402  mv velodyne_reduced velodyne
 1403  cd ..
 1404  ll
 1405  cd training/
 1406  ls
 1407  mv velodyne velodyne_reduced
 1408  ls
 1409  cd
 1410  ls
 1411  cd train_pointpillars/
 1412  ls
 1413  ll
 1414  ls eval_checkpoints/
 1415  ls summary/
 1416  ll
 1417  cd
 1418  find . -iname "*.tckpt"
 1419  cd train_pointpillars/
 1420  ll
 1421  ls summary/ results/ eval_checkpoints/
 1422  ll summary/ results/ eval_checkpoints/
 1423  ls
 1424  ll
 1425  df .
 1426  cd
 1427  cd Downloads/
 1428  ls
 1429  ll
 1430  cd ../Documents/
 1431  ll
 1432  ls
 1433  cd ..
 1434  ls
 1435  ll
 1436  ls
 1437  cd Videos/
 1438  ls
 1439  cd ..
 1440  ls
 1441  cd data-nuscenes/
 1442  ls
 1443  ll
 1444  rm v1.0-trainval10_blobs.tgz 
 1445  rm v1.0-trainval09_blobs.tgz 
 1446  rm v1.0-trainval08_blobs.tgz 
 1447  rm v1.0-trainval07_blobs.tgz 
 1448  ll
 1449  df .
 1450  cd
 1451  cd pretrained_pointpillars_nusc/
 1452  ll
 1453  cd eval_results/
 1454  ll
 1455  cd step_0/
 1456  ll
 1457  cd ../../
 1458  ls
 1459  ll
 1460  ll predict_test/
 1461  ll predict_test/step_0/
 1462  ls
 1463  ll
 1464  cd
 1465  ls
 1466  cd epoch1_train_pointpillars/
 1467  ll
 1468  ll -t
 1469  cd
 1470  ls
 1471  cd pretrained_pointpillars_nusc/
 1472  ls
 1473  ll
 1474  code .
 1475  ll
 1476  nvidia_smi
 1477  nvidia-smi
 1478  exit
 1479  cd Downloads/
 1480  wget -O 1.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval01_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=cDQZCIAhrgr/RzB/B1s7nN2MZgA=&Expires=1562137463"
 1481  df .
 1482  wget -O 1.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval01_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=cDQZCIAhrgr/RzB/B1s7nN2MZgA=&Expires=1562137463"
 1483  history|grep tgz
 1484  df .
 1485  ll
 1486  ll *.tgz
 1487  df .
 1488  history|grep tgz
 1489  ll *.tgz
 1490  df .
 1491  ll *.tgz
 1492  df .
 1493  exit
 1494  history
 1495  wget "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval04_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D&Expires=1562049676"
 1496  ll *.t*
 1497  mv v1.0-trainval04_blobs.tgz\?AWSAccessKeyId\=AKIA6RIK4RRMFUKM7AM2\&Signature\=WqUUUYtRT8X0E%2B3A%2BUSljS2pZ9s%3D\&Expires\=1562049676 Downloads/
 1498  ll *.t*
 1499  cd Downloads/
 1500  ll *.t*
 1501  history
 1502  ll *.t*
 1503  wget --help
 1504  wget -O 3.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval03_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2/We9F9tkzhWjiFZQo=&Expires=1562049707"
 1505  wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
 1506  wget -O test.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-test_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HHkgw0OT5z6kYHhEFFRfQgd9EbE=&Expires=1562140025"
 1507  wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
 1508  ll *.tgz
 1509  wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
 1510  bg
 1511  fg
 1512  wget -O 2.tgz "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=ZtWfSDc2kmROTck1h8VI9jRlWXo=&Expires=1562073953"
 1513  fg
 1514  history|grep tgz
 1515  wget -O 3.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval03_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2/We9F9tkzhWjiFZQo=&Expires=1562049707"
 1516  history|grep tgz
 1517  exit
 1518  jobs
 1519  ll *.tgz
 1520  kill %1
 1521  jobs
 1522  ll *.tgz
 1523  rm 6.tgz 
 1524  nautilus .
 1525  ll *.tgz
 1526  df . .
 1527  df .
 1528  ll
 1529  ll *.tgz
 1530  history|grep tgz
 1531  wget -O 3.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=HRE7s9OOV2/We9F9tkzhWjiFZQo=&Expires=1562049707"
 1532  wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1533  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1534  find . -iname "*.pkl"
 1535  find . -iname "kitti*.pkl"
 1536  cd data2-kitti/
 1537  ls
 1538  cd training/
 1539  ls
 1540  cd ..
 1541  code .
 1542  cd ..
 1543  find . -iname "config_tool.py"
 1544  find . -iname "config*tool.py"
 1545  find . -iname "*tool.py"
 1546  find . -iname "config.py"
 1547  find . -iname "config_.py"
 1548  find . -iname "config_*.py"
 1549  ls
 1550  cd second.pytorch-nutonomy/
 1551  find . -iname "config_*.py"
 1552  find . -iname "*tool.py"
 1553  find . -iname "config*tool.py"
 1554  cd
 1555  ls
 1556  cd nuscenes-devkit/
 1557  find . -iname "*tool.py"
 1558  cd
 1559  data2-kitti/
 1560  cd data2-kitti/
 1561  ls
 1562  cd training/
 1563  ls
 1564  ll
 1565  mv velodyne velodyne_reduced
 1566  nvidia-smi
 1567  firefox 
 1568  nvidia-smi
 1569  gparted
 1570  sudo gparted
 1571  history|grep chgrp
 1572  ls /media/computer/
 1573  sudo chgrpu adm /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
 1574  sudo chgrp adm /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
 1575  history |less
 1576  history|grep chgrp
 1577  history --help
 1578  history -d 700
 1579  history
 1580  history |less
 1581  sudo chmod g+w /media/computer/d1d53a2d-7bae-4372-9fb7-c454f107330e/
 1582  sudo chmod g+w /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
 1583  ls
 1584  cd ..
 1585  ls
 1586  cd
 1587  ls
 1588  cd kitti-originals/
 1589  ls
 1590  unzip models_lsvm.zip 
 1591  ls
 1592  mkdir models_lsvm
 1593  mv *.mat models_lsvm/
 1594  mv readme.txt orientation.pdf models_lsvm/
 1595  ls
 1596  cd models_lsvm/
 1597  ls
 1598  code .
 1599  ls
 1600  source activate_conda.sh 
 1601  conda env list
 1602  conda activate pointpillars
 1603  ls -a
 1604  cat .bash_profile 
 1605  source .bash_profile 
 1606  gedit .bash_profile 
 1607  source .bash_profile 
 1608  cd second.pytorch-nutonomy/
 1609  ls
 1610  cd second/
 1611  ls
 1612  history|grep python
 1613  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
 1614  python --version
 1615  pip --help
 1616  pip list
 1617  pip install -U protobuf
 1618  python --version
 1619  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
 1620  ls
 1621  code .
 1622  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --measure_time=True --batch_size=1
 1623  ls
 1624  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --batch_size=1
 1625  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --batch_size=1 > first-pass.log
 1626  gedit fire-trace.log
 1627  ls
 1628  code fire-trace.log first-pass.log 
 1629  tail first-pass.log 
 1630  cat first-pass.log 
 1631  ls
 1632  history
 1633  cd
 1634  ll
 1635  ls
 1636  history
 1637  ls
 1638  cd data-nuscenes/
 1639  ls
 1640  cd v1.0-mini/
 1641  code .
 1642  cd
 1643  mkdir mini-data-nuscenes
 1644  cd data-nuscenes/v1.0-mini/
 1645  ls
 1646  mv * ~/mini-data-nuscenes/
 1647  ls
 1648  cd ..
 1649  ls
 1650  rmdir v1.0-mini
 1651  cd v1.0-mini/
 1652  ll
 1653  cat .v1.0-mini.txt 
 1654  cd ..
 1655  rm -r v1.0-mini
 1656  ls
 1657  mv v1.0-mini.tgz ~/mini-data-nuscenes/
 1658  cd ../mini-data-nuscenes/
 1659  ls
 1660  cd sweeps/
 1661  ls
 1662  cd ..
 1663  ll
 1664  code .
 1665  cd 
 1666  cd data-nuscenes/
 1667  ls
 1668  ll ~/Downloads/*.tgz
 1669  mv ~/Downloads/*.tgz .
 1670  ll
 1671  tar -xf v1.0-trainval_meta.tgz 
 1672  ls
 1673  code . 
 1674  ll
 1675  df .
 1676  tar -xf v1.0-trainval10_blobs.tgz 
 1677  ls
 1678  jobs
 1679  history|grep python
 1680  cd ../second.pytorch-nutonomy/second/
 1681  ls
 1682  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/ --batch_size=1 > first-pass.log
 1683  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/  2> 2nd-pass.log
 1684  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=~/pretrained_models_v1.5/  > 2nd-pass.log
 1685  jobs
 1686  fg
 1687  ls
 1688  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_models_v1.5/
 1689  ls
 1690  jobs
 1691  fg
 1692  df .
 1693  ls
 1694  cd
 1695  ls
 1696  pp
 1697  ll
 1698  ls
 1699  ls data2-kitti/
 1700  ls kitti-originals/
 1701  cd kitti-originals/models_lsvm/
 1702  ls
 1703  code .
 1704  cd ..
 1705  mv models_lsvm ../models_lsvm-kitti
 1706  mv models_lsvm.zip ../models_lsvm-kitti/
 1707  cd
 1708  ls
 1709  cd models_lsvm-kitti/
 1710  ls
 1711  ll
 1712  ls
 1713  history|grep python
 1714  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_models_v1.5/
 1715  cd 
 1716  cd second.pytorch-nutonomy/second/
 1717  history|grep eval
 1718  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/models_lsvm-kitti/
 1719  cd
 1720  ls
 1721  mkdir empty
 1722  cd second.pytorch-nutonomy/second/
 1723  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/empty/
 1724  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/empty/ --pickle_result=False
 1725  python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/train_pointpillars
 1726  history|grep create
 1727  python create_data.py create_kitti_info_file --data_path=/home/computer/data2-kitti/
 1728  python create_data.py create_groundtruth_database --data_path=/home/computer/data2-kitti/
 1729  python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/train_pointpillars
 1730  history|grep python
 1731  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/empty/ --pickle_result=False
 1732  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False
 1733  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False --predict_test=True
 1734  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False --ckpt_path=/home/computer/pretrained_pointpillars_nusc/
 1735  python ./pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/ --pickle_result=False --ckpt_path=/home/computer/pretrained_pointpillars_nusc/voxelnet-296960.tckpt 
 1736  python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/computer/pretrained_pointpillars_nusc/
 1737  fg
 1738  ls
 1739  find . -iname "*.tckpt"
 1740  mkdir pretrained_pointpillars_nusc
 1741  cp pretrained_models_v1.5/pp_model_for_nuscenes_pretrain/voxelnet-296960.tckpt pretrained_pointpillars_nusc/
 1742  cd pretrained_pointpillars_nusc/
 1743  ll
 1744  cd
 1745  find . -iname "*.tckpt"
 1746  df .
 1747  cd train_pointpillars/
 1748  ls
 1749  ll
 1750  ll eval_checkpoints/
 1751  cat pipeline.config 
 1752  ll
 1753  ll summary/
 1754  ll summary/loss/
 1755  ll summary/loss/loc_elem/
 1756  ll summary/loss/loc_elem/0/
 1757  ll
 1758  ll results/
 1759  ll results/step_18560/
 1760  cat checkpoints.json 
 1761  ll
 1762  ll -tl
 1763  ls
 1764  cd ..
 1765  mv train_pointpillars epoch1_train_pointpillars
 1766  ls
 1767  cd epoch1_train_pointpillars/
 1768  ll
 1769  cd ..
 1770  ls
 1771  cd second.pytorch-nutonomy/
 1772  find . -iname "*.config"
 1773  cd ../second.pytorch
 1774  find . -iname "*.config"
 1775  cd ../nuscenes-devkit/
 1776  find . -iname "*.config"
 1777  find . -iname "*.proto"
 1778  cd ../second.pytorch-nutonomy/
 1779  find . -iname "*.proto"
 1780  cd ../second.pytorch
 1781  find . -iname "*.proto"
 1782  cd 
 1783  find . -iname "*.proto"
 1784  ls
 1785  cd /media/computer/f4e42898-25a5-4741-b235-9d3597483a8e/
 1786  ll
 1787  cd nuscenes/
 1788  ll
 1789  ls
 1790  df .
 1791  rsync -ax --progress /home/computer/Downloads/1.tgz .
 1792  df .
 1793  ll
 1794  ls
 1795  lls
 1796  ls
 1797  df .
 1798  rsync v1.0-trainval04_blobs.tgz /home/computer/Downloads/
 1799  rsync -ax --progress v1.0-trainval04_blobs.tgz /home/computer/Downloads/
 1800  ls
 1801  rm v1.0-trainval04_blobs.tgz 
 1802  df .
 1803  rsync /home/computer/Downloads/v1.0-trainval06_blobs.tgz .
 1804  rsync -ax --progress /home/computer/Downloads/v1.0-trainval06_blobs.tgz .
 1805  ls
 1806  rsync -ax v1.0-trainval05_blobs.tgz /home/computer/Downloads/
 1807  rsync -ax --progress v1.0-trainval05_blobs.tgz /home/computer/Downloads/
 1808  ls
 1809  ll
 1810  cd Downloads/
 1811  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1812  wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1813  wget -O 6.tgz -c https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628
 1814  wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1815  history|grep wget
 1816  history|grep trainval06
 1817  wget -O 6.tgz -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=D+rP9Kh4xdr6myCPKFRgLXtnajA=&Expires=1562137628"
 1818  wget -c "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval06_blobs.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=%2BwaO%2FfrI%2B8TxCB5wOfg9F12uVRk%3D&Expires=1562204679" -O 62.tgz
 1819  history > ~/Documents/bash_history-numerated

```
