# if you also want to create a conda venv with this script
#read -p 'Name for the conda virtual environment (press Enter for "lanedet"): ' env_name
#if [[ "$env_name" == "" ]]
#then
#    env_name='lanedet'
#fi
#conda create -n $env_name python=3.8 -y
#conda activate $env_name

pip3 install setuptools==58.3.0
pip3 install cycler==0.10  python-dateutil==2.7.3 scipy==1.10.1 pytz==2017.3 numpy==1.24.4 #attrs==18.1.0
pip3 install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64  kiwisolver==1.4.7 pyparsing==3.2.0 imageio==2.36.0 networkx==3.4.2 tifffile==2024.9.20 joblib==1.4.2 threadpoolctl==3.5.0 mmcv==1.7.2 mmcv-full==1.7.2
pip3 install attrs
pip3 install six
pip3 install future
pip3 install -U wheel mock pillow
pip3 install testresources
pip3 install pyyaml
pip3 install Cython

pip3 install gdown

## OLD JETSON INSTALLATION START
# gdown https://drive.google.com/uc?id=1MnVB7I4N8iVDAkogJO76CiQ2KRbyXH_e  # old Jetson pytorch version
# pip3 install torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl
# rm torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl

# gdown https://drive.google.com/uc?id=1BaBhpAizP33SV_34-l3es9MOEFhhS1i2
# pip3 install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
# rm torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
## OLD JETSON INSTALLATION END

## NEW JETSON ORIN NANO DEV KIT INSTALLATION START
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl -O torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
rm torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl -O torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
pip3 install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
rm torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
pip3 install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
rm torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
## NEW JETSON ORIN NANO DEV KIT INSTALLATION END

pip3 install pathos addict yapf mmcv mmcv-full imgaug albumentations scikit-learn p_tqdm catkin_pkg #six

# experimental start
#pip3 install opencv-python-headless==4.8.0.74 kiwisolver==1.0.1 pyparsing==2.2.1 imageio==2.3.0 networkx==2.0 tifffile==2019.7.26 joblib==0.11 threadpoolctl==2.0.0

# experimental end

python setup.py build develop