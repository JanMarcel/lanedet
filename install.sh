pip3 install future
pip3 install -U wheel mock pillow
pip3 install testresources
pip3 install pyyaml
pip3 install setuptools==58.3.0
pip3 install Cython

pip3 install gdown

gdown https://drive.google.com/uc?id=1MnVB7I4N8iVDAkogJO76CiQ2KRbyXH_e

pip3 install torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl

rm torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl

gdown https://drive.google.com/uc?id=1BaBhpAizP33SV_34-l3es9MOEFhhS1i2
pip3 install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
rm torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl

pip3 install pathos
pip3 install cycler==0.10 kiwisolver==1.0.1 pyparsing==2.2.1 python-dateutil==2.7.3 opencv-python-headless==4.8.0.74 scipy==1.10.1 imageio==2.3.0 six networkx==2.0 tifffile==2019.7.26 joblib==0.11 threadpoolctl==2.0.0 pytz==2017.3
pip3 install numpy==1.24.4

python setup.py build develop