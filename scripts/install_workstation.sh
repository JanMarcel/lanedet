read -p 'Name for the conda virtual environment (press Enter for "lanedet"): ' env_name
if [[ "$env_name" == "" ]]
    env_name = 'lanedet'
fi
conda create -n $env_name python=3.8 -y
conda activate $env_name

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements_workstation.txt

python setup.py build develop