conda create --name eval_env python==3.10
conda activate eval_env
pip install git+https://github.com/google-research/bleurt.git
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
pip install rouge-score nltk