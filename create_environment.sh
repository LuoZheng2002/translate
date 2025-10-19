set -e
cd /work/nvme/bfdz/zluo8/translate
module load python/3.12.1
python -m venv .venv
source .venv/bin/activate
cd /projects/bfdz/zluo8/translate
pip install --upgrade pip
pip install -r requirements.txt