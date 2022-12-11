# Thai SER Model

- dataset: https://github.com/vistec-AI/dataset-releases/releases/tag/v1

## How to use github with Colab

```py
# !pip install python-dotenv -q

from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv('git_token')

!git config --global user.email ""
!git config --global user.name ""
!git remote add origin https://{TOKEN}@github.com/chuan-khuna/ThaiSER.git

!git remote set-url origin https://{TOKEN}@github.com/chuan-khuna/ThaiSER.git
```
