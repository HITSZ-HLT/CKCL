# CKCL

> The official implementation for Findings of the ACL 2023 paper *Context or Knowledge is Not Always Necessary: A Contrastive Learning Framework for Emotion Recognition in Conversations*.

<img src="https://img.shields.io/badge/Venue-ACL--23-blue" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">

## Requirements
* Python 3.7.11
* PyTorch 1.8.0
* Transformers 4.1.1
* CUDA 11.1

## Preparation
Download [**features**](https://drive.google.com/file/d/1p1Gm0kHXjWpOzkhPy7z6uNwggOY6uLXF/view?usp=drive_link) and save them in ./.

## Training & Evaluation
You can train the models with the following codes:

For IEMOCAP: ```python train_iemocap.py --active-listener```

For DailyDialog: ```python train_dailydialog.py --active-listener --class-weight --residual```

For MELD Emotion: ```python train_meld.py --active-listener --attention simple --dropout 0.5 --rec-dropout 0.3 --lr 0.0001 --mode1 2 --classify emotion --mu 0 --l2 0.00003 --epochs 60```

For MELD Sentiment: ```python train_meld.py --active-listener --class-weight --residual --classify sentiment```

For EmoryNLP Emotion: ```python train_emorynlp.py --active-listener --class-weight --residual```

For EmoryNLP Sentiment: ```python train_emorynlp.py --active-listener --class-weight --residual --classify sentiment```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:

```
@inproceedings{tu2023context,
  title={Context or knowledge is not always necessary: A contrastive learning framework for emotion recognition in conversations},
  author={Tu, Geng and Liang, Bin and Mao, Ruibin and Yang, Min and Xu, Ruifeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={14054--14067},
  year={2023}
}
```

## Credits
The code of this repository partly relies on [COSMIC](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC) and I would like to show my sincere gratitude to the authors behind these contributions.

