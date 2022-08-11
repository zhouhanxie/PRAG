# NRT (Neural Rating and Tips generation)
PyTorch re-implementation without review prediction task ([original implementation in Theano](https://github.com/lipiji/NRT-theano))

## Paper
- Li, Piji, et al. [Neural Rating Regression with Abstractive Tips Generation for Recommendation](http://lipiji.com/docs/li2017neural.pdf). SIGIR'17.

**A small ecosystem for Recommender Systems-based Natural Language Generation is available at [NLG4RS](https://github.com/lileipisces/NLG4RS)!**

## Datasets to [download](https://drive.google.com/drive/folders/1z90ExLiEc1ZTyPir5qxbXxQOWslsspIH?usp=sharing)
- TripAdvisor Hong Kong
- Amazon Movies & TV
- Yelp 2019

For those who are interested in how to obtain (feature, opinion, template, sentiment) quadruples, please refer to [Sentires-Guide](https://github.com/lileipisces/Sentires-Guide).

## Usage
Below is an example of how to run NRT.
```
python -u main.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisor/ >> tripadvisor.log
```

## Code dependencies
- Python 3.6
- PyTorch 1.6

## Friendly reminders
- If you want to equip the model with Byte Pair Encoding (BPE), please refer to [PEPLER](https://github.com/lileipisces/PEPLER).
- If you want to add the review prediction task back, please refer to the context prediction task in [PETER](https://github.com/lileipisces/PETER).
- If you want to change back to the original settings as reported in the paper (which would deteriorate the performance), please comment out line 92 and uncomment out line 93 in [main.py](main.py).

## Citations
If you find this re-implementation useful, please consider citing our papers.
```
@article{2022-PEPLER,
	title={Personalized Prompt Learning for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	journal={arXiv preprint arXiv:2202.07371},
	year={2022}
}
@inproceedings{ACL21-PETER,
	title={Personalized Transformer for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	booktitle={ACL},
	year={2021}
}
@inproceedings{CIKM20-NETE,
	title={Generate Neural Template Explanations for Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	booktitle={CIKM},
	year={2020}
}
@inproceedings{WWW20-NETE,
	title={Towards Controllable Explanation Generation for Recommender Systems via Neural Template},
	author={Li, Lei and Chen, Li and Zhang, Yongfeng},
	booktitle={WWW Demo},
	year={2020}
}
```
