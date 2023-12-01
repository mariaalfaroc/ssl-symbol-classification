<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">Few-Shot Symbol Classification via Self-Supervised Learning and Nearest Neighbor</h1>

<h4 align="center">Full text available <a href="https://doi.org/10.1016/j.patrec.2023.01.014" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>


## About

We propose a three-stage process for unlabelled self-supervised learning (SSL) symbol classification:

<p align="center">
  <img src="extras/workflow_final.png" alt="content" style="border: 1px solid black; width: 500px;">
</p>

1) **Extraction of isolated symbols from unlabelled documents.** Symbols are automatically extracted from unlabelled documents using a sliding-window approach. The documents are divided into patches, which are then converted to grayscale and binarized. The entropy value of each patch is calculated, and patches with an entropy value greater than a user-defined threshold are considered potential symbols.
    
2) **Training of a neural feature extractor using SSL.** A CNN is trained using the [Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906) SSL method. 
    
3) **Symbol classification using a k-nearest neighbours (kNN) classifier.** Firstly, a query and a labelled set of symbol images are mapped to the representation space defined by the CNN obtained in Stage 2. Secondly, the kNN rule is applied to classify the query based on the labels of its k closest neighbours.

## How To Use

The datasets used in this work, namely *Capitan*, *TKH*, *Egyptian*, and *GRPOLY-DB*, are available upon [request](mailto:malfaro@dlsi.ua.es). After obtaining these datasets, please place them in the [`datasets`](datasets) folder. 

To run the code, you'll need to meet certain requirements which are specified in the [`Dockerfile`](Dockerfile). Alternatively, you can set up a virtual environment if preferred.

Once you have prepared your environment (either a Docker container or a virtual environment), you are ready to begin. Execute the [`experiments/run.py`](experiments/run.py) script to replicate the experiments from our work:

```python
python experiments/run.py
```

## Citations

```bibtex
@article{alfaro2023few,
  title     = {{Few-Shot Symbol Classification via Self-Supervised Learning and Nearest Neighbor}},
  author    = {Alfaro-Contreras, Mar{\'\i}a and R{\'\i}os-Vila, Antonio and Valero-Mas, Jose J and Calvo-Zaragoza, Jorge},
  journal   = {{Pattern Recognition Letters}},
  volume    = {167},
  pages     = {1--8},
  year      = {2023},
  publisher = {Elsevier},
  doi       = {10.1016/j.patrec.2023.01.014},
}

@inproceedings{rios2022few,
  title     =   {{Few-Shot Music Symbol Classification via Self-Supervised Learning and Nearest Neighbor}},
  author    =   {R{\'\i}os-Vila, Antonio and Alfaro-Contreras, Mar{\'\i}a and Valero-Mas, Jose J and Calvo-Zaragoza, Jorge},
  booktitle =   {{Proceedings of the 3rd International Workshop Pattern Recognition for Cultural Heritage}},
  pages     =   {93--107},
  year      =   {2022},
  publisher =   {Springer},
  address   =   {Montréal, Canada},
  month     =   aug,
  doi       =   {10.1007/978-3-031-37731-0_8},
}
```

## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License
This work is under a [MIT](LICENSE) license.
