**Graduation Project: Simulated Optical Scattering Signals for Precancerous Detection**

This repository contains the code and documentation for my graduation project, focused on the detection of precancerous conditions using simulated optical scattering signals. In this project, I employed Deep Convolutional Generative Adversarial Networks (DCGANs) for data augmentation and customized pre-trained Convolutional Neural Networks (CNNs) for classification.

**Project Overview**

The objective of this project was to explore the potential of simulated optical scattering signals in detecting precancerous conditions. Optical scattering signals are an important component of tissue characterization, and by leveraging machine learning techniques, we aimed to enhance the accuracy and efficiency of precancerous detection.

**Methodology**

To address the scarcity of real-world scattering signal datasets, I utilized DCGANs for data augmentation. By generating synthetic scattering signals, we were able to increase the diversity and size of the training dataset, thereby improving the generalization and robustness of the trained models.

For classification, I employed tailored and pre-trained CNN networks. These networks were designed specifically to capture and analyze the features present in scattering signals indicative of precancerous conditions. By leveraging pre-trained models, we aimed to leverage the knowledge gained from large-scale image classification tasks and transfer it to our specific domain.

**Repository Structure**
This repo contains two main files. One with the Data Augmentation Algorthim (DCGAN) and the Other includes all my implementation with CNN networks for this use case.



**notice that the dataset for this project is still considered private. contact darifler@metu.edu.tr for the dataset permission**

**Getting Started**

To reproduce or build upon this project, follow these steps:

Clone this repository to your local machine using git clone .
Set up the necessary environment by installing the required dependencies listed in requirements.txt.
Ensure the simulated optical scattering signals dataset is obtained or generated and placed in the appropriate data/ directory.
Execute the provided Jupyter notebooks in the notebooks/ directory in the specified order for data preprocessing, training, fine-tuning, and evaluation.
Experiment with different model architectures, hyperparameters, or techniques by modifying the code in the notebooks or creating new ones.
Utilize the pre-trained models provided in the models/ directory for further analysis or deploy them in your own applications.

**Acknowledgments**

I would like to express my gratitude to my advisor and the members of the research team for their guidance and support throughout this project. Additionally, I would like to acknowledge the authors of relevant papers and the open-source community for their valuable contributions.

**License**

This project is licensed under MIT License, allowing you to use, modify, and distribute the code for academic and non-commercial purposes.

Please note that this repository contains code and data related to simulated optical scattering signals and is not intended for clinical or real-world medical applications.

