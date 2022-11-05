# Colour the Flower

A machine learning project completed as part of Monash University's student engineering team, Monash DeepNeuron. This project was completed with the [pix2pix](https://github.com/phillipi/pix2pix) architecture as the inspiration.  

The model uses Generative Adversarial Networks (GANs) to colour a black and white image of flowers into coloured images.


## Setup
### Dataset
The dataset was obtained from [kaggle](https://www.kaggle.com/datasets/vaibhavrmankar/colour-the-flower-gan-data). After downloading and extracting, the downloaded dataset can be used to replace the `Data` folder. The `Data` folder requires `test` and `train` subfolders, which is what the kaggle dataset has. 

### Setting up the environment
First remove all of the empty `.gitignore` files within the folders. These exist to bypass github not allowing empty directories.

To install using conda:

Tested using `python 3.7.13`.
Change to the directory where the project is (and subsequently where the `environment.yml` file is located) and run:
```bash
conda env create -f environment.yml
```

## Results
Input:
![blackandwhite](https://user-images.githubusercontent.com/78593106/200111048-6b6d8fe1-18b4-47c8-bf12-e2a960f1e9a4.jpg)

Output:
![training40](https://user-images.githubusercontent.com/78593106/200111089-d54b77bd-9fe5-47a6-99f8-a8517459b989.jpg)

Target:
![real](https://user-images.githubusercontent.com/78593106/200111170-ca8cee3e-8a81-4ace-9af3-7167fcb5a525.jpg)

The model has an easy time learning green, white and yellow. It struggles a bit with some other colours (particularly blue) as well as things that are not part of a flower (walls, people etc...). 

Fake vs Real:

![fake_real](https://user-images.githubusercontent.com/78593106/200111708-9bf9b455-86c3-4923-9a53-3fbc4c3d30e4.jpg)




Below is a gif of the training process and how the output had improved overtime.
![ezgif com-gif-maker](https://user-images.githubusercontent.com/78593106/200111057-53957c3a-0f06-44d8-b77a-bce2f0643c57.gif)
