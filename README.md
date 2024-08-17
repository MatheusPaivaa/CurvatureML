<h1 align="center">Curvature Estimation Using Machine Learning Algorithms ⌒🖥️</h1>
<p align="center"> Research project supported by FAPESP during February 2024 - February 2025. </p>

<p align="center">
  <a href="#project-structure">Project Structure</a> • 
  <a href="#installation">Installation</a> • 
  <a href="#usage">Usage</a> • 
  <a href="#dataset">Dataset</a> • 
  <a href="#algorithms-and-models">Algorithms and Models</a> •
  <a href="#results">Results</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

<p align="center">
This proposal is part of the context of machine learning algorithms for estimating mean and Gaussian curvature, as a counterpoint to conventional geometric techniques. The research involves analysing and implementing regression algorithms trained from a feature vector containing information extracted from the 3D mesh (mapped as a graph), together with information used in the geometric calculation. Once the machine learning regression model has been trained on this labelled data, it is hoped that the test phase will produce results close to those obtained by the geometric approach, but in a considerably shorter processing time.
</p>

<div align="center">
  <img src="https://github.com/MatheusPaivaa/CurvatureML/blob/master/img/met.png" alt="metodology" />
</div>

## <div id="project-structure"></div>Project Structure


## <div id="installation"></div>Installation
To set up the project environment and install the necessary dependencies, follow these steps:

```bash
# Clone the Repository
git clone https://github.com/MatheusPaivaa/CurvatureML.git

# Navigate to the Project Directory
cd CurvatureML

# Install Dependencies
pip install -r requirements.txt

# (Optional) Setup the Environment
# If the project uses a virtual environment, create and activate it:
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install the dependencies within the virtual environment
pip install -r requirements.txt

# Run the Project
# After installation, you can start using the project by running the main script or any other relevant scripts
```
Ensure that you have Python 3.7 or later installed on your machine.

## <div id="usage"></div>Usage

## <div id="Dataset"></div>Dataset
The dataset used in this project is [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces?resource=download) from Kaggle. This dataset contains a collection of 3D human face models that are essential for training and evaluating the curvature estimation algorithms.

Additionally, another valuable resource is the [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html), which is highly recommended for further training and evaluation purposes. This dataset provides a different set of facial images that can be used to enhance the robustness and generalization of the models.

## <div id="algorithms-and-models"></div>Algorithms and Models

## <div id="results"></div>Results

## <div id="contributing"></div>Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## <div id="license"></div>License
This project is licensed under the MIT License. See the LICENSE file for details.

## <div id="acknowledgements"></div>Acknowledgements
I would like to thank Joao E.S. Batista Neto, Farid Tari, and Antonio Castelo Filho for their guidance and support throughout this project.

[comment]: <> (Notebook: https://colab.research.google.com/drive/1vOrjJOcYwbcm86kxqWYi1lbH6Wy4KLBU?usp=sharing)
