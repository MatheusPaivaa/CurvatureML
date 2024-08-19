<h1 align="center">Curvature Estimation Using Machine Learning Algorithms ⌒🖥️</h1>
<p align="center"> Research project supported by FAPESP during February 2024 - February 2025. </p>

<p align="center">
  <a href="#project-structure">Project Structure</a> • 
  <a href="#installation">Installation</a> • 
  <a href="#usage">Usage</a> • 
  <a href="#dataset">Dataset</a> • 
  <a href="#algorithms-and-models">Algorithms and Models</a> •
  <a href="#feature-selection">Feature Selection</a> •
  <a href="#results">Results</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgements">Acknowledgements</a>
  <a href="#bibliography">Bibliography</a>
</p>

<p align="center">
This proposal is part of the context of machine learning algorithms for estimating mean and Gaussian curvature, as a counterpoint to conventional geometric techniques. The research involves analysing and implementing regression algorithms trained from a feature vector containing information extracted from the 3D mesh (mapped as a graph), together with information used in the geometric calculation. Once the machine learning regression model has been trained on this labelled data, it is hoped that the test phase will produce results close to those obtained by the geometric approach, but in a considerably shorter processing time.
</p>

<div align="center">
  <img src="https://github.com/MatheusPaivaa/CurvatureML/blob/master/img/met_ing.png" alt="metodology" />
</div>
<br>
**Notebook:** https://colab.research.google.com/drive/1vOrjJOcYwbcm86kxqWYi1lbH6Wy4KLBU?usp=sharing

## <div id="project-structure"></div>Project Structure
This project is organized into two main functions:

1. **3D Mesh Generation and Preprocessing**
   - **3D Mesh Formation:** Using MediaPipe, the project generates 3D meshes from 2D images, creating `.obj` files that represent the facial structures. These meshes are then used for further analysis and model training. [Folder Link](https://github.com/MatheusPaivaa/CurvatureML/tree/master/scripts/object_generation)
   - **Feature and Label Extraction:** Features and labels are extracted by annotating the mesh, which involves identifying key points and regions on the 3D models that correspond to specific curvature characteristics. [Folder Link](https://github.com/MatheusPaivaa/CurvatureML/tree/master/scripts/curv_features_calculation)
   - **Dataset Assembly:** The preprocessed data is compiled into a structured dataset, ready for training machine learning models. [Folder Link](https://github.com/MatheusPaivaa/CurvatureML/tree/master/scripts/curv_features_calculation)

2. **Model Implementation and Testing**
   - Several machine learning models have been implemented and tested to estimate curvature. The models currently tested include:
     - **Support Vector Regression (SVR)**
     - **Decision Tree**
     - **Random Forest**
     - **Multilayer Perceptron (MLP)**

   [Models Folder](https://github.com/MatheusPaivaa/CurvatureML/tree/master/models)

#### Disclaimer

In addition to the main functions of the project, there are also several **useful scripts** included in the repository. These scripts can assist with various tasks such as data visualization, additional preprocessing, and debugging. While not central to the primary objectives of the project, they provide valuable tools that can streamline and enhance your workflow.

You can find these scripts in the [useful_scripts](https://github.com/MatheusPaivaa/CurvatureML/tree/master/scripts/useful_scripts) directory or scattered throughout the other directories depending on their specific functionality.

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
# After installation, you can start using the project by running the main
# script or any other relevant scripts
```
Ensure that you have Python 3.7 or later installed on your machine.

## <div id="usage"></div>Usage

#### Generating 3D OBJ Files (./scripts/object_generation.py)

To generate 3D `.obj` files from 2D images, follow these steps:

1. **Organize Your Images:**
   Place the images you want to convert into a folder.

2. **Run the Script:**
   Execute the provided script to process the images and generate `.obj` files. You will need to specify the path to the images folder when prompted.

3. **Check the Output:**
   The generated `.obj` files will be saved in the designated output folder (e.g., `../../data/face_processed`).
   
```bash
python3 main.py

Enter the path to the images folder: /path/to/your/images/folder
```

<p align="center">
  <img src="https://github.com/MatheusPaivaa/CurvatureML/blob/master/img/mediapipe_example.png" alt="face" width="200"/>
</p>

#### Curvature Calculation and Feature Extraction (./scripts/curv_features_calculation)

To calculate the curvatures and extract features from the 3D `.obj` files, follow these steps:

1. **Prepare the Input Files:**
   Ensure that your processed 3D `.obj` files are located in the designated input directory (e.g., `../../data/face_processed`).

2. **Run the C++ Program:**
   Execute the C++ program to process each `.obj` file in the input directory. The program will:
   - Calculate the Gaussian and Mean curvatures.
   - Extract features from the 3D mesh.
   - Save the curvature results and extracted features in separate output files.

3. **Check the Output:**
   The output files will be saved in the specified directories (e.g., `../../data/in_out_processed/output` for curvatures and `../../data/in_out_processed/input` for features).

After running the program, you will have both curvature data and feature vectors ready for use in training and evaluating your machine learning models.

```bash
make all
make run
```

#### Model Implementation (./models)

This project includes the implementation of four different models for curvature estimation. Each model has its own dedicated directory, following a standardized structure:

- **Train:** Contains the script used to train the model.
- **Test:** Contains the script used to test the model on a specific face and plot the results.
- **Tune:** Contains scripts for hyperparameter tuning, allowing you to optimize the model’s performance by adjusting key parameters.
  
```bash
python3 [script]
```

## <div id="Dataset"></div>Dataset
The dataset used in this project is [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces?resource=download) from Kaggle. This dataset contains a collection of 3D human face models that are essential for training and evaluating the curvature estimation algorithms.

<p align="center">
  <img src="https://github.com/MatheusPaivaa/CurvatureML/blob/master/img/rosto.jpg" alt="face" width="200"/>
  <img src="https://github.com/MatheusPaivaa/CurvatureML/blob/master/img/rosto_aug_5.jpg" alt="face_aug" width="200"/>
  <img src="https://github.com/MatheusPaivaa/CurvatureML/blob/master/img/rosto_aug_6.jpg" alt="face_aug_2" width="200"/>
</p>

Additionally, another valuable resource is the [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html), which is highly recommended for further training and evaluation purposes. This dataset provides a different set of facial images that can be used to enhance the robustness and generalization of the models.

## <div id="algorithms-and-models"></div>Algorithms and Models

#### Curvature Calculation and Feature Extraction

Curvatures, both mean and Gaussian, are geometric properties used to characterize the shape and curvature of surfaces, with broad applications in computer graphics and computational geometry. The geometric definitions and calculations of both curvatures are provided in the bibliography.

To efficiently calculate the labels (curvature values) and features, a custom algorithm was implemented in C++ using an `unordered_map` structure. This data structure was chosen for its average-case constant time complexity, allowing for fast access and organization of the calculated curvature values.

#### Algorithm Overview:

1. **Input Data Structure:**
   - The algorithm takes 3D mesh data as input, typically stored in a format like `.obj`. Each vertex and its corresponding normal vectors are processed to calculate the mean and Gaussian curvatures.

2. **Curvature Calculation:**
   - For each vertex in the mesh, the algorithm computes the mean and Gaussian curvatures using discrete differential geometry methods. The calculations are based on the local neighborhood of each vertex, considering the angles and areas of the surrounding triangles.

3. **Storage in `unordered_map`:**
   - The calculated curvature values (both mean and Gaussian) for each vertex are stored in an `unordered_map`, where the key is the vertex identifier, and the value is a pair of curvature values (mean, Gaussian).
   - This approach ensures that each vertex's curvature can be accessed in constant time, making the process highly efficient, especially for large meshes.

4. **Feature Extraction:**
   - Additional features related to the surface's geometric properties are extracted and also stored in the `unordered_map`. These features include the angles between adjacent normal vectors, edge lengths, and triangle areas.
   - The extracted features, along with the calculated curvatures, form the input dataset for the subsequent machine learning models.

5. **Performance Optimization:**
   - The use of `unordered_map` significantly reduces the time complexity of organizing and retrieving the curvature and feature data, allowing for rapid comparisons with the proposed machine learning models. This efficiency is critical when dealing with large datasets, as it minimizes computational overhead and speeds up the entire process.

#### Model Implementation and Testing

The project involves the implementation and testing of several machine learning models to estimate curvatures based on the extracted features:

- **Support Vector Regression (SVR):** A regression model that finds the optimal hyperplane for predicting continuous values, effective for high-dimensional data.
- **Decision Tree:** A non-linear model that makes predictions based on learned decision rules from the data, easy to interpret but prone to overfitting.
- **Random Forest:** An ensemble method that builds multiple decision trees and merges them to improve prediction accuracy and control overfitting.
- **Multilayer Perceptron (MLP):** A type of neural network model composed of multiple layers of neurons, capable of capturing complex patterns in the data.

## <div id="feature-delection"></div>Feature Selection

In this project, a careful selection of features was made to ensure that the model can accurately capture the geometric properties of the 3D mesh. The features chosen for training the model are as follows:

- **X, Y, Z:** These are the Cartesian coordinates of each vertex in the 3D mesh, representing its position in space.
- **MeanAdjX, MeanAdjY, MeanAdjZ:** These features represent the mean coordinates of the adjacent vertices for each vertex, providing a measure of the local geometric context.
- **VertexDegree:** This indicates the number of edges connected to each vertex, giving insight into the complexity of the local mesh topology.
- **starArea:** The total area of the triangles (or faces) adjacent to each vertex, reflecting the local surface area around the vertex.
- **NumAdjFaces:** The number of faces (triangles) adjacent to each vertex, which can be used to infer the density of the mesh in that region.
- **NormalX, NormalY, NormalZ:** These are the components of the normal vector at each vertex, essential for understanding the surface orientation.
- **MeanNormalX, MeanNormalY, MeanNormalZ:** These features represent the mean normal vectors of the adjacent vertices, offering additional information about the surface's smoothness and curvature.

#### Importance of Selected Features

Each of these features plays a critical role in characterizing the geometry of the 3D mesh:

- **Spatial Position (X, Y, Z):** The basic position of vertices is fundamental to understanding the overall shape and layout of the mesh.
- **Local Geometry (MeanAdjX, MeanAdjY, MeanAdjZ, VertexDegree, starArea, NumAdjFaces):** These features provide information about the local structure around each vertex, crucial for capturing variations in the surface curvature.
- **Surface Orientation (NormalX, NormalY, NormalZ, MeanNormalX, MeanNormalY, MeanNormalZ):** The normal vectors and their means are key to understanding the direction and smoothness of the surface at each point, directly influencing the curvature calculations.

#### Tuning features

## <div id="results"></div>Results

🚧 Testing... 🚧

## <div id="contributing"></div>Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## <div id="license"></div>License
This project is licensed under the MIT License. See the LICENSE file for details.

## <div id="acknowledgements"></div>Acknowledgements
I would like to thank Joao E.S. Batista Neto, Farid Tari, and Antonio Castelo Filho for their guidance and support throughout this project.

## <div id="bibliography"></div>Bibliography

1. **Crane, K.** *Discrete Differential Geometry - CMU 15-458/858. Lecture 16.* 2019. Available at: [https://www.youtube.com/watch?v=NlU1m-OfumE](https://www.youtube.com/watch?v=NlU1m-OfumE). Accessed on: December 11, 2023.

2. **Crane, K.** *Discrete Differential Geometry - CMU 15-458/858. Lecture 17.* 2019. Available at: [https://www.youtube.com/watch?v=sokeN5VxBB8](https://www.youtube.com/watch?v=sokeN5VxBB8). Accessed on: December 11, 2023.

3. **Bruce, J. W.; Giblin, P. J.** *Curves and Singularities: A Geometrical Introduction to Singularity Theory.* 2nd ed. Cambridge: Cambridge University Press, 1992.

2. **Meyer, M. et al.** *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds.* In: Springer. *Visualization and Mathematics III.* [S.l.], 2003, pp. 35–57.

3. **Lugaresi, C. et al.** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv preprint arXiv:1906.08172, 2019.

[comment]: <> (Notebook: https://colab.research.google.com/drive/1vOrjJOcYwbcm86kxqWYi1lbH6Wy4KLBU?usp=sharing)
