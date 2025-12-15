# CPSC 3800 Final Project: Facial Feature Analysis for Personalized Makeup Recommendation
### Chris Shia, Sophia Dai, Maryeva Gonzalez

Makeup is becoming a widely used tool across all people, ages, and genders. While makeup tutorials and information are widely accessible, people often have trouble determining what type of makeup is optimal for their own individual facial features. In this repository, we developed a model that can make makeup recommendations for an individual based on their facial features identified through a facial image. We utilized publicly available face datasets (eg. CelebA) to extract facial features, using pre-trained CNN encoders to embed facial regions. Then, we extracted specific facial features from random images and compared those to the embedding spaces to create our makeup recommendations for those faces. 

## How to Run
1. Navigate to the cv-final-project directory and create a virtual environment in your terminal with the command ```python3 -m venv .venv``` on macOS/Linux, or ```python -m venv .venv``` on bash. Activate the virtual environment with ```source .venv/bin/activate``` on macOS/Linux or ```.venv\Scripts\Activate.ps1``` in Windows Powershell. 
2. Install library requirements from the requirements.txt file into your virtual environment in your terminal with the command ```pip install -r requirements.txt```
3. Open the ```final_makeup_recommendation_notebook.ipynb``` file, and follow the instructions on uploading your photo. We saved our trained models in the models directory and have them ready for use in this cleaned notebook. Within seconds you should have your makeup recommendations!


## Methodology / Pipeline
Our main project deliverable is condensed in the ```final_makeup_recommendation_notebook.ipynb``` file. The other files include steps in our methodology and development process, with python notebooks numbered 02-07 being the main steps we took in creating this makeup recommendation model. We will outline them below. Feel free to check them out if interested.
1.  ```02_data_exploration.ipynb```: in this python notebook we parsed the CelebA dataset (see Datasets section for where to download from) and performed some basic data exploration and cleaning, such as using heatmaps and parsing through non-front facing faces. The ouput was the ```celeba_front_facing.csv``` in the ```celeba_metadata/ ``` directory, which has the cleaned and processed faces. 
2. ```03_feature_cropping.ipynb```: in this notebook we took a subset of the cleaned and processed faces from ```celeba_front_facing.csv``` and further processed them by cropping them by the specific feature. This step was run in Google Collab and can only be run there due to the depreciation of dlib in python. See Datasets section to find our cropped_features dataset.
3. ```04_embedding_classification.ipynb```: in this notebook we used our processed, cropped images and created vector embeddings for each feature with the pre-trained resnet-18 CNN. These are outputted in the ```embeddings/``` directory and can be directly accessed from there.
4. ```05_region_classification.ipynb```: in this notebook we used the outputed embeddings to train logistic regression classifiers for each class of feature embedings. These classifiers are saved to the ```models/``` directory and can be directly accessed from there.
5. ```06_07_inference_pipeline_recommendation.ipynb```: in this notebook we wrote methods to create embeddings for an input image and compare them to our logistic classifiers. These methods are packaged and called in our  ```final_makeup_recommendation_notebook.ipynb``` deliverable.

## Datasets
As our datasets took up lots of storage, we have linked where you can download them for your own exploration purposes. **Again, our deliverable runs on models and embeddings we have saved from previous runs, so don't feel the need to download these yourself (it takes quite a long time, especially the preprocessing steps).**

**CelebA Dataset:** You can download the CelebA dataset from this kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download . It takes a long time to load and process, so we've included extracted embeddings and pre-trained classifiers saved in the embeddings and classifiers directories. Place these in a directory with the name ```img_align_celeba``` in the project directory. 

**cropped_features Dataset:** You can find the cropped_features dataset in this google drive: https://drive.google.com/drive/folders/1OPfC6cHF4N7fR-cBfN7L-61qXZIGbiTl?usp=sharing. Place these in a directory with the name ```cropped_features``` in the project directory.


## Project Structure 
This project is organized as follows:

```text
cv-final-project/
│
├── celeba_metadata/                 # CelebA attribute & landmark metadata
│   ├── list_attr_celeba.csv
│   ├── list_landmarks_celeba.csv
│   └── ...
│
├── embeddings/                      # Saved ResNet-18 embeddings (.npy)
│   ├── eyes_Narrow_Eyes_embeddings.npy
│   ├── eyes_Narrow_Eyes_labels.npy
│   └── ...
│
├── models/                          # Trained logistic regression models
│   ├── eyes_Narrow_Eyes_logreg.pkl
│   ├── nose_Big_Nose_logreg.pkl
│   └── ...
│
├── 02_data_exploration.ipynb        # Step 2, Data exploration and cleaning notebook
├── 03_feature_cropping.ipynb        # Step 3, Landmark-based region extraction
├── 04_embedding_classification.ipynb# Step 4, Embedding extraction + initial classification
├── 05_region_classification.ipynb   # Step 5, Region-specific logistic regression models
├── 06_07_inference_pipeline_recommendation.ipynb
│                                   # Step 6+7, End-to-end inference & recommendation pipeline
├── final_makeup_recommendation_notes.ipynb
│                                   # Project main deliverable
├── inference_pipeline_recommendation.py
│                                   # Scripted inference pipeline of the 06_07 notebook
│
├── test_face.jpg                    # Example input image used in notebook 06_07
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Git ignore rules
```
