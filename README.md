# <img style="float: left; padding-right: 10px; width: 45px" src="https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/iacs.png"> CS109B Advanced Topics in Data Science

## **Classifying Bird Sounds From Around the World**

**Harvard University**<br/>
**Spring 2024**<br/>
**Instructors**: Pavlos Protopapas<br/>
**Project Members**: Elianne Sacher, Shane Kissinger, and Oleg Pavliv

## **Problem Statement**

Our project aims to develop a machine learning model capable of accurately classifying bird sounds based on their spectrograms. We utilize audio data and potentially geographical data to build a robust model for identifying bird families from their sounds and associated visual information. This tool is crucial for biodiversity monitoring and conservation efforts, helping in the automated identification of bird species from their vocalizations.

## **Significance of the Project**

The ability to identify bird families through their calls is vital for ecological research, especially in understanding bird populations and behaviors which are essential for habitat conservation and assessing the impacts of climate change. Our project contributes to advancements in bioacoustics technology and provides essential tools for automated wildlife monitoring. Moreover, by creating a model and dataset for bird call classification, we support conservation policies and promote citizen science initiatives, enhancing public engagement and appreciation for wildlife conservation.

## **Repository Structure**

This repository contains several Jupyter notebooks that document our project workflow:

1. **EDA_and_Basic_Analysis.ipynb**
   - This notebook includes exploratory data analysis and basic statistical analysis of the bird sound dataset.

2. **Preprocessing_on_VM.py**
   - Conducted on a server/VM for enhanced memory and compute power, this script details the preprocessing steps and the downloading of audio files for the top 10 most common bird species.
   - This script downloads the top species "call" type mp3 data from our dataset, rids the samples of silent/noise to only contain actual information (in this case bird sounds), then it divides the "clean" sound samples into 10-second portions, normalizes the amplitude of these samples, and finally converts them into a spectrogram for the model notebook to later model accordingly.

3. **Model_Training_and_Evaluation.ipynb**
   - This notebook contains the implementation of our machine learning model, analysis of the results, and the conclusions drawn from our study.

4. **Transfer_Learning_with_WAV2VEC2.0.ipynb** (Optional)
   - An exploration of transfer learning using the WAV2VEC2.0 model. This was not used in the final project as our primary model achieved satisfactory performance and the processing time with WAV2VEC2.0 was significantly higher.

## **Usage**

To replicate our analysis or explore the notebooks:
- Clone this repository to your local machine or open it in a Jupyter notebook environment.
- Ensure all dependencies are installed by running `pip install -r requirements.txt`.
- Follow the notebooks in the order listed for a comprehensive understanding of the project workflow.

## **Contributions**

We welcome contributions and suggestions to improve our model or extend the project scope. Please feel free to fork the repository, make changes, and submit a pull request.
