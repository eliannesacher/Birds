# <img style="float: left; padding-right: 10px; width: 45px" src="https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/iacs.png"> CS109B Advanced Topics in Data Science

## **Classifying Bird Sounds From Around the World**

**Harvard University**<br/>
**Spring 2024**<br/>
**Instructors**: Pavlos Protopapas<br/>
**Project Members**: Elianne Sacher, Shane Kissinger, and Oleg Pavliv

## **Problem Statement**

Our project aims to develop a machine-learning model capable of accurately classifying bird sounds based on their spectrograms. We utilize audio data and potentially geographical data to build a robust model for identifying bird families from their sounds and associated visual information. This tool is crucial for biodiversity monitoring and conservation efforts, helping in the automated identification of bird species from their vocalizations.

## **Significance of the Project**

The ability to identify bird families through their calls is vital for ecological research, especially in understanding bird populations and behaviors which are essential for habitat conservation and assessing the impacts of climate change. Our project contributes to advancements in bioacoustics technology and provides essential tools for automated wildlife monitoring. Moreover, by creating a model and dataset for bird call classification, we support conservation policies and promote citizen science initiatives, enhancing public engagement and appreciation for wildlife conservation.

## **Repository Structure**

This repository contains several Jupyter notebooks that document our project workflow:

1. **EDA_and_Basic_Analysis.ipynb**
   - This notebook includes exploratory data analysis and basic statistical analysis of the bird sound dataset.

2. **Preprocessing_on_VM.py**
   - Conducted on a server/VM for enhanced memory and compute power, this script details the preprocessing steps and the downloading of audio files for the top 10 most common bird species.
   - This script downloads the top species's "call" type mp3 data from our dataset, rids the samples of silence/noise to only contain actual information (in this case bird sounds), then it divides the "clean" sound samples into 10-second portions, later it normalizes the amplitude of these samples, and finally converts them into spectrogram images for the next notebook, the model's notebook, to later model accordingly.

3. **Model_Training_and_Evaluation.ipynb**
   - This notebook contains the implementation of our machine learning model, analysis of the results, and the conclusions drawn from our study.

4. **Transfer_Learning_with_WAV2VEC2.0.ipynb** (Optional)
   - An exploration of transfer learning using the WAV2VEC2.0 model. This was not used in the final project as our primary model achieved satisfactory performance and the processing time with WAV2VEC2.0 was significantly higher.

## **Libraries Used**

Here’s a comprehensive explanation of each library used in the provided scripts/notebooks:

1. **requests**: A Python library used for making HTTP requests to fetch data from the web.
2. **zipfile**: Provides tools to create, read, write, append, and list ZIP files.
3. **io**: Offers interfaces to stream handling, often used for managing binary data read from a network or files.
4. **os**: A module for interacting with the operating system, handling file system operations, and manipulating paths.
5. **pickle**: Implements binary protocols for serializing and de-serializing Python object structures.
6. **seaborn** (sns): A data visualization library based on matplotlib for creating attractive statistical graphics.
7. **matplotlib.pyplot** (plt): Functions that make matplotlib work like MATLAB, useful for creating figures and plots.
8. **pandas** (pd): Provides high-performance data structures and tools for data analysis.
9. **numpy** (np): Essential for scientific computing, supporting large, multi-dimensional arrays and matrices.
10. **sys**: Accesses variables maintained by the interpreter and functions that interact with the interpreter.
11. **scipy** (sp): Used for scientific and technical computing.
12. **sklearn**: Features various algorithms for machine learning, including classification, regression, and clustering.
13. **tensorflow** (tf): An open-source platform for machine learning, facilitating model building and training.
14. **keras**: A high-level neural networks API running on top of TensorFlow, enabling fast experimentation.
15. **PIL.Image**: Enhances Python with image processing capabilities, supporting extensive file formats and operations.
16. **warnings**: Manages warning messages in Python, allowing suppression of specific categories of warnings.
17. **torch** (PyTorch): An open-source library for machine learning, used in applications like computer vision and NLP.
18. **torchaudio**: Provides tools for audio processing within the PyTorch framework.
19. **soundfile** (sf): Functions for reading and writing sound files, supporting various formats.
20. **Resample** (from torchaudio.transforms): Used to resample waveform signals.
21. **DataLoader, TensorDataset, WeightedRandomSampler** (from torch.utils.data): Tools for loading datasets with various options for batching and sampling.
22. **transformers** (from huggingface): Offers pre-trained models for tasks like text classification and summarization, which can be fine-tuned.
23. **AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor** (from transformers): Facilitates audio processing and classification using pre-trained models.
24. **evaluate**: Provides evaluation metrics and benchmarking for machine learning models.
25. **tqdm**: Displays progress bars for loops, enhancing the visibility of process execution in command line.
26. **gc** (Garbage collection): Interfaces with Python's garbage collector for manual memory management.
27. **re** (Regular expressions): For string searching and manipulation using pattern matching.
28. **compute_class_weight** (from sklearn.utils): Calculates class weights for imbalanced datasets in machine learning.
29. **concatenate_datasets, Dataset, load_dataset, ClassLabel, load_from_disk** (from datasets): Functions for loading and manipulating datasets efficiently.
30. **nn** (from torch): Provides components to build neural networks within PyTorch.
31. **itertools**: Offers a collection of tools for creating iterators for efficient looping.
32. **shutil**: High-level file operations, particularly useful for copying and removing directories.
33. **librosa**: A library for music and audio analysis, providing the necessary tools for music information retrieval.
34. **accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score** (from sklearn.metrics): Measures various aspects of classification model performance.


## **Contributions**

We welcome contributions and suggestions to improve our model or extend the project scope. Please feel free to fork the repository, make changes, and submit a pull request.
