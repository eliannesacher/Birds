import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import shutil
import random
import pandas as pd
import requests

############################################################################
####################  HELPER FUNCTIONS  ####################################
############################################################################

# Load audio file and filter out silent parts based on a threshold
def load_and_filter_audio(mp3_path, silence_threshold=25):
    y, sr = librosa.load(mp3_path)
    non_silent_intervals = librosa.effects.split(y, top_db=silence_threshold)
    filtered_audio = np.concatenate([y[start:end] for start, end in non_silent_intervals])
    return filtered_audio, sr

# Split the filtered audio into fixed-length segments
def split_audio_into_segments(filtered_audio, sr, segment_length=10, min_length=3):
    segment_samples = segment_length * sr  # Calculate the number of samples per segment
    min_samples = min_length * sr  # Calculate minimum number of samples for a valid segment
    segments = [filtered_audio[i:i + segment_samples] for i in range(0, len(filtered_audio), segment_samples)
                if len(filtered_audio[i:i + segment_samples]) >= min_samples]
    return segments

# Save spectrograms of audio segments to a specified directory
def save_spectrograms(segments, sr, output_dir, base_name='segment'):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    for idx, segment in enumerate(segments):
        segment_name = f"{base_name}_{idx + 1}"
        segment = librosa.util.normalize(segment)
        S = librosa.feature.melspectrogram(y=segment, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{segment_name}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

# Process all MP3 files in a folder, filter them, segment, and save their spectrograms
def process_audio_folder(input_folder, output_folder, silence_threshold=25, segment_length=10, min_length=3):
    output_file_names = set(os.listdir(output_folder))  # Cache output filenames to avoid reprocessing
    for file_name in os.listdir(input_folder):
        if file_name[:-4] + "_1.png" in output_file_names:
            continue
        
        if file_name.lower().endswith('.mp3'):
            try:
                mp3_path = os.path.join(input_folder, file_name)
                base_name = os.path.splitext(file_name)[0]
                filtered_audio, sr = load_and_filter_audio(mp3_path, silence_threshold=silence_threshold)
                segments = split_audio_into_segments(filtered_audio, sr, segment_length=segment_length, min_length=min_length)
                save_spectrograms(segments, sr, output_folder, base_name=base_name)
            except:
                print(f"Error occurred with: {file_name}")

# Move a specified number of random files from one folder to another
def move_random_files(source_folder, destination_folder, num_files):
    files = os.listdir(source_folder)  # Get list of files in the source folder
    random_files = random.sample(files, num_files)  # Randomly select files
    for file_name in random_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)  # Move selected files

# Download an MP3 file from a URL
def download_mp3(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        return None

# Download specific species sounds based on a DataFrame and species classification
def download_species(df, species_class):
    folder_path = f"./sounds/{species_class}/"
    current_sounds = set(os.listdir(folder_path))
    count = len(current_sounds)
    j = 0
    for i, row in df.iterrows():        
        url = row["identifier"]
        gbifID = row["gbifID"]
        sound_name = str(gbifID) + ".mp3"
        
        if sound_name in current_sounds:
            continue

        if count + j > 988:
            return
        
        try:
            sound = download_mp3(url)
        except:
            print(f"species class: {species_class} count: {j} gbifID: {gbifID} | failed downloading")
            continue
    
        if sound is None:
            print(f"species class: {species_class} count: {j} gbifID: {gbifID} | download returned None")
            continue 
    
        save_path = folder_path + str(gbifID) + ".mp3"
        with open(save_path, 'wb') as f:
            f.write(sound)
        print(f"species class: {species_class} count: {j} gbifID: {gbifID} | downloaded")
        j += 1

############################################################################
####################    DOWNLOAD MP3s   ####################################
############################################################################

df_multimedia = pd.read_csv("./data/multimedia.txt", delimiter="\t")
df_multimedia_sound = df_multimedia[df_multimedia['type']=="Sound"]
columns_to_be_merged = [
    "gbifID", "behavior", "continent", "countryCode", "family", "species"
]
df_occurrence = pd.read_csv("./data/occurrence.txt", delimiter="\t")
df_merged = pd.merge(df_occurrence[columns_to_be_merged], df_multimedia_sound, on="gbifID")
columns_to_drop = [
    "title", "publisher", "description", "source", "audience", 
    "creator", "contributor", "license", "rightsHolder",
    "created", "format", "type", "references"
]
# clean df
df_merged_clean = df_merged.drop(columns_to_drop, axis=1)
df_merged_clean = df_merged_clean.dropna(subset=['identifier'])
df_merged_clean = df_merged_clean[df_merged_clean['behavior'] == "call"]
df_merged_clean = df_merged_clean.drop(['behavior'], axis=1)
df_merged_clean = df_merged_clean.drop(['family'], axis=1)
df_merged_clean = df_merged_clean.dropna(subset=['species'])

species = list(set(df_merged_clean["species"]))
species_counts = df_merged_clean["species"].value_counts()[:10]
species_list = ["Parus major", "Phylloscopus collybita", "Cyanistes caeruleus", "Passer montanus", "Fringilla coelebs", "Passer domesticus", "Erithacus rubecula", "Aegithalos caudatus", "Strix aluco", "Corvus corax"]  
species_dfs = [df_merged_clean[df_merged_clean["species"] == species_list[i]] for i in range(len(species_list))]

for i in range(len(species_list)-1, -1, -1):
    download_species(species_dfs[i], i)

############################################################################
####################    PREPROCESS MP3s   ##################################
############################################################################

for i in range(10):
    input_folder = f"./sounds/{i}"
    output_folder = f"./generated_spectrograms/{i}"
    if (not os.path.exists(output_folder)):
        os.mkdir(output_folder)
    process_audio_folder(input_folder, output_folder)
    print(i, "done")

############################################################################
####################    BALANCE DATASET   ##################################
############################################################################

source_folder = './generated_spectrograms/'
destination_folder = './generated_spectrograms_balanced/'

# Top 5 most frequent
new_idx = [0,1,2,3,9] 

num_files_to_move = min([len(os.listdir(f"./generated_spectrograms/{i}")) for i in new_idx])

for new_i, old_i in enumerate(new_idx):
    curr_source = source_folder + f"{old_i}"
    curr_dest = destination_folder + f"{new_i}"
    if (not os.path.exists(curr_dest)):
        os.mkdir(curr_dest)
    move_random_files(curr_source, curr_dest, num_files_to_move)
