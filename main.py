import streamlit as st
import os
import base64
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


# TODO: Take data from Biene40 or UFZ.
base_dir = os.path.dirname(os.path.abspath(__name__))
queen = os.path.join(
    base_dir,
    "data/Hive1_12_06_2018_QueenBee_H1_audio___16_00_00-bee-14-18.wav",
)

# queen = os.path.join(
#     base_dir, "data/queen_presence/Hive1_12_06_2018_QueenBee_H1_audio___15_00_00.wav"
# )
no_queen = os.path.join(
    base_dir,
    "data/CJ001 - Missing Queen - Day - (101)-bee-1-6.wav",
)

ae_image = os.path.join(base_dir, "data/images/autoencoder.jpg")


def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    return fig


def plot_frequencies(
    signal, sample_rate: int = 22050, cut_freq: int = 700, plot_name: str = None
):
    # STFT
    D = np.fft.fft(signal)

    # Step 2: Power spectrum
    S = np.abs(D) ** 2

    # Step 3: Compute the frequency axis
    frequencies = np.fft.fftfreq(signal.size, d=1 / sample_rate)

    if cut_freq is None:
        cut_freq = frequencies.max() / 2
    # check if cut_freq is legitimate
    cut_idx = np.argmax(frequencies >= cut_freq)

    # Step 4: Plot the intensity vs. frequency
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[:cut_idx], S[:cut_idx], color="orange")
    plt.title(
        f"Power Spectrum {plot_name}"
        if plot_name
        else "Power Spectrum w.o. time domain"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim(0, cut_freq)
    plt.grid(True)
    plt.tight_layout()

    return plt


def get_audio_player_html(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return f'<audio controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'


def plot_spectrogram(
    y, sr=22050, hop_length=512, scale="log", cmap="magma", title=None
):
    fig, ax = plt.subplots(figsize=(12, 8))

    if scale == "mel":
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    else:
        spec = librosa.stft(y, hop_length=hop_length)

    specDB = librosa.amplitude_to_db(abs(spec))

    img = librosa.display.specshow(
        specDB,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis=scale,
        ax=ax,
        cmap=cmap,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Spectrogram of {title}" if title is not None else "Spectrogram")

    return fig


def main():

    st.title("Audio Player App")

    # Sidebar for file upload
    st.sidebar.header("Select Audio Type")
    audio_type = st.sidebar.radio("Choose audio type:", ("Normal", "Anomaly"))

    # Main area for audio playback
    st.header("Audio Playback")

    # Display audio players for the selected playlist
    if audio_type == "Normal":
        st.subheader("Playing: Normal Audio")
        audio_file = queen
    else:
        st.subheader("Playing: Anomaly Audio")
        audio_file = no_queen

    # Load audio file
    y, sr = load_audio(audio_file)

    # Display audio player
    st.markdown(get_audio_player_html(audio_file), unsafe_allow_html=True)

    # Display waveform
    st.pyplot(plot_frequencies(y, sr))

    # Extract features
    features = extract_features(y, sr)

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Convert to PyTorch tensor
    tensor_features = torch.FloatTensor(normalized_features)

    # Create and train autoencoder
    input_dim = tensor_features.shape[1]
    autoencoder = Autoencoder(input_dim)
    train_autoencoder(autoencoder, tensor_features)

    # Display autoencoder architecture
    st.subheader("Autoencoder Architecture")
    st.image(ae_image)

    # Reconstruct features
    with torch.no_grad():
        reconstructed_features = autoencoder(tensor_features).numpy()

    # Inverse transform the features
    original_features = scaler.inverse_transform(normalized_features)
    reconstructed_features = scaler.inverse_transform(reconstructed_features)

    st.subheader("Autoencoder Results")
    # Calculate reconstruction error
    mse = np.mean(np.square(original_features - reconstructed_features))
    st.write(f"Reconstruction Mean Squared Error: {mse:.4f}")

    # Plot autoencoder results
    # st.pyplot(plot_autoencoder_results(original_features, original_features))
    st.pyplot(plot_spectrogram(y))
    st.pyplot(plot_autoencoder_results(reconstructed_features))
    # Display additional information
    st.write("---")
    st.write(f"Currently playing: {audio_type} audio")
    st.write(f"Sample rate: {sr} Hz")
    st.write(f"Duration: {len(y)/sr:.2f} seconds")
    st.write("Use the sidebar to switch between Normal and Anomaly audio.")


if __name__ == "__main__":
    main()
