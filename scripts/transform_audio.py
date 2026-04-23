"""
Module for processing and transforming audio files.

This module provides functionality to load audio files, resample them to a target
sample rate, convert them to mono, and save the processed results.
"""

import argparse
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


def process_audio_files(input_path: Path, output_path: Path, sample_rate: int = 24000) -> None:
    """
    Process audio files by resampling and converting to mono.

    Loads all WAV files from the input directory, resamples them to the specified
    sample rate, converts them to mono, and saves the processed files to the output
    directory.

    Parameters
    ----------
    input_path: Path
        Path to the directory containing input WAV files.
    output_path: Path
        Path to the directory where processed audio files will be saved.
    sample_rate: int, optional
        Target sample rate in Hz. Defaults to 24000.

    Returns
    -------
        None

    Raises
    ------
    Exception
        Prints error message if any audio file fails to process, but continues
                   processing remaining files.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all .wav files in the input directory
    audio_files = list(input_path.rglob("*.wav"))

    print(f"Found {len(audio_files)} audio files to process.")

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load the audio file, resample to the target sample rate, and convert to mono
            y, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

            # Save the processed audio file
            output_file = output_path.joinpath(f"{audio_file.stem}.wav")
            sf.write(output_file, y, sr)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and transform audio files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/esc50/audio",
        help="Directory containing input audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/esc50/fg_esc50_24k_mono",
        help="Directory to save processed audio files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Target sample rate for processed audio files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sample_rate = args.sample_rate

    process_audio_files(input_dir, output_dir, sample_rate=sample_rate)

    print("Audio processing complete.")
