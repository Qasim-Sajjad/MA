# Music Analysis Project

A Docker-based tool for analyzing musical features from audio files. This project provides automated analysis of audio files through a containerized environment.

## Features
- Docker containerization for easy setup and deployment
- Support for multiple audio formats (MP3, WAV)
- Batch processing capabilities
- Automated musical feature extraction
- Directory-based file processing

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Qasim-Sajjad/MA.git
   ```

2. Navigate to the project directory:
   ```bash
   cd MA
   ```

3. Build the Docker container:
   ```bash
   docker compose build
   ```

4. Clean build cache (recommended):
   ```bash
   docker builder prune
   ```

5. Run the container:
   ```bash
   docker compose run music_analysis
   ```

## How It Works
1. **Audio File Management**:
   - Place audio files in the `audio_files` directory
   - Create subfolders for organized processing
   - Supports individual files and batch processing

2. **File Processing**:
   - Automatic detection of audio files
   - Feature extraction and analysis
   - Batch processing of directories

3. **Directory Configuration**:
   - For subfolder processing, modify `driver.py`:
     ```python
     process_directory('audio_files/your_subfolder_name')
     ```

## Project Structure
```
MA/
├── audio_files/          # Audio files directory
├── docker-compose.yml    # Docker configuration
├── Dockerfile           # Container build instructions
└── driver.py           # Main processing script
```

## Usage
1. Add your audio files to the `audio_files` directory
2. Configure processing path in `driver.py` if using subfolders
3. Run the Docker container
4. Check output for analysis results

## Requirements
- Docker
- Sufficient disk space
- Audio files in supported formats (MP3, WAV)

## Contributing
Feel free to open an issue or submit a pull request for any improvements.

## License
[Add License Information]

## Contact
[Add Contact Information]