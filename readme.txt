# Universal Video Downloader

This is a Flask web server that allows you to download videos from various websites like YouTube, Instagram, Twitch, TikTok, Reddit, SoundCloud, BBC, CNN, and more, using `yt-dlp`.

## Setup and Run

These instructions assume you have Anaconda or Miniconda installed and will guide you to set up the project in the `dduckbeagy` environment.

1.  **Create Project Files:**
    Create the necessary files and directories. You can skip this if you've already created them.
    ```bash
    mkdir -p downloads templates
    touch app.py templates/index.html requirements.txt
    ```
    Then, copy the code for `app.py`, `templates/index.html`, and `requirements.txt` from our conversation history into these files.

2.  **Activate Conda Environment:**
    Activate the `dduckbeagy` environment.
    ```bash
    conda activate dduckbeagy
    ```
    *(If you don't have the environment yet, you can create it with `conda create --name dduckbeagy python=3.9` before activating.)*

3.  **Install Dependencies:**
    Once the environment is active, install the required Python packages and `ffmpeg`.
    ```bash
    pip install -r requirements.txt
    conda install -c conda-forge ffmpeg
    ```

4.  **Run the Flask Server:**
    ```bash
    python app.py
    ```

5.  **Access the Application:**
    Open your web browser and navigate to:
    [http://localhost:5001](http://localhost:5001)

## How to Use

1.  Open the web interface.
2.  Paste the URL of the content you want to download into the input box.
3.  Click the "Download" button.
4.  The downloaded file will appear in the "Downloaded Files" list below the form.
5.  Click on the file name to download it to your computer. 