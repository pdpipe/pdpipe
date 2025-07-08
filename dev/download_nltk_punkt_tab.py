import nltk

if __name__ == "__main__":
    print("Attempting to download 'punkt_tab' NLTK resource...")
    try:
        nltk.download('punkt_tab')
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}") 