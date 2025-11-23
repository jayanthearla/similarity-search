# Image Retrieval System

A full-stack Flask web application for image similarity search using Vision Transformer (ViT-B/16) features and FAISS IVF indexing.

## Features

- **Feature Extraction**: Extract deep learning features from images using ViT-B/16
- **FAISS Indexing**: Build efficient IVF index for fast similarity search
- **Similarity Search**: Find similar images using cosine similarity
- **Modern UI**: Clean, responsive interface with TailwindCSS and Alpine.js
- **Real-time Progress**: Track feature extraction progress
- **Image Preview**: Visual grid display of search results

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML + TailwindCSS + Alpine.js
- **ML Model**: Vision Transformer (ViT-B/16) from Hugging Face
- **Indexing**: FAISS (Facebook AI Similarity Search)
- **Communication**: REST API (JSON)

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Extract Features (Tab 1)**
   - Enter the full path to your image directory
   - Click "Extract Features"
   - Wait for the process to complete (progress bar will show status)
   - View index statistics

4. **Search Similar Images (Tab 2)**
   - Upload a query image (drag & drop or click to browse)
   - Set the number of results (Top-K)
   - Click "Find Similar Images"
   - Browse results in the grid
   - Click any result to open the full image

## API Endpoints

### `POST /api/extract-features`
Extract features from images in a directory and build FAISS index.

**Request:**
```json
{
  "directory_path": "C:/path/to/images"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Feature extraction started"
}
```

### `GET /api/extraction-progress`
Get current extraction progress.

**Response:**
```json
{
  "status": "running",
  "progress": 45,
  "current_image": "image.jpg",
  "total_images": 100
}
```

### `POST /api/search`
Search for similar images.

**Request:**
- `query_image`: Image file (multipart/form-data)
- `k`: Number of results (form field)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "image_path": "/path/to/image.jpg",
      "filename": "image.jpg",
      "similarity": 0.93,
      "distance": 0.14,
      "rank": 1,
      "feature_stats": {...}
    }
  ]
}
```

### `GET /api/stats`
Get index statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "num_indexed_images": 1000,
    "feature_dim": 768,
    "ivf_region_count": 100,
    "probe_count": 10,
    "device_type": "cuda"
  }
}
```

## Project Structure

```
.
├── app.py                      # Flask application
├── backend/
│   ├── __init__.py
│   ├── feature_extractor.py   # ViT-B/16 feature extraction
│   ├── index_builder.py       # FAISS index building
│   └── search_engine.py       # Similarity search
├── templates/
│   └── index.html             # Frontend UI
├── static/                    # Static files (if needed)
├── uploads/                   # Temporary uploads (auto-created)
├── index_data/                # FAISS index & metadata (auto-created)
├── requirements.txt
└── README.md
```

## Configuration

### Index Parameters
Edit `backend/index_builder.py` to adjust:
- `n_regions`: Number of IVF regions (default: 100)
- `nprobe`: Number of regions to probe during search (default: 10)

### Model
The application uses `google/vit-base-patch16-224` from Hugging Face. It will be downloaded automatically on first use.

## Notes

- **GPU Support**: The application automatically uses CUDA if available, otherwise falls back to CPU
- **Directory Path**: On Windows, use full paths like `C:\Users\Name\Pictures\Dataset`
- **Image Formats**: Supports JPG, PNG, GIF, BMP, WEBP
- **File Size Limit**: Maximum upload size is 16MB

## Troubleshooting

1. **"Index not found" error**: Make sure to extract features first (Tab 1)
2. **Slow extraction**: Consider using GPU (CUDA) for faster processing
3. **Memory issues**: Reduce `n_regions` or process images in smaller batches
4. **Directory not found**: Ensure the path is correct and accessible

## License

This project is provided as-is for educational and research purposes.

