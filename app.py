"""
Flask application for Image Retrieval System
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import urllib.parse
from werkzeug.utils import secure_filename
from backend.feature_extractor import FeatureExtractor
from backend.index_builder import IndexBuilder
from backend.search_engine import SearchEngine
import threading
import time

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
INDEX_FOLDER = 'index_data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global state
extraction_progress = {
    'status': 'idle',
    'progress': 0,
    'current_image': '',
    'total_images': 0
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('templates', 'index.html')

@app.route('/api/extract-features', methods=['POST'])
def extract_features():
    """Extract features from images in a directory and build FAISS index"""
    try:
        data = request.get_json()
        directory_path = data.get('directory_path', '').strip()
        
        if not directory_path:
            return jsonify({
                'success': False,
                'error': 'Directory path is required'
            }), 400
        
        if not os.path.isdir(directory_path):
            return jsonify({
                'success': False,
                'error': 'Directory does not exist'
            }), 400
        
        # Reset progress
        extraction_progress['status'] = 'running'
        extraction_progress['progress'] = 0
        extraction_progress['current_image'] = ''
        extraction_progress['total_images'] = 0
        
        # Run extraction in a thread
        def extract_thread():
            try:
                extractor = FeatureExtractor()
                builder = IndexBuilder()
                
                # Extract features
                features, image_paths, metadata = extractor.extract_from_directory(
                    directory_path,
                    progress_callback=lambda current, total, img: update_progress(current, total, img)
                )
                
                if len(features) == 0:
                    extraction_progress['status'] = 'error'
                    extraction_progress['error'] = 'No images found in directory'
                    return
                
                # Build index
                index_path = os.path.join(INDEX_FOLDER, 'faiss_index.bin')
                metadata_path = os.path.join(INDEX_FOLDER, 'metadata.json')
                
                index_stats = builder.build_index(
                    features,
                    image_paths,
                    metadata.get('feature_dim', 768),
                    index_path,
                    metadata_path
                )
                
                extraction_progress['status'] = 'completed'
                extraction_progress['index_stats'] = index_stats
            except Exception as e:
                extraction_progress['status'] = 'error'
                extraction_progress['error'] = str(e)
        
        def update_progress(current, total, img):
            extraction_progress['progress'] = int((current / total) * 100) if total > 0 else 0
            extraction_progress['current_image'] = img
            extraction_progress['total_images'] = total
        
        thread = threading.Thread(target=extract_thread)
        thread.daemon = True
        thread.start()
        
        # Wait a bit for initial progress
        time.sleep(0.5)
        
        return jsonify({
            'success': True,
            'message': 'Feature extraction started',
            'status': extraction_progress['status']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/extraction-progress', methods=['GET'])
def get_extraction_progress():
    """Get current extraction progress"""
    return jsonify(extraction_progress)

@app.route('/api/search', methods=['POST'])
def search():
    """Search for similar images"""
    try:
        if 'query_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No query image provided'
            }), 400
        
        file = request.files['query_image']
        k = int(request.form.get('k', 10))
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        query_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(query_path)
        
        # Check if index exists
        index_path = os.path.join(INDEX_FOLDER, 'faiss_index.bin')
        metadata_path = os.path.join(INDEX_FOLDER, 'metadata.json')
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return jsonify({
                'success': False,
                'error': 'Index not found. Please extract features first.'
            }), 400
        
        # Perform search
        search_engine = SearchEngine(index_path, metadata_path)
        results = search_engine.search(query_path, k)
        
        # Clean up uploaded file
        if os.path.exists(query_path):
            os.remove(query_path)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get index statistics"""
    try:
        metadata_path = os.path.join(INDEX_FOLDER, 'metadata.json')
        index_path = os.path.join(INDEX_FOLDER, 'faiss_index.bin')
        
        if not os.path.exists(metadata_path) or not os.path.exists(index_path):
            return jsonify({
                'success': False,
                'error': 'Index not found'
            }), 404
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        stats = {
            'num_indexed_images': metadata.get('num_images', 0),
            'metadata_count': len(metadata.get('image_paths', [])),
            'device_type': metadata.get('device', 'unknown'),
            'ivf_region_count': metadata.get('n_regions', 0),
            'probe_count': metadata.get('nprobe', 0),
            'feature_dim': metadata.get('feature_dim', 0)
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Serve images from the indexed directory"""
    try:
        # Decode URL-encoded path
        image_path = urllib.parse.unquote(image_path)
        
        # Security: ensure path is absolute and exists
        if not os.path.isabs(image_path):
            # If relative, try to resolve it
            image_path = os.path.abspath(image_path)
        
        # Check if file exists and is a file
        if os.path.exists(image_path) and os.path.isfile(image_path):
            directory = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            return send_from_directory(directory, filename)
        
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

