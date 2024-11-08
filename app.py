import base64
import io
import logging
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request
from helper import combine_2model_v11
from sklearn import preprocessing
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['UPLOAD_FOLDER'] = 'temp'
matplotlib.use("Agg")

le = preprocessing.LabelEncoder()
le.fit(['Adidas',
        'Apple',
        'BMW',
        'Citroen',
        'Cocacola',
        'DHL',
        'Fedex',
        'Ferrari',
        'Ford',
        'Google',
        'Heineken',
        'HP',
        'Intel',
        'McDonalds',
        'Mini',
        'Nbc',
        'Nike',
        'Pepsi',
        'Porsche',
        'Puma',
        'RedBull',
        'Sprite',
        'Starbucks',
        'Texaco',
        'Unicef',
        'Vodafone',
        'Yahoo'])


@app.route('/')
def home():
    return render_template('index.html')


def fig_to_base64(fig) -> str:
    if fig is None:
        return ""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return img_str
    finally:
        buf.close()
        plt.close(fig) if fig is not None else None


def process_result(model_output: Optional[Tuple]) -> Dict[str, Any]:
    """Process the model output and return appropriate response"""
    if model_output is None:
        return {
            'success': False,
            'error': 'No logo detected in the image. Please try with a different image.',
            'image1': '',
            'image2': '',
        }

    fig1, fig2, result = model_output
    return {
        'success': True,
        'label': le.inverse_transform([result])[0] if result is not None else 'Unknown',
        'image1': fig_to_base64(fig1),
        'image2': fig_to_base64(fig2),
    }


@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    # Kiểm tra phần mở rộng của file
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif'}
    if not file.filename.lower().endswith(tuple(f'.{ext}' for ext in allowed_extensions)):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
        }), 400

    try:
        # Lưu file tạm thời
        temp_path = os.path.join('temp', secure_filename(file.filename))
        os.makedirs('temp', exist_ok=True)

        try:
            file.save(temp_path)
            logger.info(f"Processing image: {temp_path}")

            # Gọi model
            model_output = combine_2model_v11(temp_path)

            # Xử lý kết quả
            result = process_result(model_output)

            if not result['success']:
                return jsonify(result), 400

            return jsonify(result)

        finally:
            # Dọn dẹp file tạm
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the image. Please try again.',
            'details': str(e) if app.debug else None
        }), 500

# Endpoint để kiểm tra trạng thái server


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'supported_formats': ['png', 'jpg', 'jpeg', 'gif'],
        'max_file_size': '16MB'
    })


if __name__ == '__main__':
    app.run(debug=True)
