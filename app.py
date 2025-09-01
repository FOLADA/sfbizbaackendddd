from flask import Flask, send_from_directory
from flask_cors import CORS
from db import init_db
from routes import bp as business_routes
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS based on environment
is_production = os.environ.get('FLASK_ENV') == 'production'
if is_production:
    # In production, restrict CORS to your frontend domain
    frontend_url = os.environ.get('FRONTEND_URL', 'https://your-frontend-domain.vercel.app')
    CORS(app, origins=[frontend_url], supports_credentials=True)
else:
    # In development, allow all origins
    CORS(app, origins="*", supports_credentials=True)

init_db()

app.register_blueprint(business_routes)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Use absolute path to uploads directory
    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    return send_from_directory(uploads_dir, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug_mode, host="0.0.0.0", port=port)  
