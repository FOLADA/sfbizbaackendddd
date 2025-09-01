from flask import request, jsonify
from db import get_db
from flask import Blueprint
import jwt
import datetime
import os
import secrets
import requests
from werkzeug.utils import secure_filename
from functools import wraps
import json
import sqlite3
import bcrypt
from passlib.hash import bcrypt as passlib_bcrypt
from passlib.exc import MissingBackendError
import openai
import random

SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_key")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "your_google_maps_api_key")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key-here")

# Initialize OpenAI client
if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here":
    try:
        openai.api_key = OPENAI_API_KEY
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"OpenAI client initialization error: {e}")
        openai_client = None
else:
    openai_client = None

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_token(user_id, email):
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header required"}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        request.user_id = payload['user_id']
        return f(*args, **kwargs)
    return decorated_function

def geocode_address(address):
    """Convert address to latitude/longitude using Google Maps Geocoding API"""
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == "your_google_maps_api_key":
        # Return dummy coordinates for development
        return 37.7749, -122.4194  # San Francisco coordinates
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": GOOGLE_MAPS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data["status"] == "OK" and data["results"]:
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

bp = Blueprint("businesses", __name__)

@bp.route("/")
def home():
    return "Flask backend is working!"

@bp.route("/businesses", methods=["GET"])
def list_businesses():
    category = request.args.get("category")
    with get_db() as conn:
        if category:
            cursor = conn.execute("SELECT * FROM businesses WHERE category = ?", (category,))
        else:
            cursor = conn.execute("SELECT * FROM businesses")
        businesses = [dict(row) for row in cursor.fetchall()]
    return jsonify(businesses), 200

@bp.route("/businesses/search", methods=["GET"])
def search_businesses():
    # Get search parameters
    query = request.args.get("q", "")
    categories = request.args.get("category", "").split(",") if request.args.get("category") else []
    locations = request.args.get("location", "").split(",") if request.args.get("location") else []
    min_rating = request.args.get("minRating")
    max_rating = request.args.get("maxRating")
    sort_by = request.args.get("sortBy", "name")
    sort_order = request.args.get("sortOrder", "asc")
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 12))
    offset = (page - 1) * limit

    with get_db() as conn:
        # Build the base query
        base_query = """
            SELECT b.*, 
                   COALESCE(AVG(r.rating), 0) as avg_rating,
                   COUNT(r.id) as total_reviews
            FROM businesses b
            LEFT JOIN reviews r ON b.id = r.business_id
        """
        
        where_conditions = []
        params = []
        
        # Add search query condition
        if query:
            where_conditions.append("(b.name LIKE ? OR b.description LIKE ? OR b.category LIKE ? OR b.location LIKE ?)")
            search_term = f"%{query}%"
            params.extend([search_term, search_term, search_term, search_term])
        
        # Add category filter
        if categories and categories[0]:
            placeholders = ",".join(["?" for _ in categories])
            where_conditions.append(f"b.category IN ({placeholders})")
            params.extend(categories)
        
        # Add location filter
        if locations and locations[0]:
            placeholders = ",".join(["?" for _ in locations])
            where_conditions.append(f"b.location IN ({placeholders})")
            params.extend(locations)
        
        # Note: Rating filters are applied in HAVING clause after GROUP BY
        
        # Combine WHERE conditions
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        # Add GROUP BY
        base_query += " GROUP BY b.id"
        
        # Add HAVING clause for rating filters (after GROUP BY)
        having_conditions = []
        if min_rating:
            having_conditions.append("COALESCE(AVG(r.rating), 0) >= ?")
            params.append(float(min_rating))
        
        if max_rating:
            having_conditions.append("COALESCE(AVG(r.rating), 0) <= ?")
            params.append(float(max_rating))
        
        if having_conditions:
            base_query += " HAVING " + " AND ".join(having_conditions)
        
        # Add ORDER BY
        order_mapping = {
            "name": "b.name",
            "rating": "avg_rating",
            "recent": "b.id",  # Assuming newer businesses have higher IDs
            "distance": "b.id"  # Placeholder for distance sorting
        }
        
        sort_field = order_mapping.get(sort_by, "b.name")
        base_query += f" ORDER BY {sort_field} {sort_order.upper()}"
        
        # Add pagination
        base_query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        # Execute the query
        cursor = conn.execute(base_query, params)
        businesses = [dict(row) for row in cursor.fetchall()]
        
        # Get total count for pagination
        count_query = """
            SELECT COUNT(DISTINCT b.id) as total
            FROM businesses b
            LEFT JOIN reviews r ON b.id = r.business_id
        """
        
        count_where_conditions = []
        count_params = []
        
        if query:
            count_where_conditions.append("(b.name LIKE ? OR b.description LIKE ? OR b.category LIKE ? OR b.location LIKE ?)")
            search_term = f"%{query}%"
            count_params.extend([search_term, search_term, search_term, search_term])
        
        if categories and categories[0]:
            placeholders = ",".join(["?" for _ in categories])
            count_where_conditions.append(f"b.category IN ({placeholders})")
            count_params.extend(categories)
        
        if locations and locations[0]:
            placeholders = ",".join(["?" for _ in locations])
            count_where_conditions.append(f"b.location IN ({placeholders})")
            count_params.extend(locations)
        
        if count_where_conditions:
            count_query += " WHERE " + " AND ".join(count_where_conditions)
        
        count_cursor = conn.execute(count_query, count_params)
        total_count = count_cursor.fetchone()["total"]
        
        # Get filter options for response
        categories_cursor = conn.execute("SELECT DISTINCT category FROM businesses")
        available_categories = [row["category"] for row in categories_cursor.fetchall()]
        
        locations_cursor = conn.execute("SELECT DISTINCT location FROM businesses")
        available_locations = [row["location"] for row in locations_cursor.fetchall()]
        
        # Get rating range
        rating_cursor = conn.execute("""
            SELECT MIN(business_rating) as min_rating, 
                   MAX(business_rating) as max_rating
            FROM (
                SELECT COALESCE(AVG(r.rating), 0) as business_rating
                FROM businesses b
                LEFT JOIN reviews r ON b.id = r.business_id
                GROUP BY b.id
            )
        """)
        rating_range = rating_cursor.fetchone()
        
        # Transform businesses to match expected format
        for business in businesses:
            business["rating"] = business.get("avg_rating", 0)
            business["totalReviews"] = business.get("total_reviews", 0)
            # Parse socials JSON
            if business.get("socials"):
                try:
                    business["socials"] = json.loads(business["socials"])
                except Exception:
                    business["socials"] = {}
            else:
                business["socials"] = {}
        
        response = {
            "businesses": businesses,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_count,
                "pages": (total_count + limit - 1) // limit
            },
            "filters": {
                "categories": available_categories,
                "locations": available_locations,
                "ratingRange": {
                    "min": rating_range["min_rating"] if rating_range else 0,
                    "max": rating_range["max_rating"] if rating_range else 5
                }
            }
        }
        
        return jsonify(response), 200

@bp.route("/search-suggestions", methods=["GET"])
def get_search_suggestions():
    query = request.args.get("q", "")
    if not query or len(query) < 2:
        return jsonify({"suggestions": []}), 200
    
    with get_db() as conn:
        # Search in business names
        business_cursor = conn.execute(
            "SELECT DISTINCT name as text, 'business' as type, COUNT(*) as count FROM businesses WHERE name LIKE ? GROUP BY name LIMIT 3",
            (f"%{query}%",)
        )
        business_suggestions = [dict(row) for row in business_cursor.fetchall()]
        
        # Search in categories
        category_cursor = conn.execute(
            "SELECT DISTINCT category as text, 'category' as type, COUNT(*) as count FROM businesses WHERE category LIKE ? GROUP BY category LIMIT 2",
            (f"%{query}%",)
        )
        category_suggestions = [dict(row) for row in category_cursor.fetchall()]
        
        # Search in locations
        location_cursor = conn.execute(
            "SELECT DISTINCT location as text, 'location' as type, COUNT(*) as count FROM businesses WHERE location LIKE ? GROUP BY location LIMIT 2",
            (f"%{query}%",)
        )
        location_suggestions = [dict(row) for row in location_cursor.fetchall()]
        
        # Combine and format suggestions
        all_suggestions = business_suggestions + category_suggestions + location_suggestions
        
        # Add unique IDs
        for i, suggestion in enumerate(all_suggestions):
            suggestion["id"] = f"{suggestion['type']}_{i}"
        
        return jsonify({"suggestions": all_suggestions}), 200

@bp.route("/businesses/filter-options", methods=["GET"])
def get_filter_options():
    with get_db() as conn:
        # Get categories
        categories_cursor = conn.execute("SELECT DISTINCT category FROM businesses")
        categories = [row["category"] for row in categories_cursor.fetchall()]
        
        # Get locations
        locations_cursor = conn.execute("SELECT DISTINCT location FROM businesses")
        locations = [row["location"] for row in locations_cursor.fetchall()]
        
        # Get rating range
        rating_cursor = conn.execute("""
            SELECT MIN(business_rating) as min_rating, 
                   MAX(business_rating) as max_rating
            FROM (
                SELECT COALESCE(AVG(r.rating), 0) as business_rating
                FROM businesses b
                LEFT JOIN reviews r ON b.id = r.business_id
                GROUP BY b.id
            )
        """)
        rating_range = rating_cursor.fetchone()
        
        return jsonify({
            "categories": categories,
            "locations": locations,
            "ratingRange": {
                "min": rating_range["min_rating"] if rating_range else 0,
                "max": rating_range["max_rating"] if rating_range else 5
            }
        }), 200

@bp.route("/businesses/<int:biz_id>", methods=["GET"])
def get_business(biz_id):
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM businesses WHERE id = ?", (biz_id,))
        business = cursor.fetchone()
        
        if business:
            # Get images for this business
            images_cursor = conn.execute("SELECT * FROM business_images WHERE business_id = ?", (biz_id,))
            images = [dict(row) for row in images_cursor.fetchall()]
            
            business_dict = dict(business)
            business_dict['images'] = images
            # Parse socials JSON
            if business_dict.get('socials'):
                try:
                    business_dict['socials'] = json.loads(business_dict['socials'])
                except Exception:
                    business_dict['socials'] = {}
            else:
                business_dict['socials'] = {}
            return jsonify(business_dict), 200
    return jsonify({"error": "Business not found"}), 404

@bp.route("/businesses", methods=["POST"])
@require_auth
def add_business():
    required_fields = ["name", "category", "description", "services"]
    data = request.get_json(force=True)
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Geocode the address if provided
    latitude, longitude = None, None
    if data.get("location"):
        latitude, longitude = geocode_address(data["location"])

    socials = data.get("socials", {})
    socials_json = json.dumps(socials)
    
    # Process service pricing data
    service_pricing = data.get("service_pricing", {})
    service_pricing_json = json.dumps(service_pricing)

    # Get user ID from token
    user_id = request.user_id

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO businesses (name, category, description, services, service_pricing, image_url, location, latitude, longitude, socials, rating, owner_id, business_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["name"],
                data["category"],
                data["description"],
                data["services"],
                service_pricing_json,
                data["image_url"],
                data.get("location", ""),
                latitude,
                longitude,
                socials_json,
                data.get("rating", None),
                user_id,
                data.get("business_hours", "")
            ),
        )
        conn.commit()
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Insert service pricing data if provided
        if service_pricing:
            for service_name, pricing_data in service_pricing.items():
                current_price = pricing_data.get("current_price", 0)
                recommended_price = pricing_data.get("recommended_price", current_price)
                pricing_strategy = pricing_data.get("pricing_strategy", "competitive")
                confidence_score = pricing_data.get("confidence_score", 0.8)
                
                conn.execute(
                    """
                    INSERT INTO service_pricing (business_id, service_name, current_price, recommended_price, pricing_strategy, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (new_id, service_name, current_price, recommended_price, pricing_strategy, confidence_score)
                )
        
        # Insert business hours if provided
        hours_data = data.get("hours", [])
        for hour in hours_data:
            day_of_week = hour.get("day_of_week")
            open_time = hour.get("open_time")
            close_time = hour.get("close_time")
            is_closed = hour.get("is_closed", False)
            
            if day_of_week is not None and 0 <= day_of_week <= 6:
                conn.execute(
                    "INSERT INTO business_hours (business_id, day_of_week, open_time, close_time, is_closed) VALUES (?, ?, ?, ?, ?)",
                    (new_id, day_of_week, open_time, close_time, is_closed)
                )
        
        conn.commit()
    return jsonify({"id": new_id}), 201

@bp.route("/businesses/<int:biz_id>/images", methods=["POST"])
@require_auth
def upload_business_image(biz_id):
    # Check if business exists
    with get_db() as conn:
        business = conn.execute("SELECT id FROM businesses WHERE id = ?", (biz_id,)).fetchone()
        if not business:
            return jsonify({"error": "Business not found"}), 404
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Generate filename and path
    filename = secure_filename(file.filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    image_url = f"/uploads/{unique_filename}"
    
    try:
        # Save the file first
        file.save(filepath)
        
        # Verify file was actually saved
        if not os.path.exists(filepath):
            return jsonify({"error": "Failed to save file"}), 500
        
        # Only then update database
        with get_db() as conn:
            conn.execute(
                "INSERT INTO business_images (business_id, image_url) VALUES (?, ?)",
                (biz_id, image_url)
            )
            conn.commit()
            image_id = conn.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
        
        return jsonify({"id": image_id, "image_url": image_url}), 201
        
    except Exception as e:
        # If database insert fails, clean up the file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass  # Log this in production
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

@bp.route("/businesses/<int:biz_id>/images", methods=["GET"])
def get_business_images(biz_id):
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM business_images WHERE business_id = ?", (biz_id,))
        images = [dict(row) for row in cursor.fetchall()]
    return jsonify(images), 200

@bp.route("/businesses/<int:biz_id>/owner-check", methods=["GET"])
@require_auth
def check_business_owner(biz_id):
    """Check if the current user is the owner of the business"""
    user_id = request.user_id
    
    with get_db() as conn:
        business = conn.execute(
            "SELECT owner_id FROM businesses WHERE id = ?",
            (biz_id,)
        ).fetchone()
        
        if not business:
            return jsonify({"error": "Business not found"}), 404
        
        is_owner = business["owner_id"] == user_id
        return jsonify({"isOwner": is_owner}), 200

@bp.route("/businesses/<int:biz_id>", methods=["PATCH"])
@require_auth
def update_business(biz_id):
    """Update business information"""
    data = request.get_json(force=True)
    
    with get_db() as conn:
        # Check if business exists and user owns it
        business = conn.execute(
            "SELECT owner_id FROM businesses WHERE id = ?", 
            (biz_id,)
        ).fetchone()
        
        if not business:
            return jsonify({"error": "Business not found"}), 404
        
        if business["owner_id"] != request.user_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        # Update allowed fields
        allowed_fields = ["name", "category", "description", "services", "image_url", "location", "socials"]
        update_data = {}
        
        for field in allowed_fields:
            if field in data:
                if field == "socials":
                    update_data[field] = json.dumps(data[field])
                else:
                    update_data[field] = data[field]
        
        if update_data:
            set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
            values = list(update_data.values()) + [biz_id]
            
            conn.execute(f"UPDATE businesses SET {set_clause} WHERE id = ?", values)
            conn.commit()
        
        return jsonify({"message": "Business updated successfully"}), 200

@bp.route("/businesses/<int:biz_id>/images/<int:image_id>", methods=["DELETE"])
@require_auth
def delete_business_image(biz_id, image_id):
    try:
        with get_db() as conn:
            # Get the image to delete
            image = conn.execute(
                "SELECT * FROM business_images WHERE id = ? AND business_id = ?", 
                (image_id, biz_id)
            ).fetchone()
            
            if not image:
                return jsonify({"error": "Image not found"}), 404
            
            # Delete from database first
            conn.execute("DELETE FROM business_images WHERE id = ?", (image_id,))
            conn.commit()
            
            # Then try to delete the file
            image_dict = dict(image)
            filepath = os.path.join(UPLOAD_FOLDER, image_dict["image_url"].split("/")[-1])
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as e:
                    # Log this in production, but don't fail the request
                    print(f"Warning: Failed to delete file {filepath}: {e}")
        
        return jsonify({"message": "Image deleted successfully"}), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to delete image: {str(e)}"}), 500

@bp.route("/businesses/<int:biz_id>/hours", methods=["GET"])
def get_business_hours(biz_id):
    """Get business hours for a specific business"""
    with get_db() as conn:
        # Check if business exists
        business = conn.execute("SELECT id FROM businesses WHERE id = ?", (biz_id,)).fetchone()
        if not business:
            return jsonify({"error": "Business not found"}), 404
        
        # Get business hours
        hours = conn.execute(
            "SELECT day_of_week, open_time, close_time, is_closed FROM business_hours WHERE business_id = ? ORDER BY day_of_week",
            (biz_id,)
        ).fetchall()
        
        # Convert to list of dictionaries
        hours_list = []
        for hour in hours:
            hours_list.append({
                "day_of_week": hour["day_of_week"],
                "open_time": hour["open_time"],
                "close_time": hour["close_time"],
                "is_closed": bool(hour["is_closed"])
            })
        
        return jsonify(hours_list), 200

@bp.route("/businesses/<int:biz_id>/hours", methods=["POST"])
@require_auth
def set_business_hours(biz_id):
    """Set business hours for a specific business"""
    data = request.get_json(force=True)
    
    with get_db() as conn:
        # Check if business exists and user owns it
        business = conn.execute(
            "SELECT owner_id FROM businesses WHERE id = ?", 
            (biz_id,)
        ).fetchone()
        
        if not business:
            return jsonify({"error": "Business not found"}), 404
        
        if business["owner_id"] != request.user_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        # Delete existing hours
        conn.execute("DELETE FROM business_hours WHERE business_id = ?", (biz_id,))
        
        # Insert new hours
        hours_data = data.get("hours", [])
        for hour in hours_data:
            day_of_week = hour.get("day_of_week")
            open_time = hour.get("open_time")
            close_time = hour.get("close_time")
            is_closed = hour.get("is_closed", False)
            
            if day_of_week is not None and 0 <= day_of_week <= 6:
                conn.execute(
                    "INSERT INTO business_hours (business_id, day_of_week, open_time, close_time, is_closed) VALUES (?, ?, ?, ?, ?)",
                    (biz_id, day_of_week, open_time, close_time, is_closed)
                )
        
        conn.commit()
        return jsonify({"message": "Business hours updated successfully"}), 200

# Authentication endpoints
@bp.route("/auth/register", methods=["POST"])
def register():
    """Register a new user"""
    data = request.get_json(force=True)
    
    if not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password are required"}), 400
    
    email = data["email"].lower().strip()
    password = data["password"]
    
    # Basic email validation
    if "@" not in email or "." not in email:
        return jsonify({"error": "Invalid email format"}), 400
    
    # Password validation
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long"}), 400
    
    try:
        with get_db() as conn:
            # Check if user already exists
            existing_user = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            
            if existing_user:
                return jsonify({"error": "User with this email already exists"}), 409
            
            # Hash password and create user
            password_hash = hash_password(password)
            
            conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email, password_hash)
            )
            conn.commit()
            
            # Get the new user
            user = conn.execute(
                "SELECT id, email FROM users WHERE email = ?", (email,)
            ).fetchone()
            
            # Generate token
            token = generate_token(user["id"], user["email"])
            
            return jsonify({
                "message": "User registered successfully",
                "token": token,
                "user": {
                    "id": user["id"],
                    "email": user["email"]
                }
            }), 201
            
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@bp.route("/auth/login", methods=["POST"])
def login():
    """Login user"""
    data = request.get_json(force=True)
    
    if not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password are required"}), 400
    
    email = data["email"].lower().strip()
    password = data["password"]
    
    try:
        with get_db() as conn:
            user = conn.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?", (email,)
            ).fetchone()
            
            if not user:
                return jsonify({"error": "Invalid email or password"}), 401
            
            # Verify password
            if not verify_password(password, user["password_hash"]):
                return jsonify({"error": "Invalid email or password"}), 401
            
            # Generate token
            token = generate_token(user["id"], user["email"])
            
            return jsonify({
                "message": "Login successful",
                "token": token,
                "user": {
                    "id": user["id"],
                    "email": user["email"]
                }
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"Login failed: {str(e)}"}), 500

@bp.route("/auth/verify", methods=["GET"])
@require_auth
def verify_token_endpoint():
    """Verify if the current token is valid"""
    user_id = request.user_id
    
    try:
        with get_db() as conn:
            user = conn.execute(
                "SELECT id, email FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            return jsonify({
                "valid": True,
                "user": {
                    "id": user["id"],
                    "email": user["email"]
                }
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500

# Review endpoints
@bp.route("/reviews", methods=["POST"])
@require_auth
def add_review():
    """Add a review for a business"""
    data = request.get_json(force=True)
    
    required_fields = ["business_id", "rating", "text"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    
    business_id = data["business_id"]
    rating = data["rating"]
    text = data["text"]
    user_id = request.user_id
    name = data.get("name", "Anonymous")
    ip_address = request.remote_addr
    
    # Validate rating
    if not (1 <= rating <= 5):
        return jsonify({"error": "Rating must be between 1 and 5"}), 400
    
    try:
        with get_db() as conn:
            # Check if business exists
            business = conn.execute(
                "SELECT id, owner_id FROM businesses WHERE id = ?", (business_id,)
            ).fetchone()
            
            if not business:
                return jsonify({"error": "Business not found"}), 404
            
            # Prevent owners from reviewing their own businesses
            if business["owner_id"] == user_id:
                return jsonify({"error": "Business owners cannot review their own businesses"}), 403
            
            # Check if user already reviewed this business
            existing_review = conn.execute(
                "SELECT id FROM reviews WHERE business_id = ? AND user_id = ?",
                (business_id, user_id)
            ).fetchone()
            
            if existing_review:
                return jsonify({"error": "You have already reviewed this business"}), 409
            
            # Add the review
            conn.execute(
                "INSERT INTO reviews (business_id, user_id, name, rating, text, ip_address) VALUES (?, ?, ?, ?, ?, ?)",
                (business_id, user_id, name, rating, text, ip_address)
            )
            
            # Update business rating
            avg_rating_result = conn.execute(
                "SELECT AVG(rating) as avg_rating, COUNT(*) as total_reviews FROM reviews WHERE business_id = ?",
                (business_id,)
            ).fetchone()
            
            avg_rating = avg_rating_result["avg_rating"] or 0
            total_reviews = avg_rating_result["total_reviews"] or 0
            
            conn.execute(
                "UPDATE businesses SET rating = ?, total_reviews = ? WHERE id = ?",
                (avg_rating, total_reviews, business_id)
            )
            
            conn.commit()
            
            # Get the new review
            review_id = conn.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
            
            return jsonify({
                "message": "Review added successfully",
                "review_id": review_id
            }), 201
            
    except Exception as e:
        return jsonify({"error": f"Failed to add review: {str(e)}"}), 500

@bp.route("/businesses/<int:biz_id>/reviews", methods=["GET"])
def get_business_reviews(biz_id):
    """Get reviews for a specific business"""
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))
    offset = (page - 1) * limit
    
    try:
        with get_db() as conn:
            # Check if business exists
            business = conn.execute(
                "SELECT id FROM businesses WHERE id = ?", (biz_id,)
            ).fetchone()
            
            if not business:
                return jsonify({"error": "Business not found"}), 404
            
            # Get reviews with pagination
            reviews = conn.execute(
                """
                SELECT r.*, u.email as user_email 
                FROM reviews r 
                LEFT JOIN users u ON r.user_id = u.id 
                WHERE r.business_id = ? 
                ORDER BY r.created_at DESC 
                LIMIT ? OFFSET ?
                """,
                (biz_id, limit, offset)
            ).fetchall()
            
            # Get total count
            total_count = conn.execute(
                "SELECT COUNT(*) as count FROM reviews WHERE business_id = ?",
                (biz_id,)
            ).fetchone()["count"]
            
            # Convert to list of dictionaries
            reviews_list = []
            for review in reviews:
                review_dict = dict(review)
                # Don't expose email in reviews
                review_dict.pop("user_email", None)
                reviews_list.append(review_dict)
            
            return jsonify({
                "reviews": reviews_list,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total_count,
                    "pages": (total_count + limit - 1) // limit
                }
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"Failed to get reviews: {str(e)}"}), 500

@bp.route("/reviews/<int:review_id>", methods=["DELETE"])
@require_auth
def delete_review(review_id):
    """Delete a review (only by the user who created it)"""
    user_id = request.user_id
    
    try:
        with get_db() as conn:
            # Get the review
            review = conn.execute(
                "SELECT id, business_id, user_id FROM reviews WHERE id = ?",
                (review_id,)
            ).fetchone()
            
            if not review:
                return jsonify({"error": "Review not found"}), 404
            
            # Check if user owns the review
            if review["user_id"] != user_id:
                return jsonify({"error": "You can only delete your own reviews"}), 403
            
            business_id = review["business_id"]
            
            # Delete the review
            conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
            
            # Update business rating
            avg_rating_result = conn.execute(
                "SELECT AVG(rating) as avg_rating, COUNT(*) as total_reviews FROM reviews WHERE business_id = ?",
                (business_id,)
            ).fetchone()
            
            avg_rating = avg_rating_result["avg_rating"] or 0
            total_reviews = avg_rating_result["total_reviews"] or 0
            
            conn.execute(
                "UPDATE businesses SET rating = ?, total_reviews = ? WHERE id = ?",
                (avg_rating, total_reviews, business_id)
            )
            
            conn.commit()
            
            return jsonify({"message": "Review deleted successfully"}), 200
            
    except Exception as e:
        return jsonify({"error": f"Failed to delete review: {str(e)}"}), 500

# Additional review endpoints for frontend compatibility
@bp.route("/reviews/<int:business_id>", methods=["GET"])
def get_reviews_by_business_id(business_id):
    """Get reviews for a business (alternative endpoint for frontend compatibility)"""
    return get_business_reviews(business_id)

@bp.route("/reviews/average/<int:business_id>", methods=["GET"])
def get_business_average_rating(business_id):
    """Get average rating for a specific business"""
    try:
        with get_db() as conn:
            # Check if business exists
            business = conn.execute(
                "SELECT id FROM businesses WHERE id = ?", (business_id,)
            ).fetchone()
            
            if not business:
                return jsonify({"error": "Business not found"}), 404
            
            # Get average rating and review count
            result = conn.execute(
                "SELECT AVG(rating) as average_rating, COUNT(*) as total_reviews FROM reviews WHERE business_id = ?",
                (business_id,)
            ).fetchone()
            
            average_rating = result["average_rating"] or 0
            total_reviews = result["total_reviews"] or 0
            
            return jsonify({
                "business_id": business_id,
                "average_rating": round(average_rating, 2),
                "total_reviews": total_reviews
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"Failed to get average rating: {str(e)}"}), 500

