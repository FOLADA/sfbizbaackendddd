#!/usr/bin/env python3
"""
Script to clean up reviews made by business owners on their own businesses.
This script identifies and removes such reviews to maintain review integrity.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("businesses.db")

def cleanup_owner_reviews():
    """Remove reviews made by business owners on their own businesses"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        # Find reviews where the reviewer is also the business owner
        cursor = conn.execute("""
            SELECT r.id, r.business_id, r.user_id, r.name, r.rating, r.text, r.created_at,
                   b.name as business_name, u.email as user_email
            FROM reviews r
            JOIN businesses b ON r.business_id = b.id
            JOIN users u ON r.user_id = u.id
            WHERE r.user_id = b.owner_id
        """)
        
        owner_reviews = cursor.fetchall()
        
        if not owner_reviews:
            print("✅ No reviews found where business owners reviewed their own businesses.")
            return
        
        print(f"⚠️  Found {len(owner_reviews)} reviews made by business owners on their own businesses:")
        print("-" * 80)
        
        for review in owner_reviews:
            print(f"Review ID: {review['id']}")
            print(f"Business: {review['business_name']} (ID: {review['business_id']})")
            print(f"Owner: {review['user_email']} (ID: {review['user_id']})")
            print(f"Rating: {review['rating']}/5")
            print(f"Text: {review['text'][:100]}{'...' if len(review['text']) > 100 else ''}")
            print(f"Created: {review['created_at']}")
            print("-" * 80)
        
        # Ask for confirmation
        response = input(f"\nDo you want to delete these {len(owner_reviews)} reviews? (y/N): ")
        
        if response.lower() == 'y':
            # Delete the reviews
            review_ids = [review['id'] for review in owner_reviews]
            placeholders = ','.join(['?' for _ in review_ids])
            
            conn.execute(f"DELETE FROM reviews WHERE id IN ({placeholders})", review_ids)
            conn.commit()
            
            print(f"✅ Successfully deleted {len(owner_reviews)} reviews made by business owners.")
            
            # Update business ratings
            print("🔄 Updating business ratings...")
            update_business_ratings(conn)
            
        else:
            print("❌ Cleanup cancelled.")

def update_business_ratings(conn):
    """Update business ratings after removing owner reviews"""
    # Get all businesses that have reviews
    cursor = conn.execute("""
        SELECT DISTINCT business_id 
        FROM reviews
    """)
    
    business_ids = [row['business_id'] for row in cursor.fetchall()]
    
    for business_id in business_ids:
        # Calculate new average rating
        avg_result = conn.execute("""
            SELECT AVG(rating) as avg_rating, COUNT(*) as total_reviews 
            FROM reviews 
            WHERE business_id = ?
        """, (business_id,)).fetchone()
        
        if avg_result['avg_rating']:
            conn.execute("""
                UPDATE businesses 
                SET rating = ? 
                WHERE id = ?
            """, (round(avg_result['avg_rating'], 1), business_id))
    
    conn.commit()
    print("✅ Business ratings updated.")

def main():
    print("🧹 Business Owner Review Cleanup Script")
    print("=" * 50)
    
    if not DB_PATH.exists():
        print(f"❌ Database file not found: {DB_PATH}")
        return
    
    try:
        cleanup_owner_reviews()
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    main() 