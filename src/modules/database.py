# --- Compatibility Patch for Python 3.10+ and older pymongo ---
import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence
# ---------------------------------------------------------------

import pymongo
import certifi
from datetime import datetime
import streamlit as st
import bcrypt
import ssl

# MongoDB Connection String
MONGO_URI = "mongodb+srv://daescucristi03_db_user:MstABkCJSZ8DyDjw@medai-cardiac.yadhrws.mongodb.net/?appName=MedAI-Cardiac"

@st.cache_resource
def init_connection():
    try:
        # Try modern connection first
        try:
            client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where(), connectTimeoutMS=5000)
        except:
            # Fallback for older pymongo versions or if tlsCAFile is unknown
            client = pymongo.MongoClient(MONGO_URI, ssl=True, ssl_cert_reqs=ssl.CERT_NONE, connectTimeoutMS=5000)
            
        # Force a connection check
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"DB Connection Error: {e}")
        return None

# --- User Management (Doctors) ---
def register_doctor(username, password, full_name, specialty):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        users = db['doctors']
        try:
            if users.find_one({"username": username}):
                return False, "Username already exists"
            
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user_record = {
                "username": username,
                "password": hashed_pw,
                "full_name": full_name,
                "specialty": specialty,
                "created_at": datetime.now()
            }
            users.insert_one(user_record)
            return True, "Registration successful"
        except Exception as e:
            return False, f"DB Error: {str(e)}"
    return False, "Database connection failed. Check internet or credentials."

def authenticate_doctor(username, password):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        users = db['doctors']
        try:
            user = users.find_one({"username": username})
            if user:
                if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                    return True, {
                        "full_name": user['full_name'],
                        "specialty": user.get('specialty', 'General'),
                        "username": user['username']
                    }
        except Exception as e:
            st.error(f"Auth Error: {e}")
            return False, None
        return False, None
    return False, None

def get_doctor_by_username(username):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        users = db['doctors']
        try:
            user = users.find_one({"username": username})
            if user:
                return {
                    "full_name": user['full_name'],
                    "specialty": user.get('specialty', 'General'),
                    "username": user['username']
                }
        except:
            return None
    return None

def update_doctor_profile(username, full_name, specialty, new_password=None):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        users = db['doctors']
        
        update_data = {
            "full_name": full_name,
            "specialty": specialty
        }
        
        if new_password:
            hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            update_data["password"] = hashed_pw
            
        try:
            users.update_one({"username": username}, {"$set": update_data})
            return True, "Profile updated successfully"
        except Exception as e:
            return False, str(e)
    return False, "DB Error"

def get_doctor_activity_log(doctor_name):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        collection = db['patient_history']
        try:
            cursor = collection.find({"doctor": doctor_name}, {'_id': 0}).sort("timestamp", -1).limit(100)
            return list(cursor)
        except Exception as e:
            st.error(f"Error fetching logs: {e}")
            return []
    return []

# --- Patient Management (EHR) ---

def add_patient(cnp, first_name, last_name, age, sex, medical_history):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        patients = db['patients']
        
        try:
            if patients.find_one({"cnp": cnp}):
                return False, "Patient with this CNP already exists."
                
            patient_record = {
                "cnp": cnp,
                "first_name": first_name,
                "last_name": last_name,
                "age": int(age),
                "sex": sex,
                "medical_history_notes": medical_history,
                "created_at": datetime.now()
            }
            patients.insert_one(patient_record)
            return True, "Patient registered successfully."
        except Exception as e:
            return False, str(e)
    return False, "DB Error"

def update_patient(cnp, first_name, last_name, age, sex, medical_history):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        patients = db['patients']
        
        update_data = {
            "first_name": first_name,
            "last_name": last_name,
            "age": int(age),
            "sex": sex,
            "medical_history_notes": medical_history
        }
        
        try:
            result = patients.update_one({"cnp": cnp}, {"$set": update_data})
            if result.modified_count > 0:
                return True, "Patient updated successfully."
            return True, "No changes made."
        except Exception as e:
            return False, str(e)
    return False, "DB Error"

def find_patient(search_term):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        patients = db['patients']
        
        query = {
            "$or": [
                {"cnp": {"$regex": search_term, "$options": "i"}},
                {"last_name": {"$regex": search_term, "$options": "i"}},
                {"first_name": {"$regex": search_term, "$options": "i"}}
            ]
        }
        
        return list(patients.find(query, {'_id': 0}))
    return []

def get_all_patients(limit=50):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        patients = db['patients']
        return list(patients.find({}, {'_id': 0}).sort("created_at", -1).limit(limit))
    return []

def get_patient_details(cnp):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        patients = db['patients']
        return patients.find_one({"cnp": cnp}, {'_id': 0})
    return None

# --- Patient History (ECG Records) ---

def save_patient_record(cnp, heart_rate, risk_score, diagnosis, doctor_name):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        collection = db['patient_history']
        
        if not db['patients'].find_one({"cnp": cnp}):
            return False
        
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doctor": doctor_name,
            "patient_cnp": cnp,
            "heart_rate": int(heart_rate),
            "risk_score": float(risk_score),
            "diagnosis": diagnosis
        }
        
        try:
            collection.insert_one(record)
            return True
        except Exception as e:
            st.error(f"Error saving to DB: {e}")
            return False
    return False

def get_patient_history(cnp=None):
    client = init_connection()
    if client:
        db = client['medai_cardiac_db']
        collection = db['patient_history']
        try:
            query = {"patient_cnp": cnp} if cnp else {}
            cursor = collection.find(query, {'_id': 0}).sort("timestamp", -1).limit(100)
            return list(cursor)
        except Exception as e:
            st.error(f"Error fetching history: {e}")
            return []
    return []
