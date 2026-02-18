from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from src.database.db import UserDBHandler

# Initialize database handler
db_handler = UserDBHandler()

# JWT Configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Functions
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db_handler: UserDBHandler, username: str):
    # Query the database for the user
    user = db_handler.read_authentication_by_login(username)
    if user:
        return {
            "username": user[1],  # login is at index 1
            "hashed_password": user[2],  # password is at index 2
            "role": user[3],  # role is at index 3
            "user_id": user[7]  # user_id is at index 7
        }

def authenticate_user(db_handler: UserDBHandler, username: str, password: str):
    user = get_user(db_handler, username)
    if not user:
        return False
    # # Truncate the password if it exceeds the maximum length
    # if len(password) > 72:
    #     password = password[:72]
    # if not verify_password(password, user["hashed_password"]):
    #     return False
    # return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(datetime.UTC) + expires_delta
    else:
        expire = datetime.now(datetime.UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(db_handler, username=username)
    if user is None:
        raise credentials_exception
    return user

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/ask")
async def ask(query: str):
    if not agent and not pipeline:
        return {"error": "Neither agent nor pipeline initialized"}
    if agent:
        result = agent.process(query)
    else:
        result = pipeline.process(query)
    return {
        "intent": str(result.intent),
        "recommendation": result.llm_recommendation,
        "constraints": result.constraints
    }

@router.post("/auth")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db_handler, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

@router.get("/nutrition_history")
async def get_nutrition_history(current_user: dict = Depends(get_current_user)):
    """
    Retrieve nutrition history for the current user.

    Returns:
        List[dict]: A list of nutrition history records.
    """
    user_id = current_user["user_id"]
    nutrition_history = db_handler.read_nutrition_history_by_user(user_id)
    return nutrition_history

@router.get("/recipe_history")
async def get_recipe_history(current_user: dict = Depends(get_current_user)):
    """
    Retrieve recipe history for the current user.

    Returns:
        List[dict]: A list of recipe history records.
    """
    user_id = current_user["user_id"]
    recipe_history = db_handler.read_recipe_history_by_user(user_id)
    return recipe_history

