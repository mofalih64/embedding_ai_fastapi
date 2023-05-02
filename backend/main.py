from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from embedding.router import Embedding_router
# from app.routers.auth import authRouter
app = FastAPI()

# app.api_route(path='/users/', operation_id='post', dependencies=register)
app.include_router(Embedding_router, tags=['ASK'])
# app.include_router(userRoute, tags=['Account'])
# app.include_router(inbox_router, tags=['Inbox'])


origins = [
    "*"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#


@app.get("/")
async def root():
    return {"message": "Hello, got to /docs"}
