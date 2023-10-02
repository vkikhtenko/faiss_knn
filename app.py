from fastapi import FastAPI
import uvicorn
from typing import Union
import numpy as np
import faiss
import pickle
import json

app = FastAPI()
dims = 66
faiss_index = None


def parse_string(vec: str) -> list[float]:
    l = vec.split(",")
    if len(l) != dims:
        return None
    return [str(el) for el in l]


@app.on_event("startup")
def start():
    global faiss_index
    global base_index_read
    with open('faiss_index.pkl', 'rb') as f:
        faiss_index = pickle.load(f)
    with open('base_index.txt', 'r') as f:
        base_index_read = json.load(f)

@app.get("/")
def main() -> dict:
    return {"status": "OK", "message": "It works! Finally!"}


@app.get("/knn")
def match(item: Union[str, None] = None) -> dict:
    global faiss_index
    if item is None:
        return {"status": "fail", "message": "No input data"}

    vec = parse_string(item)
    if vec is None:
        return {"status": "fail", "message": "Invalid input string format"}
    vec = np.ascontiguousarray(vec, dtype=np.float32)[np.newaxis, :]
    faiss_index.nprobe = 300
    knn, idx = faiss_index.search(vec, k=5)
    m = idx
 #   m = m.strip('[]').split()
    m = [int(num) for num in m[0]]
    recom = []
    for i in m:
        recom.append(base_index_read[str(i)])
    return {"status": "OK", "data": recom}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8031)