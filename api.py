from typing import Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from query_autocomplete import AutocompleteQuery

# instance of AutocompleteQuery class
autocomplete_obj = AutocompleteQuery(sents_list=[], dicts_pth='./data/autocomplete_query')


class RequestBody(BaseModel):
    query: str
    branches: Optional[int] = 2
    levels: Optional[int] = 2


app = FastAPI()

@app.post("/query_autocomplete/")
async def root(request_body: RequestBody):
        
    psbl_queries = autocomplete_obj.autocomplete_query(request_body.query, branches=request_body.branches, levels=request_body.levels)
    
    return psbl_queries


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8091, reload=True)
    