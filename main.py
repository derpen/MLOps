from derpen import DerpenData

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

dpn = DerpenData()

db = {
    "train": False,
    "input": "",
    "save" : False
    }

@app.get("/", response_class=HTMLResponse)
def read_item(request: Request):
    return templates.TemplateResponse("front_page.html", {
                                            "request": request, 
                                            "train": db["train"], 
                                            "input": db["input"],
                                            "save": db["save"]})

@app.get("/train", response_class=HTMLResponse)
def train_data():
    dpn.train_data()
    db["train"] = True
    return RedirectResponse(url="/")

@app.post("/predict", response_class=HTMLResponse)
def predict(data: str = Form(...)):
    predict_result = dpn.predict(data)
    db["input"] = predict_result
    return RedirectResponse(url="/", status_code=303)

@app.get("/save", response_class=HTMLResponse)
def save(request: Request):
    dpn.save_model()
    db["save"] = True
    return RedirectResponse(url="/")

@app.get("/reset", response_class=HTMLResponse)
def reset_dpn():
    # Reinitialize the dpn object to reset it
    global dpn
    dpn = DerpenData()

    # Reset the state in the db dictionary
    db["train"] = False
    db["input"] = ""
    db["save"] = False

    return RedirectResponse(url="/")