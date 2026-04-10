from flask import Flask, request, render_template
from pickle import load
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Subimos un nivel (.parent) para salir de 'src' y entrar a 'models'
model_path = BASE_DIR.parent / "models" / "decision_tree_classifier_default_42.sav"


app = Flask(__name__)


model = load(open(model_path, "rb"))


class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        
        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction=pred_class)


if __name__ == "__main__":
    app.run(debug=True)
