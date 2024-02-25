from flask import Flask, render_template, request
from llm import search

app = Flask(__name__)


# Placeholder for your backend service logic (replace with your actual implementation)
def call_backend_service(text):
    return f"Backend returned: {text}"


@app.route("/", methods=["GET", "POST"])
def index():
    submitted_text = ""
    response_text = ""

    if request.method == "POST":
        submitted_text = request.form.get("user_text")
        context = request.form.get("context")
        response_text = search(query=submitted_text, filter={"context": context})

    return render_template("index.html", submitted_text=submitted_text, response_text=response_text)


if __name__ == "__main__":
    app.run(debug=True)
