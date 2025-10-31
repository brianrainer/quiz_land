from app import create_app, db
from flask import render_template
from app.models import User, Subject, Quiz, QuizQuestion, QuizChoice, QuizScore

app = create_app()

@app.cli.command('db-create')
def create_db():
    db.create_all()
    print("Database created!")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
