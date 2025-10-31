from app import create_app, db
from flask import render_template, redirect, flash, url_for
from app.models import User, Subject, Quiz, QuizQuestion, QuizChoice, QuizScore
from app.forms import RegisterForm, LoginForm

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

@app.route("/register", methods=['GET', 'POST'])
def register():
    register_form = RegisterForm()
    if register_form.validate_on_submit():
        flash('Registration form is validated', category='success')
        return redirect(url_for('login'))
    return render_template("register.html", form=register_form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    login_form = LoginForm()
    if login_form.validate_on_submit():
        flash('Login form is validated', category='success')
        return redirect(url_for('dashboard'))
    return render_template("login.html", form=login_form)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
