from functools import wraps

from flask import render_template, redirect, flash, url_for
from flask_login import login_user, login_required, logout_user, current_user

from app import create_app, db, login_manager
from app.models import User, Subject, Quiz, QuizQuestion, QuizChoice, QuizScore
from app.forms import RegisterForm, LoginForm

app = create_app()

@app.cli.command('db-create')
def create_db():
    db.create_all()
    print("Database created!")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_role_required(func):
    @wraps(func)
    def decorated_view(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return login_manager.unauthorized()
        return func(*args, **kwargs)
    return decorated_view

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
        new_user = User(
            username=register_form.username.data,
            fullname=register_form.fullname.data,
        )
        new_user.set_password(register_form.password.data)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', category='success')
        return redirect(url_for('login'))
    return render_template("register.html", form=register_form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    login_form = LoginForm()
    if login_form.validate_on_submit():
        user = User.query.filter_by(username=login_form.username.data).first()
        if user and user.check_password(login_form.password.data):
            flash('Login successful!', category='success')
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Authentication failed!', category='error')
    return render_template("login.html", form=login_form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('Logout successful!', category='success')
    return redirect(url_for('home'))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/admin/manage_user")
@admin_role_required
@login_required
def manage_users():
    users = User.query.all()
    return render_template('admin/manage_users.html', data_label="Users", data=users)

@app.route("/admin/manage_subjects")
@admin_role_required
@login_required
def manage_subjects():
    subjects = Subject.query.all()
    return render_template('admin/manage_subjects.html', data_label="Subjects", data=subjects)

@app.route("/admin/manage_quiz.html")
@admin_role_required
@login_required
def manage_quizzes():
    quizzes = Quiz.query.all()
    return render_template('admin/manage_quizzes.html', data_label="Quizzes", data=quizzes)

@app.route("/admin/manage_questions.html")
@admin_role_required
@login_required
def manage_questions():
    questions = QuizQuestion.query.all()
    return render_template('admin/manage_questions.html', data_label="Quiz Questions", data=questions)


if __name__ == "__main__":
    app.run(debug=True)
