from functools import wraps

from flask import render_template, redirect, flash, url_for, request
from flask_login import login_user, login_required, logout_user, current_user

from app import create_app, db, login_manager
from app.models import User, Subject, Quiz, QuizQuestion, QuizScore
from app.forms import RegisterForm, LoginForm, SubjectForm, QuizForm, QuizQuestionForm, QuizChoiceForm
from weather import get_three_days_weather

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
            flash("You don't have permission to access that page", category="error")
            return redirect(url_for('dashboard'))
        return func(*args, **kwargs)
    return decorated_view

@app.route("/", methods=['GET', 'POST'])
def home():
    weather_data = None
    city_name = ""
    error_message = None

    if request.method == 'POST':
        # Get the city name from the submitted form data
        city_name = request.form.get('cityName')
        if city_name:
            weather_data = get_three_days_weather(city_name)
            if not weather_data:
                error_message = f"Could not find weather data for '{city_name}'. Please try a different city."
        else:
            error_message = "Please enter a city name."

    if error_message:
        flash(error_message, category='error')
    # Render the HTML template, passing the data to be displayed
    return render_template('home.html', weather_data=weather_data)

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
    quizzes = Quiz.query.all()
    return render_template("dashboard.html", quizzes=quizzes)

@app.route("/admin/user")
@admin_role_required
@login_required
def manage_users():
    users = User.query.all()
    return render_template('admin/manage_users.html', data_label="Users", data=users)

@app.route("/admin/subject")
@admin_role_required
@login_required
def manage_subjects():
    subjects = Subject.query.all()
    return render_template('admin/manage_subjects.html', data_label="Subjects", data=subjects)

@app.route("/admin/add/subject", methods=['GET', 'POST'])
@admin_role_required
@login_required
def add_subject():
    subject_form = SubjectForm()
    if subject_form.validate_on_submit():
        new_subject = Subject(
            name=subject_form.name.data,
            description=subject_form.description.data
        )
        db.session.add(new_subject)
        db.session.commit()
        flash('Subject successfully created!', category='success')
        return redirect(url_for('manage_subjects'))
    return render_template('admin/add_subject.html', form=subject_form)

@app.route('/admin/subject/<int:subject_id>', methods=['GET', 'POST'])
@admin_role_required
@login_required
def edit_subject(subject_id):
    subject = Subject.query.get_or_404(subject_id)
    subject_form = SubjectForm(obj=subject)
    if subject_form.validate_on_submit():
        subject.name = subject_form.name.data
        subject.description = subject_form.description.data
        db.session.commit()
        flash('Subject successfully updated!', category='success')
        return redirect(url_for('manage_subjects'))
    return render_template('admin/edit_subject.html', form=subject_form, data=subject, subject_id=subject_id)

@app.route('/admin/delete/subject/<int:subject_id>', methods=['POST'])
@admin_role_required
@login_required
def delete_subject(subject_id):
    subject = Subject.query.get_or_404(subject_id)
    db.session.delete(subject)
    db.session.commit()
    flash('Subject successfully deleted!', category='success')
    return redirect(url_for('manage_subjects'))

@app.route("/admin/quiz")
@admin_role_required
@login_required
def manage_quizzes():
    quizzes = Quiz.query.all()
    return render_template('admin/manage_quizzes.html', data_label="Quizzes", data=quizzes)

@app.route('/admin/add/quiz', methods=['GET', 'POST'])
@admin_role_required
@login_required
def add_quiz():
    quiz_form = QuizForm()
    quiz_form.subject_id.choices = [(s.id, s.name) for s in Subject.query.all()]
    if quiz_form.validate_on_submit():
        new_quiz = Quiz(
            name=quiz_form.name.data,
            description=quiz_form.description.data,
            date_of_quiz=quiz_form.date_of_quiz.data,
            duration=quiz_form.duration.data,
            subject_id=quiz_form.subject_id.data
        )
        db.session.add(new_quiz)
        db.session.commit()
        flash('Quiz successfully created!', category='success')
        return redirect(url_for('manage_quizzes'))
    return render_template('admin/add_quiz.html', form=quiz_form)


@app.route('/admin/quiz/<int:quiz_id>', methods=['GET', 'POST'])
@admin_role_required
@login_required
def edit_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    quiz_form = QuizForm(obj=quiz)
    quiz_form.subject_id.choices = [(s.id, s.name) for s in Subject.query.all()]
    if quiz_form.validate_on_submit():
        quiz.name = quiz_form.name.data
        quiz.description = quiz_form.description.data
        quiz.date_of_quiz = quiz_form.date_of_quiz.data
        quiz.duration = quiz_form.duration.data
        quiz.subject_id = quiz_form.subject_id.data
        db.session.commit()
        flash('Quiz updated!', category='success')
        return redirect(url_for('manage_quizzes'))
    return render_template('admin/edit_quiz.html', form=quiz_form, data=quiz, quiz_id=quiz_id)

@app.route('/admin/delete/quiz/<int:quiz_id>', methods=['POST'])
@admin_role_required
@login_required
def delete_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    db.session.delete(quiz)
    db.session.commit()
    flash('Quiz deleted!', category='success')
    return redirect(url_for('manage_quizzes'))

@app.route("/admin/question")
@admin_role_required
@login_required
def manage_questions():
    questions = QuizQuestion.query.all()
    return render_template('admin/manage_questions.html', data_label="Quiz Questions", data=questions)

@app.route("/admin/quiz/<int:quiz_id>/question")
@admin_role_required
@login_required
def manage_questions_by_quiz(quiz_id):
    questions = QuizQuestion.query.filter_by(quiz_id=quiz_id)
    return render_template('admin/manage_questions.html', data_label='Quiz Questions', data=questions)

@app.route('/admin/add/question', methods=['GET', 'POST'])
@admin_role_required
@login_required
def add_question():
    question_form = QuizQuestionForm()
    question_form.quiz_id.choices = [(q.id, q.name) for q in Quiz.query.all()]
    if question_form.validate_on_submit():
        new_question = QuizQuestion(
            question_statement=question_form.question_statement.data,
            quiz_id=question_form.quiz_id.data,
            option_1 = question_form.option_1.data,
            option_2 = question_form.option_2.data,
            option_3 = question_form.option_3.data,
            option_4 = question_form.option_4.data,
            correct_option = question_form.correct_option.data
        )
        db.session.add(new_question)
        db.session.commit()
        flash('Question created!', category='success')
        return redirect(url_for('manage_questions'))
    return render_template('admin/add_question.html', form=question_form)

@app.route('/admin/delete/question/<int:question_id>', methods=['POST'])
@admin_role_required
@login_required
def delete_question(question_id):
    question = QuizQuestion.query.get_or_404(question_id)
    quiz_id = question.quiz_id
    db.session.delete(question)
    db.session.commit()
    flash('Question deleted!', category='success')
    return redirect(url_for('manage_questions_by_quiz', quiz_id=quiz_id))

@app.route('/quiz/<int:quiz_id>', methods=['GET', 'POST'])
@login_required
def attempt_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    questions = quiz.questions
    total_questions = len(questions)

    if request.method == "POST":
        correct = 0
        for question in questions:
            user_answer = request.form.get(f'question_{question.id}')
            if user_answer and user_answer == question.correct_option:
                correct += 1
        
        total_scored = (correct // total_questions) * 100

        previous_score = QuizScore.query.filter_by(
            user_id=current_user.id,
            quiz_id=quiz_id
        ).first()
        if not previous_score:
            user_score = QuizScore(
                total_scored=total_scored,
                quiz_id=quiz_id,
                user_id=current_user.id
            )
            db.session.add(user_score)
        else:
            previous_score.total_scored = max(previous_score.total_scored, total_scored)
        db.session.commit()
        flash(f'Your score is: {total_scored}', category='success')
        return redirect(url_for('results_by_quiz', quiz_id=quiz_id))
    return render_template('attempt_quiz.html', quiz_id=quiz_id, quiz=quiz)


@app.route('/results/quiz/<int:quiz_id>')
@login_required
def results_by_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    score = QuizScore.query.filter_by(user_id=current_user.id, quiz_id=quiz_id).first()
    return render_template('quiz_results.html', quiz=quiz, score=score)



if __name__ == "__main__":
    app.run(debug=True)
