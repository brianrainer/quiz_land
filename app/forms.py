from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, SelectField, IntegerField, DateTimeLocalField, BooleanField
from wtforms.validators import Email, Length, EqualTo, DataRequired

class RegisterForm(FlaskForm):
    username = StringField('Email', validators=[Email(), DataRequired()])
    password = PasswordField('Password', validators=[Length(min=8), DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[EqualTo('password')])
    fullname = StringField('Full Name')
    submit = SubmitField('Submit')

class LoginForm(FlaskForm):
    username = StringField('Email', validators=[Email(), DataRequired()])
    password = PasswordField('Password', validators=[Length(min=8), DataRequired()])
    submit = SubmitField('Submit')

class SubjectForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    description = TextAreaField('Description')
    submit = SubmitField('Submit')

class QuizForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    description = TextAreaField('Description')
    date_of_quiz = DateTimeLocalField('Date of Quiz', validators=[DataRequired()])
    duration = IntegerField('Time Duration (in seconds)', validators=[DataRequired()])
    subject_id = SelectField('Subject', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Submit')

class QuizQuestionForm(FlaskForm):
    question_statement = StringField('Question Statement', validators=[DataRequired()])
    quiz_id = SelectField('Quiz', coerce=int, validators=[DataRequired()])
    option_1 = StringField('Option 1', validators=[DataRequired()])
    option_2 = StringField('Option 2', validators=[DataRequired()])
    option_3 = StringField('Option 3', validators=[DataRequired()])
    option_4 = StringField('Option 4', validators=[DataRequired()])
    correct_option = StringField('Answer', validators=[DataRequired()])
    submit = SubmitField('Submit')

class QuizChoiceForm(FlaskForm):
    description = StringField('Description', validators=[DataRequired()])
    is_correct = BooleanField('Is Correct', default=False)
    submit = SubmitField('Submit')
