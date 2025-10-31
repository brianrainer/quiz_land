from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import Email, Length, EqualTo

class RegisterForm(FlaskForm):
    username = StringField('Email', validators=[Email()])
    password = PasswordField('Password', validators=[Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[EqualTo('password')])
    fullname = StringField('Full Name')
    submit = SubmitField('Submit')

class LoginForm(FlaskForm):
    username = StringField('Email', validators=[Email()])
    password = PasswordField('Password', validators=[Length(min=8)])
    submit = SubmitField('Submit')
