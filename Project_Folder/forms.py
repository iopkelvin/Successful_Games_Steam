from flask_wtf import FlaskForm #pip install flask_wtf

from wtforms import IntegerField, SelectField, FloatField, SubmitField #different ways of inputting,
																	   #depending on what kind of data is fetched

class Form(FlaskForm):
	singleplayer = IntegerField('singleplayer')
	steam_cloud = FloatField('steam cloud')
	#for SelectField, choices must be a list of tuples where each tuple is (<value sent to app>, <text displayed in html>)
	indie = SelectField('Is it an indie game?', choices = [('y', 'Yes'), ('n', 'No')])
	anime = SelectField('Is it an anime game?', choices = [('y', 'Yes'), ('n', 'No')])
	multiplayer = SelectField('Multiplayer?', choices = [('y', 'Yes'), ('n', 'No')])

	very_cheap = SelectField('Very cheap?',  choices = [('y', 'Yes'), ('n', 'No')])
	#don't forget the SubmitField, which gives the button for submitting the form
	submit = SubmitField()
