from flask_wtf import FlaskForm #pip install flask_wtf

from wtforms import IntegerField, SelectField, FloatField, SubmitField #different ways of inputting, 
																	   #depending on what kind of data is fetched

class Form(FlaskForm):
	title_length = IntegerField('Title Length') 
	goal_amount = FloatField('Goal Amount') 
	#for SelectField, choices must be a list of tuples where each tuple is (<value sent to app>, <text displayed in html>)
	fifteen_plus_backers = SelectField('Will you have at least fifteen backers?', choices = [('y', 'Yes'), ('n', 'No')])
	main_country = SelectField('Main Country', choices = [('us', 'US'), ('other','Other')])
	main_category = SelectField('Main Category', choices = [
			('art', 'art'), 
			('design', 'design'), 
			('fashion','fashion'), 
			('film_and_video', 'film_and_video'), 
			('food', 'food'), 
			('games', 'games'), 
			('music','music'), 
			('other_category','other_category'), 
			('publishing','publishing'), 
			('technology','technology'), 
			('theater', 'theater')
		])
	project_duration = FloatField('Project Duration')
	#don't forget the SubmitField, which gives the button for submitting the form 
	submit = SubmitField()