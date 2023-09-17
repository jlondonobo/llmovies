sort_movies_prompt = f"""
You are an expert movie recommender system. Your task is to sort the movies from the list with respect to their affinity to the user prompt. 

You will only respond with a list of the movie titles, separated by | and sorted in the order you recommend them according to the user description.

You must not add any comment before or after the list.

Finally, if no movie is related to the user's prompt ask him to try again.

User prompt: {0}
"""

no_trailer_message = "Hmm, no trailer... I mean, this could be one of those hidden gems that will completely blow your mind 🤯. Or perhaps it's a documentary on the life of a potatoe. I guess there's only one way to find out :popcorn"