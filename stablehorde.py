# TODO 
# Implment Stable Horde API
# https://stablehorde.net/api/v2/docs 
# This is a work in progress
# This file is part of the AI Image Generator project
# API key is meant to be set in the environment variable STABLE_HORDE_API_KEY in the .env file
# env file is not created in this repo for security reasons, but should be created by the install script
# .env file should contain the line STABLE_HORDE_API_KEY=your_api_key
# Make sure to add .env to your .gitignore file to avoid committing it to version control
# You can get an API key by creating an account on https://stablehorde.net and going to https://stablehorde.net/user/settings
# You can also use the API without an API key, but you will be limited
# the install script should prompt the user to enter their API key and create the .env file if.
# if the user does not want to enter an API key, the script should create an empty .env file without the STABLE_HORDE_API_KEY=your_api_key line
# The API key is optional, but recommended for better performance and to avoid rate limiting

# this stablehorde.py should be called by the main.py file when the user selects Stable Horde as the AI model
# the main.py file should import the StableHorde class from this file and create an instance
# This way the image generation can be handled in a modular way and not clutter the main.py file or be hardcoded