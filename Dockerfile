# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the docker image
WORKDIR /usr/src/app

# Copy the content of the local src directory to the working directory
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "./app.py"]
