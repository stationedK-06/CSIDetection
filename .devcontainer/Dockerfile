#Base Image
From python:3.13.5

# Set the working directory
WORKDIR /app

#pip version fix
RUN python -m pip install --upgrade pip==25.1.1

# Copy the requirements file into the container at /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

CMD ["python", "app/main.py"]
