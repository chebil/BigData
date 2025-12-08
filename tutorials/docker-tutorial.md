# Docker Tutorial for Data Science

## Basic Docker Commands

```bash
# Build image
docker build -t bigdata:latest .

# Run container
docker run -p 8888:8888 bigdata:latest

# List containers
docker ps
docker ps -a  # Include stopped

# Stop container
docker stop container_id

# Remove container
docker rm container_id

# Remove image
docker rmi image_id
```

## Docker Compose

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs
```

## Dockerfile Example

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0"]
```
