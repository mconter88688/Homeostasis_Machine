# Use Java image (PlantUML runs on Java)
FROM openjdk:17-jdk-slim

# Install Graphviz (required for diagram rendering)
RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*

# Create a working directory in the container
WORKDIR /workspace

# Install PlantUML
RUN apt-get update && apt-get install -y plantuml && rm -rf /var/lib/apt/lists/*

# By default, open a shell
CMD ["bash"]
