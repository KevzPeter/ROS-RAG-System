# Finetuned RAG Systems Engineering

> CS-GY 6613 Project: A RAG system that can be used by an ROS2 robotics developer to develop the navigation stack of a agent with egomotion. This means that the robotics is the domain but the system must be particularly helpful (be able to answer very specific questions about the subdomains). The subdomains are:  
> - ros2 robotics middleware subdomain.
> - nav2 navigation subdomain.
> - movit2 motion planning subdomain.
> - gazebo simulation subdomain.

#### Team Members
- Kevin Peter
  - Email: kpk4354@nyu.edu
  - Github: github.com/KevzPeter
  - HuggingFace: https://huggingface.co/kevin-peter

## Project Milestones

### Environment and Tooling Milestone

The docker compose file is available [here](./docker-compose.yaml)

Here's a screenshot of the output of `docker ps` command

<img src="./media/containers.png" height="300px">


### ETL Milestone

We have used `clearml` orchestrator to ingest the following sources:
- Websites (ROS2, Moveit etc)
- Github (ROS2, MoveIT, Gazebo etc)
- Youtube (tutorials)

The ingested data is saved in `MongoDB` database, under 3 collections - `raw_docs`, `github_repo` & `youtube_transcripts`

Collections stored in MongoDB:

<img src="./media/etl/mongo_1.png" height="300px">

[Github Ingester](./github_ingester.py)

A Python script designed to collect and store ROS2-related documentation from various GitHub repositories. The script crawls specified repositories, extracts file contents, and stores them in MongoDB for later use in vector embeddings and RAG (Retrieval-Augmented Generation) applications.
Features
- Recursive repository content extraction
- Automatic handling of directories and files
- MongoDB integration with duplicate prevention
- ClearML tracking for monitoring and metrics
- Size limit handling (16MB max per document)

Github Data in DB:

<img src="./media/etl/github_data.png" height="300px">

Target Repositories
- ROS2 core repositories
- Navigation2 documentation
- MoveIt2 resources
- Gazebo simulation documentation

[Youtube Ingester](./youtube_ingester.py)

A Python script that automatically searches for ROS2-related YouTube videos using Google's Client Discovery API, extracts their transcripts, and stores them in MongoDB. This tool is part of a larger RAG system, collecting educational content about ROS2, navigation, MoveIt2, and Gazebo.

Features
- Automated YouTube video search
- Transcript extraction and processing
- MongoDB storage with duplicate prevention
- ClearML integration for tracking
- Support for multiple ROS2-related topics

Search Categories
- ROS2 tutorials
- Navigation2 guides
- MoveIt2 tutorials
- Gazebo integration
- ROS2 Humble tutorials

Youtube Transcripts data:

<img src="./media/etl/mongo_2.png" height="300px">

[Web Crawler](./web_crawler.py)


A Python web crawler designed to extract documentation and code examples from ROS2-related websites. The crawler systematically visits web pages, extracts text content and code snippets, while respecting domain boundaries and page limits.

Features
- Domain-restricted crawling
- Text and code snippet extraction
- Automatic link discovery
- Duplicate URL prevention
- Configurable page limit
- HTML cleaning and formatting

Content Extraction
- Main text content
- Code examples from `<code>` and `<pre>` tags
- Internal links for navigation
- Stripped of scripts and styling

### Featurization Pipelines Milestone

Here's the code python script for featurization pipeline: [featurizer.py](./featurizer.py)

The featurization pipeline processes data from multiple sources (GitHub repositories, web documentation, and YouTube transcripts) and transforms them into vector embeddings for use in our RAG system. The pipeline follows three main steps as shown in the diagram: Clean, Chunk, and Embed.

#### Pipeline Components

##### 1. Data Cleaning
- Removes URLs and special characters
- Normalizes whitespace
- Handles text validation and empty content
- Preserves essential punctuation (periods, hyphens)

##### 2. Text Chunking
- Splits text into semantic chunks using sentence tokenization
- Maintains context with a maximum chunk size of 512 tokens
- Preserves sentence boundaries for better coherence
- Handles overflow with intelligent chunk management

##### 3. Vector Embedding
- Uses the `all-MiniLM-L6-v2` model for creating embeddings
- Processes chunks in batches for efficiency
- Adds rich metadata to each vector:
  - Source type (github/web/youtube)
  - Original URL
  - Content type
  - Full text chunk

##### 4. Storage
- Initializes and manages Qdrant vector database
- Implements batch processing (100 vectors per batch)
- Uses COSINE similarity for vector comparisons
- Generates unique IDs for each vector entry

#### Data Sources Integration
The pipeline processes three main data sources:
- **GitHub Repositories**: Code documentation and README files
- **Web Documentation**: ROS2 official documentation and tutorials
- **YouTube Transcripts**: Tutorial video transcripts

#### Monitoring
- Progress tracking through **ClearML**
- Batch upload monitoring
- Processing status for each data source
- Vector creation metrics

The pipeline ensures all processed data is properly cleaned, chunked, and embedded before being stored in the vector database for efficient retrieval during the RAG process.

Vector data in QdrantDB:

<img src="./media/featurization/qdrant.png" height="400px">

Sample Point

<img src="./media/featurization/qdrant_2.png" height="400px">

Data visualization

<img src="./media/featurization/qdrant_3.png" height="400px">

### Finetuning Milestone


### Deploying the App Milestone

