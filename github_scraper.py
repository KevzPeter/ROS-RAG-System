import requests
from clearml import Task
import logging
from pymongo import MongoClient
from bson.binary import Binary
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize ClearML task
task = Task.init(project_name='GitHub Scraping', task_name='Repository Content')

# MongoDB configuration

DB_NAME = "ros2_rag"
COLLECTION_NAME = "github_repo"

# Initialize MongoDB client
client = MongoClient(os.getenv('MONGO_URI'))
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def get_repo_content(owner, repo, path='', token=None):
    """Fetch repository content using GitHub API"""
    headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {token}',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        if isinstance(response.json(), list):
            # Directory content
            contents = []
            for item in response.json():
                if item['type'] == 'file':
                    file_content = get_file_content(item['download_url'])
                    contents.append({
                        'path': item['path'],
                        'content': file_content,
                        'type': item['type']
                    })
                elif item['type'] == 'dir':
                    # Recursively get directory contents
                    sub_contents = get_repo_content(owner, repo, item['path'], token)
                    contents.extend(sub_contents)
            return contents
        else:
            # Single file content
            file_content = get_file_content(response.json()['download_url'])
            return [{
                'path': response.json()['path'],
                'content': file_content,
                'type': 'file'
            }]

    except Exception as e:
        task.logger.report_text(f"Error accessing {url}: {str(e)}", level=logging.ERROR)
        return []


def get_file_content(download_url):
    """Fetch raw file content"""
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def store_in_mongodb(repo_info, contents):
    """Store repository contents in MongoDB, skipping files larger than 16MB"""
    MAX_DOCUMENT_SIZE = 16000000  # 16MB in bytes

    for item in contents:
        try:
            document = {
                'owner': repo_info['owner'],
                'repo': repo_info['repo'],
                'path': item['path'],
                'type': item['type'],
                'content': item['content']
            }

            # Check document size
            if item['content'] and len(item['content'].encode()) > MAX_DOCUMENT_SIZE:
                task.logger.report_text(
                    f"Skipping {item['path']}: File size exceeds 16MB limit",
                    level=logging.WARNING
                )
                continue

            collection.update_one(
                {
                    'owner': repo_info['owner'],
                    'repo': repo_info['repo'],
                    'path': item['path']
                },
                {'$set': document},
                upsert=True
            )

            task.logger.report_text(f"Stored in MongoDB: {item['path']}", level=logging.INFO)

        except Exception as e:
            task.logger.report_text(f"Error storing {item['path']}: {str(e)}", level=logging.ERROR)


def main():
    # GitHub configuration
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

    # List of repositories to scrape
    repositories = [
        {'owner': 'ros2', 'repo': 'ros2'},
        {'owner': 'ros-planning', 'repo': 'navigation2'},
        {'owner': 'ros-planning', 'repo': 'moveit2'},
        {'owner': 'gazebosim', 'repo': 'gz-sim'}
    ]

    # Get repository contents for each repo
    for repo_info in repositories:
        task.logger.report_text(
            f"Processing repository: {repo_info['owner']}/{repo_info['repo']}",
            level=logging.INFO
        )

        contents = get_repo_content(
            owner=repo_info['owner'],
            repo=repo_info['repo'],
            token=GITHUB_TOKEN
        )

        # Store contents in MongoDB
        store_in_mongodb(repo_info, contents)

        # Process and log results
        for item in contents:
            if item['type'] == 'file':
                task.logger.report_text(
                    f"Scraped file: {item['path']}",
                    level=logging.INFO
                )

                # Store content metrics
                if item['content']:
                    task.logger.report_scalar(
                        "File Sizes",
                        f"{repo_info['repo']}/{item['path']}",
                        value=len(item['content']),
                        iteration=0
                    )


if __name__ == "__main__":
    main()

# check
