from youtube_transcript_api import YouTubeTranscriptApi
from pymongo import MongoClient
from urllib.parse import quote_plus
import logging
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

load_dotenv()


def get_video_ids_from_search(api_key, search_queries):
    """Search YouTube and return video IDs"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = set()

    for query in search_queries:
        try:
            request = youtube.search().list(
                part='id',
                q=query,
                type='video',
                maxResults=50,  # Adjust as needed
                relevanceLanguage='en'
            )
            response = request.execute()

            for item in response['items']:
                if item['id']['kind'] == 'youtube#video':
                    video_ids.add(item['id']['videoId'])

        except Exception as e:
            logging.error(f"Error searching for query '{query}': {str(e)}")

    return list(video_ids)


def extract_video_id(url):
    """Extract video ID from a YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    return None


def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Process transcript to get clean text
        full_text = ""
        for entry in transcript:
            text = entry['text'].strip()
            if text:
                full_text += f"{text} "

        # Create document for MongoDB
        document = {
            'video_id': video_id,
            'transcript': full_text,
            'timestamps': transcript
        }

        return document

    except Exception as e:
        logging.error(f"Error getting transcript for video {video_id}: {str(e)}")
        return None


def run_youtube_scraper():
    # YouTube API key from environment variables
    api_key = os.getenv('YOUTUBE_API_KEY')
    client = MongoClient(os.getenv('MONGO_URI'))
    db = client['ros2_rag']
    collection = db['youtube_transcripts']

    # Define search queries
    search_queries = [
        'ROS2 tutorial',
        'ROS2 navigation tutorial',
        'MoveIt2 tutorial',
        'Gazebo ROS2 tutorial',
        'Nav2 tutorial',
        'ROS2 humble tutorial'
    ]

    # Get video IDs from YouTube search
    video_ids = get_video_ids_from_search(api_key, search_queries)

    successful_scrapes = 0
    for video_id in video_ids:
        try:
            transcript_doc = get_transcript(video_id)
            if transcript_doc:
                # Check if transcript already exists
                if not collection.find_one({'video_id': video_id}):
                    collection.insert_one(transcript_doc)
                    successful_scrapes += 1
                    print(f"Successfully scraped transcript for video {video_id}")
        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")

    print(f"Successfully scraped {successful_scrapes} out of {len(video_ids)} videos")
    client.close()


if __name__ == "__main__":
    run_youtube_scraper()
