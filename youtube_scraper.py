from youtube_transcript_api import YouTubeTranscriptApi
from pymongo import MongoClient
from urllib.parse import quote_plus
import logging
import os
from dotenv import load_dotenv
load_dotenv()


# MongoDB connection setup
client = MongoClient(os.getenv('MONGO_URI'))
db = client['ros2_rag']
collection = db['youtube_transcripts']

# List of ROS2 tutorial video IDs
ros2_videos = [
    '7TVWlADXwRw',  # ROS2 Tutorial
    'Gg25GfA456o',  # ROS2 Humble Crash Course
    'fsoi6faumrw',  # ROS2 Installation
    'idQb2pB-h2Q'   # ROS2 navigation
    'IrJmuow1r7g'   # ROS2 navigating to a specific pose
]


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


def main():
    successful_scrapes = 0

    for video_id in ros2_videos:
        try:
            transcript_doc = get_transcript(video_id)
            if transcript_doc:
                collection.insert_one(transcript_doc)
                successful_scrapes += 1
                print(f"Successfully scraped transcript for video {video_id}")
        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")

    print(f"Successfully scraped {successful_scrapes} out of {len(ros2_videos)} videos")
    client.close()


if __name__ == "__main__":
    main()
