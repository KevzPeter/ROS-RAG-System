import json
import requests
from typing import List, Dict
from tqdm import tqdm
import concurrent.futures
import logging
from time import sleep
import re

class InstructionDatasetCreator:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.api_base = "http://localhost:1234/v1/chat/completions"
        
        self.domains = {
            "ros2": {
                "description": "ROS2 middleware and core concepts",
                "subtopics": ["nodes", "topics", "services", "actions", "parameters", "lifecycle", "QoS", "launch"]
            },
            "nav2": {
                "description": "Navigation and path planning",
                "subtopics": ["costmaps", "planners", "controllers", "behaviors", "recovery", "transforms", "localization"]
            },
            "moveit2": {
                "description": "Motion planning and manipulation",
                "subtopics": ["planning", "kinematics", "collision", "trajectories", "controllers", "perception"]
            },
            "gazebo": {
                "description": "Robot simulation and testing",
                "subtopics": ["worlds", "models", "plugins", "physics", "sensors", "visualization"]
            }
        }
        
        self.topics = [
            "installation",
            "configuration",
            "implementation",
            "troubleshooting",
            "optimization",
            "integration",
            "best_practices",
            "error_handling"
        ]

    def clean_json_string(self, content: str) -> str:
        """Clean and validate JSON string."""
        try:
            # Remove any leading/trailing whitespace
            content = content.strip()
            
            # Ensure content starts and ends with curly braces
            if not (content.startswith('{') and content.endswith('}')):
                return ""
            
            # Remove any control characters
            content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
            
            # Validate JSON structure
            json.loads(content)
            return content
        except Exception as e:
            self.logger.debug(f"JSON cleaning error: {str(e)}")
            return ""

    def generate_instruction_pair(self, domain: str, description: str, topic: str, subtopic: str) -> Dict:
        prompt = f"""Generate a technical instruction-response pair about {domain} ({description}) 
        focusing on {topic} related to {subtopic}. Include implementation details and code examples.

        Return ONLY a valid JSON object in this EXACT format, with no additional text or formatting:
        {{"instruction": "Write a clear technical question here", "response": "Write a detailed technical response here"}}"""
        
        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a robotics expert. Generate ONLY valid JSON output with no formatting or additional text."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_base, json=payload)
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    cleaned_content = self.clean_json_string(content)
                    if cleaned_content:
                        parsed = json.loads(cleaned_content)
                        if "instruction" in parsed and "response" in parsed:
                            return {
                                "instruction": parsed["instruction"],
                                "response": parsed["response"]
                            }
                    sleep(1)  # Brief pause before retry
                else:
                    self.logger.error(f"API error {response.status_code}")
                    sleep(2)  # Longer pause on API error
            except Exception as e:
                self.logger.error(f"Request error (attempt {attempt+1}): {str(e)}")
                sleep(2)
        
        return {"instruction": "", "response": ""}

    def generate_dataset(self, pairs_per_combination: int = 3) -> List[Dict]:
        instruction_pairs = []
        total_combinations = sum(len(info["subtopics"]) * len(self.topics) for info in self.domains.values())
        
        with tqdm(total=total_combinations * pairs_per_combination) as pbar:
            for domain, info in self.domains.items():
                for topic in self.topics:
                    for subtopic in info["subtopics"]:
                        successful_pairs = 0
                        attempts = 0
                        max_attempts = pairs_per_combination * 2
                        
                        while successful_pairs < pairs_per_combination and attempts < max_attempts:
                            pair = self.generate_instruction_pair(domain, info["description"], topic, subtopic)
                            if pair["instruction"] and pair["response"]:
                                instruction_pairs.append(pair)
                                successful_pairs += 1
                                self.logger.info(f"Generated pair for {domain}-{topic}-{subtopic}")
                            attempts += 1
                            pbar.update(1)
                            sleep(1)  # Rate limiting
        
        self.logger.info(f"Generated {len(instruction_pairs)} valid pairs")
        return instruction_pairs

    def save_to_jsonl(self, pairs: List[Dict], output_file: str):
        try:
            valid_pairs = [p for p in pairs if p["instruction"] and p["response"]]
            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in valid_pairs:
                    f.write(json.dumps({
                        "instruction": pair["instruction"],
                        "response": pair["response"]
                    }, ensure_ascii=False) + '\n')
            self.logger.info(f"Successfully saved {len(valid_pairs)} pairs to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving to file: {str(e)}")

if __name__ == "__main__":
    creator = InstructionDatasetCreator()
    dataset = creator.generate_dataset(pairs_per_combination=3)
    creator.save_to_jsonl(dataset, "finetune_data1.jsonl")