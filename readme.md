# Video Retrieval with ActivityNet

## Usage
To use the Hybrid Video Retrieval application demo, follow these steps:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/chadyoungs/video_retrieval_with_activitynet.git
   cd video_retrieval_with_activitynet
   ```
2. **Install Dependencies:**
   Make sure you have Python 3.x installed. Then, install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application:**
   Start the application by running:
   ```bash
   python app.py
   ```
   Access the demo at `http://localhost:5000`.

## Principle
The hybrid retrieval approach combines both content-based and semantic-based retrieval methods to achieve improved accuracy and efficiency. The system utilizes visual features extracted from video frames along with semantic embeddings of actions and objects within the videos. This enables the application to retrieve videos that match user queries more effectively.

## Solution
The system architecture consists of three main components:
1. **Feature Extraction:** Visual features are extracted from videos using convolutional neural networks (CNNs). 
2. **Semantic Analysis:** The application employs Natural Language Processing (NLP) techniques to analyze user queries and map them to corresponding semantic embeddings.
3. **Retrieval Engine:** The retrieval engine matches user queries against the stored video features using a hybrid approach, returning the most relevant results based on both visual and semantic similarity.

This comprehensive solution allows users to perform efficient and accurate video retrieval based on a wide range of queries.