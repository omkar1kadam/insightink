import re
import logging
import praw
from emoji import demojize

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reddit API Client using praw
reddit = praw.Reddit(
    client_id="DVdj3FMkl9acqHcQ_2JBVQ",
    client_secret="FnooP_APKltJ1YI07w5jiALlcgK1pw",
    user_agent="my_reddit_bot"
)

class RedditAPI:
    def __init__(self):
        pass  # No need for session-based requests when using praw

    def fetch_post_details(self, post_id: str, max_comments: int = 10) -> dict:
        """Fetches post image and top comments for a given post ID using PRAW"""

        def shorten_comment(comment, limit=500):
            words = comment.split()
            shortened = []
            char_count = 0

            for word in words:
                if char_count + len(word) + 1 > limit:  # +1 accounts for spaces
                    break
                shortened.append(word)
                char_count += len(word) + 1  # Include space

            return " ".join(shortened) if char_count < len(comment) else comment
        
        try:
            submission = reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Load all comments
            
            # ✅ Improved Post Image Fetching
            post_image = None
            image_formats = (".png", ".jpg", ".jpeg", ".gif", ".webp")

            # 1️⃣ Direct URL Image
            if submission.url and submission.url.lower().endswith(image_formats):
                post_image = submission.url

            # 2️⃣ Check "preview" metadata
            elif hasattr(submission, "preview") and "images" in submission.preview:
                post_image = submission.preview["images"][0]["source"]["url"]

            # 3️⃣ Check "media_metadata" for gallery posts
            elif hasattr(submission, "media_metadata"):
                for media in submission.media_metadata.values():
                    if "p" in media:  # 'p' contains different resolutions
                        post_image = media["p"][-1]["u"]  # Highest resolution
                        break

            # 4️⃣ Check "thumbnail" (fallback)
            elif submission.thumbnail and submission.thumbnail.startswith("http"):
                post_image = submission.thumbnail

            # 🔹 Process Comments
            comments_data = []
            for comment in submission.comments.list()[:max_comments]:
                comment_body = demojize(comment.body)  # Convert emojis to plain text
                comments_data.append(comment_body)

            # Filter out comments containing image URLs
            image_regex = re.compile(r'\bhttps?://\S+\.(?:png|jpg|jpeg|gif|webp)\b', re.IGNORECASE)
            not_image_comments = [comment for comment in comments_data if not image_regex.search(comment)]
            shortened_comments = [shorten_comment(comment, 450) for comment in not_image_comments]

            return {
                "post_image": post_image,
                "comments": shortened_comments
            }

        except Exception as e:
            logger.error(f"Failed to fetch post details: {e}")
            return {"post_image": None, "comments": []}


if __name__ == "__main__":
     RedditAPI()
