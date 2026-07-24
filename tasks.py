"""docker run -d -p 6379:6379 redis:alpine"""
import os
from celery import Celery
import httpx
from github import Github
from agent_core import app as agent_graph

celery_app = Celery(
    "code_review_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

celery_app.conf.worker_prefetch_multiplier = 1

@celery_app.task(
    name="tasks.analyze_pr_diff",
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def analyze_pr_diff(self, pr_data: dict):
    """Celery worker task that fetches the diff, processes via LangGraph, and replies to GitHub."""
    GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
    gh_client = Github(GITHUB_TOKEN)
    
    repo_full_name = pr_data["repository"]["full_name"]
    pr_number = pr_data["pull_request"]["number"]
    diff_url = pr_data["pull_request"]["diff_url"]
    
    print(f"[Worker] Starting evaluation on PR #{pr_number} from {repo_full_name}...")
    
    try:
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(diff_url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch diff from GitHub: {response.text}")
            code_diff = response.text

        graph_inputs = {"code_diff": code_diff}
        graph_outputs = agent_graph.invoke(graph_inputs)
        final_markdown_report = graph_outputs.get("final_report")

        if not final_markdown_report:
            raise ValueError("LangGraph completed execution but generated empty report.")

        repo = gh_client.get_repo(repo_full_name)
        pull_request = repo.get_pull(pr_number)
        pull_request.create_issue_comment(final_markdown_report)
        
        return f"Successfully processed PR #{pr_number}"

    except Exception as exc:
        print(f"[Worker Error] Encountered exception: {exc}. Retrying task...")
        raise self.retry(exc=exc)