import os
import sys

# Fix Windows console unicode printing
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

from pathlib import Path
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

from agent import HybridMemoryAgent

def main():
    print("Initializing Hybrid Memory Agent...")
    agent = HybridMemoryAgent()
    
    # Simulate a user having some previous conversations/notes
    print("\n[Simulating Episodic Memory Ingestion]")
    memories = [
        "Tuần trước mình có đọc tài liệu về Kubernetes, nó giúp tự động quản lý container rất tốt.",
        "Dạo này mình thấy bảo mật Cloud (Cloud Security) đang có nhiều lỗ hổng, đặc biệt là IAM config sai.",
        "Mình cần setup hệ thống co giãn linh hoạt (auto-scaling) cho hạ tầng của công ty để chịu tải cao.",
        "AWS S3 là dịch vụ lưu trữ object rẻ nhưng dễ bị public nhầm nếu không cẩn thận.",
    ]
    
    # Hardcode user_id to match one from Feast offline data generator (e.g. u_001)
    test_user = "u_001"
    
    for mem in memories:
        agent.remember(mem, user_id=test_user)
        
    print("\n" + "="*60)
    print("RUNNING 5 SCENARIO QUERIES")
    print("="*60)
    
    queries = [
        ("Query 1 (Vector Hit)", "Tôi đã đọc gì về Kubernetes?"),
        ("Query 2 (Cần Profile Context)", "Recommend đọc gì tiếp?"),
        ("Query 3 (Cần Fresh Activity)", "Tôi đang quan tâm gì gần đây?"),
        ("Query 4 (Paraphrase - Vector wins)", "Tài liệu về tự động mở rộng hạ tầng?"),
        ("Query 5 (Mixed - Hybrid + Profile)", "Cho tôi summary cloud security")
    ]
    
    for title, q in queries:
        print(f"\n>>> {title}\nUser hỏi: '{q}'")
        context = agent.recall(query=q, user_id=test_user)
        print(context)

if __name__ == "__main__":
    main()
