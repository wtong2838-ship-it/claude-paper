import os
import requests
import json
API_KEY = "sk-880a7ff38b6e590eaaee528fa3581e0a1f2c7b310316f96a21438672044cd532"
API_URL = "https://w.ciykj.cn/v1/responses"
def generate_review():
    with open("REVIEW_INSTRUCTIONS.md", "r", encoding="utf-8") as f:
        instructions = f.read()

    payload = {
    "model": "gpt-5.4",
    "text": "Hello, are you working?"
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print("正在发送请求到 T1API ...")
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        report = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        with open("codex_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("审查完成！报告已保存至 codex_report.md")
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    generate_review()