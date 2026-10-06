"""GUIDE Phần 1 - Định nghĩa subagent (tác tử con).   >>> SINH VIÊN CÀI ĐẶT <<<

Pseudo-code: guides/pseudocode/02_subagents.md
Kiểm tra:    pytest tests/test_02_agent.py
"""


def get_subagents() -> list[dict]:
    """Trả về danh sách subagent (ít nhất 2, tên khác nhau).

    Mỗi phần tử là một dict có các khóa bắt buộc:
      "name":          tên duy nhất (chữ thường, có thể có dấu gạch ngang)
      "description":   khi nào tác tử chính nên giao việc cho subagent này (viết như một hướng dẫn hành động)
      "system_prompt": chỉ dẫn cho subagent
    Gợi ý vai trò: explorer (đọc và báo cáo), implementer (thực hiện), reviewer (kiểm tra độc lập).
    """
    return [
        {
            "name": "explorer",
            "description": "Use when you need to inspect the task instructions, repository files, data, or logs before changing anything. Report facts and constraints without editing files.",
            "system_prompt": "You are an exploration subagent. Read the task instructions and relevant files, identify facts and constraints, and return a concise evidence-based report. Do not modify files.",
        },
        {
            "name": "implementer",
            "description": "Use when the task requires making files changes, running focused checks, and reporting the resulting behavior.",
            "system_prompt": "You are an implementation subagent. Make the required code or data changes in the sandbox, run focused validation, and return a short report of what changed and what passed or failed.",
        },
        {
            "name": "reviewer",
            "description": "Use when you need an independent check of a completed result against the task rules, edge cases, and verification commands.",
            "system_prompt": "You are a review subagent. Inspect the result independently, compare it with the task rules, test edge cases, and return only verifiable findings.",
        },
    ]
