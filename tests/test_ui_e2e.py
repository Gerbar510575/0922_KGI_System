# 需要 playwright，等 UI 完整再用
# pip install playwright pytest-playwright
# playwright install

from playwright.sync_api import sync_playwright

def test_ui_question():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")
        page.fill("textarea", "ETF是什麼？")
        page.click("button:has-text('送出')")
        page.wait_for_timeout(5000)
        assert "ETF" in page.inner_text("div.stMarkdown")
        browser.close()
