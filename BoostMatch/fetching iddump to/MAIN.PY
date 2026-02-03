# main.py
from facebook_api import (
    extract_post_id,
    fetch_facebook_post,
    fetch_article_text
)
from matcher import check_misleading


print("✅ System ready")
print("Type 'exit' anytime to quit.\n")


while True:
    fb_link = input("🔗 Paste Facebook post link: ").strip()
    if fb_link.lower() == "exit":
        print("👋 Exiting program.")
        break

    try:
        # --- Fetching ---
        post_id = extract_post_id(fb_link)
        print("🔍 Extracted Post ID:", post_id)

        if not post_id:
            print("❌ Invalid Facebook post link.\n")
            continue

        caption, article_link = fetch_facebook_post(post_id)

        print("\n📄 Facebook Caption:\n", caption)
        print("\n🔗 Article Link:\n", article_link)

        if not article_link:
            print("⚠️ No article link found.\n")
            continue

        article_text = fetch_article_text(article_link)

        # --- Matching ---
        cos_sim, prediction = check_misleading(caption, article_text)

        print(f"\n🔍 Cosine Similarity: {cos_sim:.4f}")
        print(f"{'🚨' if prediction == 'MISLEADING' else '✅'} Prediction: {prediction}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")
