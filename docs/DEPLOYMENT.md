# Deploying a free public demo

The app runs on [Streamlit Community Cloud](https://streamlit.io/cloud), which is free for public GitHub repositories. Pair it with a free-tier chat API such as Google Gemini or Groq and the whole demo costs nothing.

## 1. Get a free API key

Pick one provider:

- **Google Gemini**: create a key in [Google AI Studio](https://aistudio.google.com/apikey).
- **Groq**: create a key in the [Groq console](https://console.groq.com/keys).

## 2. Create the app

1. Sign in to [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Choose this repository, the `main` branch and `app.py` as the main file.
4. Open **Advanced settings**:
   - Python version: 3.11 or newer.
   - Secrets: paste the contents of [`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example) and replace the placeholder key with your real key.
5. Click **Deploy**. The first build installs the dependencies, and the first time notes are processed the embedding model (about 90 MB) is downloaded, so the first run takes a few minutes.

Secrets are copied into environment variables when the app starts, so every setting from [`.env.example`](../.env.example) can also be set there.

## 3. What demo mode does

`DEMO_MODE = "true"` is set in the example secrets because a public demo is shared by everyone who opens the link:

- Each browser session gets its own ChromaDB collection, so visitors never see or delete each other's notes.
- Each session can ask up to `DEMO_MAX_QUESTIONS` questions, which keeps one visitor from using up the free API quota.
- The example secrets also lower `MAX_UPLOADED_FILE_MB` to 10 MB.

Visitors who have no PDFs can click **Try with sample notes** to load the bundled lecture notes in [`sample_data/`](../sample_data/) and try the example questions.

## 4. Things to know

- Community Cloud apps go to sleep after a period without visitors. The next visitor wakes the app up, which takes about a minute.
- Storage on Community Cloud is temporary. Uploaded notes disappear whenever the app restarts, which suits a demo.
- Retrieved note text is sent to the chat API provider you configured. Some free tiers may use prompts to improve their models, so the demo is not the place for private material.

## 5. Add the link to the README

When the app is live, copy its URL and add it near the top of the README, for example:

```markdown
**Live demo:** https://your-app-name.streamlit.app
```
