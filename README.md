# Shadowboard (server version)

Paste a YouTube link (or upload a video) and watch it with a 3D board that follows the game.
Works on phones: no screen sharing. The server downloads the video, finds the board,
reads it about 3 times a second, and sends back a timeline of positions; the page shows
whichever position matches the current playback time, so pausing and seeking just work.

## Run it

With Docker:

    docker build -t shadowboard .
    docker run -p 8080:8080 -v shadowboard-data:/data shadowboard

Then open http://localhost:8080 (or your server's address) on any device.

Without Docker (needs Python 3.10+ and ffmpeg):

    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8080

For phones on the same Wi-Fi, open http://<your computer's IP>:8080. To use it away from
home, put it behind HTTPS on a small VPS or a tunnel (Cloudflare Tunnel, Tailscale Funnel, ngrok).

## How it works

- `pipeline.py`: finds the board automatically by searching every size and position for
  the strongest 8x8 checkerboard pattern, then fine-tunes the edges. Reads each frame with
  the CNN, smooths over time, and turns readings into positions, preferring legal moves
  (python-chess) so one shaky square can't derail the game. Each position is timestamped
  to when it first appeared.
- `recognizer.py`: the square classifier (same weights as the browser version), in numpy.
- `app.py`: FastAPI. Jobs run in the background; the page polls and can start playing
  while the rest of the video is still being read. Results are cached per video.
- `static/index.html`: the page (3D board + YouTube's embedded player).

Speed: about 80 ms per sampled frame on one CPU core, i.e. a 20-minute video takes roughly
5 minutes on a single core and much less on a multi-core machine. You can watch from the start
while it works.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SHADOWBOARD_DATA` | `./data` | where downloaded videos and results are kept |
| `SHADOWBOARD_MAX_SECONDS` | 10800 | longest video accepted |
| `SHADOWBOARD_MAX_UPLOAD_MB` | 1024 | largest upload accepted |
| `SHADOWBOARD_WORKERS` | 1 | videos processed at the same time |
| `YTDLP_COOKIES` | (none) | path to a cookies.txt, if YouTube blocks the server |

## Things to know

- **YouTube's terms of service don't allow downloading videos** outside YouTube's own apps.
  Running this for yourself is a personal-use gray area; don't run it as a public service.
  Uploading your own recordings avoids the issue.
- **YouTube often blocks downloads from cloud servers** ("Sign in to confirm you're not a bot").
  It usually works from a home connection. On a VPS, export your browser's YouTube cookies to
  a cookies.txt and set `YTDLP_COOKIES`, and keep yt-dlp up to date.
- If the board isn't found, or the wrong area is used, press **Fix board area** and draw it.
- Only flat 2D digital boards are recognized (Lichess/chess.com style), not filmed physical boards.
- Downloaded videos stay in the data folder until you delete them.
