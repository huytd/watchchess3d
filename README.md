# Watch Chess in 3D (server version)

Paste a YouTube link (or open a video file) and watch it with a 3D board that follows the game.
Works on phones: no screen sharing. The server only downloads the video; your browser reads the
board about 3 times a second and builds a timeline of positions, so pausing and seeking just work.
Arrows and circles the streamer draws show up on the 3D board too.

<img width="1000" height="514" alt="image" src="https://github.com/user-attachments/assets/c7592d26-c61b-4a7f-a311-ae325a0e1928" />

## Run it

With Docker:

    docker build -t shadowboard .
    docker run -p 8080:8080 -v shadowboard-data:/data shadowboard

Without Docker (needs Python 3.10+ and ffmpeg):

    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8080

Then open http://localhost:8080, or http://<your computer's IP>:8080 from a phone on the same
Wi-Fi. To use it away from home, put it behind HTTPS on a small VPS or a tunnel (Cloudflare
Tunnel, Tailscale Funnel, ngrok).

## How it works

The server does no video work: it fetches YouTube videos with yt-dlp, repackages them as plain
MP4 (no re-encoding) and serves them to the browser, which does all the reading. Video files you
open never leave your device.

- `app.py`: FastAPI. Runs downloads in the background and keeps finished readings, so the next
  person to open the same video gets it instantly. Also serves the episode lists on the start card.
- `static/read.js`, `static/mp4.js`: get frames. MP4s are decoded in a Web Worker with WebCodecs;
  anything else falls back to stepping a hidden `<video>`.
- `static/pipeline.js`, `static/engine.js`: find the board (the strongest 8x8 checkerboard
  pattern in the frame), work out which way up it is, read each square with a small CNN
  (`static/cnn.js`, weights in `static/model.json`, run with TensorFlow.js), and turn the readings
  into moves, preferring legal ones (`static/rules.js`) so one shaky square can't derail the game.
  Streamers' arrows, circles and highlighted squares are found with plain image processing.
- Which way up: chess.com and Lichess put the player's colour at the bottom, so this is settled
  per game from the starting position, and the reader also notices when the board is flipped
  mid-video. The 3D board turns to match.
- `static/index.html`: the page, with the 3D board (three.js) and YouTube's embedded player.
  Board, piece and table finishes, camera views and auto-rotate are under Settings and the camera bar.
- `tools/make_pieces.py`, `tools/make_props.py`: build `static/pieces.glb` and `static/props.glb`
  from the Poly Haven models listed under Credits.

Speed depends on the device: without a GPU (WebAssembly on 4 CPU cores), 20 minutes of video
takes about 2.5 minutes, and you can watch from the start while it works. The server stays under
100 MB of memory and uses almost no CPU.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SHADOWBOARD_DATA` | `./data` | where downloaded videos and saved readings are kept |
| `SHADOWBOARD_MAX_SECONDS` | 10800 | longest video accepted |
| `SHADOWBOARD_WORKERS` | 2 | videos downloaded at the same time |
| `YTDLP_COOKIES` | (none) | path to a cookies.txt, if YouTube blocks the server |

The start card's series are set in `SERIES` in `app.py`; `static/series.json` is the fallback
list (refresh it with `python app.py series`).

## Things to know

- **YouTube's terms of service don't allow downloading videos** outside YouTube's own apps.
  Running this for yourself is a personal-use gray area; don't run it as a public service.
  Opening your own recordings avoids the issue.
- **YouTube often blocks downloads from cloud servers** ("Sign in to confirm you're not a bot").
  It usually works from a home connection. On a VPS, export your browser's YouTube cookies to
  a cookies.txt, set `YTDLP_COOKIES`, and keep yt-dlp up to date. Never commit that file.
- Only flat 2D digital boards are recognized (Lichess/chess.com style), not filmed physical boards.
- If the board isn't found, open Settings (the gear above the video) › **Draw the board area yourself**.
- Faint arrows can be missed; turn arrows off under Settings › Show arrows from the video.
- The page loads TensorFlow.js and three.js from CDNs (jsDelivr, cdnjs).
- Downloaded videos stay in the data folder until you delete them.

## Credits

3D models and textures from [Poly Haven](https://polyhaven.com), all CC0: the
[Chess Set](https://polyhaven.com/a/chess_set) by Riley Queen,
[Cherry Veneer](https://polyhaven.com/a/cherry_veneer), Wood Table 001, Wood Table, Brown Leather,
Velour Velvet, Rough Linen, Tea Set 01, Stationery Supplies, Round Spectacles, Digital Wrist Watch,
Potted Plant 04, Croissant, Food Apple 01, Magnifying Glass 01 and Vintage Pocket Watch.

## License

[BSD 3-Clause](LICENSE). The Poly Haven assets above are CC0.
