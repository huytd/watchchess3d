# Watch Chess in 3D

Watch a chess video with a 3D board that follows the game. Play the video in another tab and share
that tab with the page: your browser reads the board about 3 times a second as it plays, on your
own device, and the 3D board keeps up. Nothing is downloaded or uploaded, so it works with YouTube
(or any site) without a server fetching videos. Arrows and circles the streamer draws show up on
the 3D board too. You can also open a video file.

There are two pages:

- `/`: share a tab (or open a file). Needs only static files, so it can be hosted anywhere
  (see [Deploy to Vercel](#deploy-to-vercel)).
  Screen sharing needs a desktop browser (Chrome and Edge share a single tab, Firefox and Safari a
  window); phones can only open files.
- `/server`: paste a YouTube link and the server downloads it, so the reading builds a timeline
  you can pause and seek through, and works on phones. Needs `app.py` running (see below).

<img width="1000" height="514" alt="image" src="https://github.com/user-attachments/assets/c7592d26-c61b-4a7f-a311-ae325a0e1928" />

## Run it

With Docker:

    docker build -t shadowboard .
    docker run -p 8080:8080 -v shadowboard-data:/data shadowboard

Without Docker (needs Python 3.10+ and ffmpeg):

    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8080

Then open http://localhost:8080 (or http://localhost:8080/server to paste YouTube links), or
http://<your computer's IP>:8080 from a phone on the same Wi-Fi. To use it away from home, put it behind HTTPS on a small VPS or a tunnel (Cloudflare
Tunnel, Tailscale Funnel, ngrok).

For `/` alone, any static file server will do, e.g. `python -m http.server -d static 8080`.
Screen sharing needs HTTPS (or localhost).

## Deploy to Vercel

`/` needs no server, so it can go on Vercel as plain static files:

    ./deploy.sh              # production
    ./deploy.sh --preview    # a preview at its own URL

It needs Node.js. The first run asks you to log in to Vercel and to pick or create a project, and
remembers the project in `.vercel/`. In CI, set `VERCEL_TOKEN`, `VERCEL_ORG_ID` and
`VERCEL_PROJECT_ID` instead. Importing the repository on vercel.com works too, with no settings to
change. `vercel.json` publishes the `static` folder as it is (no build) and sends `/server` to
`/`; `.vercelignore` leaves out `app.py`, the `/server` page and everything else only the server
uses.

## How it works

All the reading happens in the browser. On `/`, the shared tab is shown in the video card and
read as it plays (`readScreen` in `static/read.js`): the board is found, then read about 3 times a
second. Each reading is checked against the board it was found on, so when the board goes away
(an ad, another video, the page scrolled or resized) frames are skipped and the board is looked
for again. On `/server`, the server fetches YouTube videos with yt-dlp, repackages them as plain
MP4 (no re-encoding) and serves them to the browser. Video files you open never leave your device.

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
- `static/index.html` (`/`) and `static/server.html` (`/server`): the two pages, sharing
  `static/app.js` (the 3D board with three.js, the video card, settings) and `static/style.css`.
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

- **YouTube's terms of service don't allow downloading videos** (this is about `/server`; `/` downloads nothing) outside YouTube's own apps.
  Running this for yourself is a personal-use gray area; don't run it as a public service.
  Opening your own recordings avoids the issue.
- **YouTube often blocks downloads from cloud servers** ("Sign in to confirm you're not a bot").
  It usually works from a home connection. On a VPS, export your browser's YouTube cookies to
  a cookies.txt, set `YTDLP_COOKIES`, and keep yt-dlp up to date. Never commit that file.
- Only flat 2D digital boards are recognized (Lichess/chess.com style), not filmed physical boards.
- If the board isn't found, open Settings (the gear above the video) › **Draw the board area yourself**.
- On `/`, keep the page in front while you watch: browsers slow down background tabs, so the
  reading slows while you are in the video's tab (to pause or skip, say). Chrome and Edge keep this
  page in front when you share.
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
