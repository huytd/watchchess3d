# Watch Chess in 3D (server version)

Paste a YouTube link (or open a video file) and watch it with a 3D board that follows the game.
Works on phones: no screen sharing. The server only downloads the video; your browser finds
the board, reads it about 3 times a second and builds a timeline of positions, and the page
shows whichever position matches the current playback time, so pausing and seeking just work.
Arrows and circles the streamer draws on the board show up on the 3D board too.

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

The server does no video work: it fetches YouTube videos with yt-dlp, copies them into a plain
MP4 (no re-encoding), and serves them to the browser, which does all the reading. Video files
you open never leave your device.

- `app.py`: FastAPI. Downloads run in the background while the page shows progress. A finished
  reading is sent back and kept, so the next person to open the same video gets it instantly
  (only readings with the board found automatically and the orientation detected).
- `static/read.js` picks how to get frames. An MP4 (every download, most files) is decoded in a
  Web Worker with WebCodecs (`static/mp4.js` reads the file's index; each frame is decoded once
  and it keeps going in a background tab). Anything else, or a browser without WebCodecs, falls
  back to stepping a hidden `<video>` through the video.
- `static/pipeline.js`: the steps of a reading, the same for both: find the board, work out which
  way up it is, then read the video frame by frame.
- `static/engine.js` (in the worker): finds the board automatically by searching every size and
  position for the strongest 8x8 checkerboard pattern, then fine-tunes the edges. Reads each frame
  with the CNN (only the squares that changed since they were last read), smooths over time, and
  turns readings into positions, preferring legal moves (`static/rules.js`, which follows
  python-chess) so one shaky square can't derail the game. Each position is timestamped to when
  it first appeared. It also finds the arrows, circles and red/blue squares streamers draw
  (Lichess and chess.com style) with plain image processing: arrows always join two square
  centres, so it tests every possible arrow (plus chess.com's L-shaped knight arrows) against a
  mask of strongly coloured pixels. A mark must hold for two readings to show and be gone for two
  to disappear. This is a line-by-line port of the Python pipeline the server used to run, and
  gives the same moves and marks on the same frames, apart from deciding which way up the board
  is (below), which the Python version did once for the whole video.
- Which way up the board is: chess.com and Lichess put the player's colour at the bottom, so a
  video with several games can show some from White's side and some from Black's. The reader
  settles it per game from the starting position (two readings in a row with 16 pieces of one
  colour on the top two rows and 16 of the other on the bottom two), and reads the board that way
  up until a game starts the other way round. Before the first game start (a video that joins a
  game in progress), it goes by the pawns, which can't pass each other on a file, and by where
  the pieces stand. The board can also be flipped at any point (one click on chess.com or Lichess,
  sometimes animated in the video): when every piece moves at once and the screen shows one of the
  last few positions turned round, for two readings in a row, the reader turns too, and drops
  anything it took from the upside-down board. Every position records which way up it was read,
  and the 3D board turns to match as playback reaches it. Readings saved before this (no
  `version: 4`) are read again.
- `static/cnn.js`: the square classifier (`static/model.json`), run with TensorFlow.js on the
  GPU (WebGL) or with WebAssembly, whichever is faster on the device.
- `static/index.html`: the page (3D board + YouTube's embedded player).
- `static/pieces.glb`: the chess pieces, from Poly Haven's [Chess Set](https://polyhaven.com/a/chess_set)
  by Riley Queen (CC0), cut down to one mesh per piece type and 512px textures (690 KB) by
  `tools/make_pieces.py`. Simple built-in pieces show until it has loaded. Both sides share one
  light colour map that the page tints, so the colours of the pieces and the board can be changed
  under Settings › Board colours / Piece colours (presets or any colour; remembered in the browser).
  Ticking Wood turns the chosen colours into wood of that tone: the board lays a neutral grain
  (`static/wood.jpg`, from Poly Haven's [Cherry Veneer](https://polyhaven.com/a/cherry_veneer), CC0)
  over each square, and the pieces get turned-wood grain drawn in the shader.
- `static/table/`: what the board stands on (Settings › Table: walnut by default, oak planks, leather,
  green baize, linen, or none), from Poly Haven's CC0 Wood Table 001, Wood Table, Brown Leather,
  Velour Velvet and Rough Linen. The table fades into a dim room with fog, under a soft vignette.
- On the table, laid out afresh each time the page opens: each player's scoresheet (the moves of the
  game on show are written in as it goes) with a pen and a cup of coffee, a chess clock whose time
  runs for the side to move while the video plays, and a few odds and ends picked at random. The
  clock and scoresheets are drawn in the page; the rest are Poly Haven models (CC0): Tea Set 01,
  Stationery Supplies, Round Spectacles, Digital Wrist Watch, Potted Plant 04, Croissant, Food
  Apple 01, Magnifying Glass 01 and Vintage Pocket Watch, packed into `static/props.glb` (880 KB,
  loaded after the pieces) by `tools/make_props.py`.
- The start card offers two series to pick an episode from: GothamChess's Slowrun (from
  [a playlist a viewer keeps](https://www.youtube.com/playlist?list=PLxH8DZpY_FC4vPjn-dz0-VmcZwmiuKWRa), as
  GothamChess has none of his own for it) and Hikaru's
  [Master Slowkaru](https://www.youtube.com/playlist?list=PL4KCWZ5Ti2H4hFLv7HBwVzYflOrUZ3qum). The server
  fetches the playlists with yt-dlp at most every 12 hours (`GET /api/series`, kept in the data folder);
  until it has, or if YouTube refuses, it uses `static/series.json` (refresh it with `python app.py series`).
  Episodes someone has already read are marked Ready, the ones opened in this browser are marked
  watched, and the list starts at the next one. Change the series in `SERIES` in `app.py`.
- The camera bar at the top right: auto-rotate (a very slow swing, about 25° left and right, remembered in the
  browser), views (overview, top down, player's eye, from the side, the whole table) and a button
  to look from the other side.

Speed depends on the device. Without a GPU (WebAssembly on 4 CPU cores) a 71-second test video
took about 15 seconds, so roughly 2.5 minutes for 20 minutes of video; you can watch from the
start while it works. The server itself stays under 100 MB of memory and uses almost no CPU.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SHADOWBOARD_DATA` | `./data` | where downloaded videos and saved readings are kept |
| `SHADOWBOARD_MAX_SECONDS` | 10800 | longest video accepted |
| `SHADOWBOARD_WORKERS` | 2 | videos downloaded at the same time |
| `YTDLP_COOKIES` | (none) | path to a cookies.txt, if YouTube blocks the server |

## Things to know

- **YouTube's terms of service don't allow downloading videos** outside YouTube's own apps.
  Running this for yourself is a personal-use gray area; don't run it as a public service.
  Opening your own recordings avoids the issue.
- The page loads TensorFlow.js and three.js from CDNs (jsDelivr, cdnjs) the first time, so the browser needs to reach them.
- A video is downloaded to the server and then once more by each browser that reads it (480p,
  video only: a few MB per minute).
- **YouTube often blocks downloads from cloud servers** ("Sign in to confirm you're not a bot").
  It usually works from a home connection. On a VPS, export your browser's YouTube cookies to
  a cookies.txt and set `YTDLP_COOKIES`, and keep yt-dlp up to date.
- If the board isn't found, or the wrong area is used, open Settings (the gear above the video) › **Draw the board area yourself**.
- Only flat 2D digital boards are recognized (Lichess/chess.com style), not filmed physical boards.
- Arrows come in four colours (green, red, blue, orange). Faint ones (a pale arrow on a board of
  a similar colour, thin Lichess circles in a low-quality video) can be missed. Turn them off
  under Settings › Show arrows from the video.
- Downloaded videos stay in the data folder until you delete them.
