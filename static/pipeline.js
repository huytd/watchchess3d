/* Reading a whole video: find the board, work out which way up it is, then read it about 3
   times a second. The same steps for either frame source: WebCodecs in the worker
   (mp4.js) or a <video> element on the page (read.js).

   source: { width, height, duration, frameAt(t, area) -> RGBA pixels {w, h, data},
             scan(area, fps, onFrame(t, pixels) -> Promise, live) }
   ops:    { search(img), refine(img, box), orient(crops), start(whiteBottom, decided, fixed), frame(t, img) }
   report(update) gets the status, the board area and the positions as they are found;
   live() throws when the reading has been called off. */
import { FPS, pickBoard } from './engine.js';

export async function readSource(source, ops, { region = null, orientation = 'auto', report, live }){
  const W = source.width, H = source.height, duration = source.duration;
  if (!region){
    // look at frames spread through the video; the most consistent strong board box wins
    report({ status: 'finding_board' });
    const found = [];
    for (let i = 0; i < 12; i++){
      const t = duration * (i + 0.5) / 12;
      const img = await source.frameAt(t); live();
      const r = await ops.search(img); live();
      if (r.score > 12 && r.box[2] > 0) found.push({ ...r, t });
    }
    const win = pickBoard(found);
    if (!win) throw new Error('no_board');
    region = await ops.refine(await source.frameAt(win.t), win.box); live();
  }
  report({ status: 'processing', region, width: W, height: H, duration, progress: 0 });
  const x = Math.max(0, Math.round(region[0])), y = Math.max(0, Math.round(region[1]));
  const crop = [x, y, Math.max(8, Math.min(W - x, Math.round(region[2]))), Math.max(8, Math.min(H - y, Math.round(region[3])))];
  let whiteBottom = orientation !== 'b', decided = orientation !== 'auto';
  if (!decided){
    // which way up the board is, by majority over frames spread through the whole video
    const crops = [];
    for (let i = 0; i < 16; i++){ crops.push(await source.frameAt(duration * (i + 0.5) / 16, crop)); live(); }
    const r = await ops.orient(crops); live();
    if (r !== null){ whiteBottom = r; decided = true; }
  }
  await ops.start(whiteBottom, decided, orientation !== 'auto'); live();
  await source.scan(crop, FPS, (t, img) => ops.frame(t, img).then(r => report({
    events: r.events, marks: r.marks, whiteBottom: r.decided ? r.whiteBottom : null,
    progress: Math.min(1, (t + 1 / FPS) / Math.max(duration, 1e-6)),
  })), live);
  live();
  report({ status: 'done', progress: 1 });
}
