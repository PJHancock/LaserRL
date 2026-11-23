"""Vector optics board and portable HTML policy playback (no network assets)."""
from __future__ import annotations
import html
import json
import math
from pathlib import Path


def render_svg(puzzle, frame):
    def point(p):
        return f"{p[0]*800:.3f},{p[1]*800:.3f}"
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="-22 -22 844 844" role="img" aria-label="Laser optics puzzle">',
        '<defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#24334b" stroke-width="1"/></pattern><filter id="glow"><feGaussianBlur stdDeviation="4"/></filter></defs>',
        '<rect x="-22" y="-22" width="844" height="844" rx="24" fill="#0c1424"/>',
        '<rect width="800" height="800" fill="url(#grid)" stroke="#425474" stroke-width="2"/>']
    for x,y,w,h in puzzle.obstacles:
        elements.append(f'<rect x="{x*800}" y="{y*800}" width="{w*800}" height="{h*800}" rx="5" fill="#493746" stroke="#dc7c8e" stroke-width="2"/>')
    color = '#70f1b8' if frame['solved'] else '#e9b25d'
    tx,ty = puzzle.target
    elements.append(f'<circle cx="{tx*800}" cy="{ty*800}" r="{puzzle.target_radius*800}" fill="{color}" fill-opacity=".15" stroke="{color}" stroke-width="3"/>')
    elements.append(f'<circle cx="{tx*800}" cy="{ty*800}" r="5" fill="{color}"/>')
    for a,b in frame['trace']['segments']:
        elements.append(f'<path d="M {point(a)} L {point(b)}" fill="none" stroke="#ffce6d" stroke-width="9" opacity=".5" filter="url(#glow)"/>')
        elements.append(f'<path d="M {point(a)} L {point(b)}" fill="none" stroke="#ffe7a7" stroke-width="2.5"/>')
    for i,(x,y,t) in enumerate(frame['poses']):
        dx,dy = puzzle.mirror_length/2*math.cos(t),puzzle.mirror_length/2*math.sin(t)
        elements.append(f'<path d="M {point((x-dx,y-dy))} L {point((x+dx,y+dy))}" stroke="#1e445f" stroke-width="13" stroke-linecap="round"/>')
        elements.append(f'<path d="M {point((x-dx,y-dy))} L {point((x+dx,y+dy))}" stroke="#83d6fc" stroke-width="5" stroke-linecap="round"/>')
        elements.append(f'<circle cx="{x*800}" cy="{y*800}" r="7" fill="#e2f6ff"/><text x="{x*800+12}" y="{y*800-12}" fill="#c0e9ff" font-family="sans-serif" font-size="18">M{i+1}</text>')
    sx,sy = puzzle.source
    elements.append(f'<circle cx="{sx*800}" cy="{sy*800}" r="12" fill="#ffe7a7" stroke="#9b6c32" stroke-width="4"/>')
    elements.append('</svg>')
    return ''.join(elements)


def render_rgb(puzzle, frame):
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle, Circle
    fig = Figure(figsize=(7,7), facecolor='#0c1424')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111,facecolor='#0c1424')
    for x,y,w,h in puzzle.obstacles:
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#493746',edgecolor='#dc7c8e'))
    for a,b in frame['trace']['segments']:
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#ffe7a7',lw=2)
    for x,y,t in frame['poses']:
        dx,dy = puzzle.mirror_length/2*math.cos(t),puzzle.mirror_length/2*math.sin(t)
        ax.plot([x-dx,x+dx],[y-dy,y+dy],color='#83d6fc',lw=5)
    ax.add_patch(Circle(puzzle.target,puzzle.target_radius,fill=False,color='#70f1b8' if frame['solved'] else '#e9b25d'))
    ax.plot(*puzzle.source,'o',color='#ffe7a7')
    ax.set(xlim=(0,1),ylim=(1,0),aspect='equal')
    ax.tick_params(colors='#b9c9de')
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:,:,:3].copy()


STYLE = '''*{box-sizing:border-box}body{margin:0;background:#080f1d;color:#dae4f3;font:16px system-ui,sans-serif}main{max-width:1150px;margin:36px auto;padding:0 24px}h1{font-size:32px;margin:8px 0}small,.muted{color:#8fa5c1}header{margin-bottom:24px}.layout{display:grid;grid-template-columns:minmax(300px,740px) 280px;gap:24px}.board svg{display:block;width:100%;height:auto}.panel{background:#101d31;border:1px solid #293c58;border-radius:16px;padding:20px;align-self:start}button,select{color:#dce9ff;background:#1a304b;border:1px solid #3c5879;border-radius:8px;padding:10px;cursor:pointer}button:hover{background:#294969}input[type=range]{width:100%;accent-color:#83d6fc}input[type=number]{width:90px;background:#1a304b;color:white;border:1px solid #3c5879;padding:8px}label{display:block;margin:16px 0 6px}pre{white-space:pre-wrap;color:#b9cce5;font-size:13px;line-height:1.6}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.badge{color:#83d6fc;letter-spacing:.16em;font-size:12px;text-transform:uppercase}.hint{font-size:14px;line-height:1.6;color:#8fa5c1}@media(max-width:800px){.layout{grid-template-columns:1fr}main{margin:20px auto}}'''


def save_playback(path, puzzle, frames, *, title='Optics policy playback'):
    if not frames:
        raise ValueError('At least one playback frame is required')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = json.dumps([{'svg':render_svg(puzzle,f),'steps':f['steps'],'solved':f['solved'],
                           'miss':f['trace']['distance'],'stop':f['trace']['stop']} for f in frames]).replace('<','\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Optics playback</title><style>STYLE</style><main><header><div class="badge">Optics lab / frozen rollout</div><h1>TITLE</h1><span class="muted">Blue mirrors · amber laser · rose obstacles · circular target</span></header><div class="layout"><div class="board" id="board"></div><aside class="panel"><h2>Playback</h2><div class="controls"><button id="play">Play</button><button id="restart">Restart</button><select id="speed"><option value="350">Slow</option><option value="120" selected>Normal</option><option value="40">Fast</option></select></div><label for="frame">Action / frame</label><input id="frame" type="range" min="0" value="0"><pre id="metrics"></pre><p class="hint">Slide through the rollout to inspect mirror poses and the recomputed beam. The policy weights remain frozen.</p></aside></div></main><script>const frames=PAYLOAD;let i=0,timer=null;const slider=document.getElementById('frame');slider.max=frames.length-1;function draw(){const f=frames[i];document.getElementById('board').innerHTML=f.svg;slider.value=i;document.getElementById('metrics').textContent=`Step ${f.steps}\nSolved: ${f.solved}\nTarget miss: ${f.miss.toFixed(4)}\nRay ended at: ${f.stop}`;}function pause(){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';}document.getElementById('play').onclick=()=>{if(timer){pause();return;}if(i===frames.length-1)i=0;timer=setInterval(()=>{if(i<frames.length-1){i++;draw();}else pause();},Number(document.getElementById('speed').value));document.getElementById('play').textContent='Pause';};slider.oninput=()=>{pause();i=Number(slider.value);draw();};document.getElementById('restart').onclick=()=>{pause();i=0;draw();};document.getElementById('speed').onchange=pause;draw();</script></html>'''
    path.write_text(page.replace('STYLE',STYLE).replace('TITLE',html.escape(title)).replace('PAYLOAD',payload))
    return path
