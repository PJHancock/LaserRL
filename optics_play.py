"""Local manual control. HTTP requests call the same Python physics as training."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from optics_env import ContinuousOpticsEnv, puzzle_named
from optics_render import STYLE, render_svg

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Optics lab</title><style>STYLE</style><main><header><div class="badge">Optics lab / manual puzzle</div><h1>Find the light.</h1><span class="muted">Move and rotate the mirrors to guide the beam into the target.</span></header><div class="layout"><div id="board" class="board"></div><aside class="panel"><h2>Mirror controls</h2><label for="mirror">Selected mirror</label><select id="mirror"></select><div class="controls"><button data-move="0,-1,0">↑</button><button data-move="-1,0,0">←</button><button data-move="1,0,0">→</button><button data-move="0,1,0">↓</button></div><div class="controls"><button data-move="0,0,-1">Rotate −</button><button data-move="0,0,1">Rotate +</button></div><label for="size">Action magnitude</label><input id="size" type="range" min="0.05" max="1" value="0.3" step="0.05"><label for="seed">Starting-pose seed</label><input id="seed" type="number" min="0" value="0"><div class="controls"><button id="reset">Reset seed</button><button id="next">Next seed</button></div><pre id="metrics"></pre><p class="hint">Arrows move the selected mirror. Q/E rotate it. Circles mark the laser source and target; rose rectangles block the beam. The step limit is 100 actions.</p></aside></div></main><script>let busy=false;async function update(body){if(busy)return;busy=true;try{const r=await fetch('/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const s=await r.json();if(!r.ok)throw Error(s.error);document.getElementById('board').innerHTML=s.svg;document.getElementById('metrics').textContent=`Step ${s.steps} / ${s.limit}\nSolved: ${s.solved}\nMiss distance: ${s.distance.toFixed(4)}\nReward: ${s.reward.toFixed(4)}\n${s.done?'Episode ended. Reset to continue.':''}`;const select=document.getElementById('mirror');if(!select.options.length)for(let i=0;i<s.mirrors;i++)select.add(new Option(`Mirror ${i+1}`,i));}catch(e){alert(e.message);}finally{busy=false;}}function move(v){update({move:v.map(x=>x*Number(document.getElementById('size').value)),mirror:Number(document.getElementById('mirror').value)});}document.querySelectorAll('[data-move]').forEach(b=>b.onclick=()=>move(b.dataset.move.split(',').map(Number)));document.getElementById('reset').onclick=()=>update({seed:Number(document.getElementById('seed').value)});document.getElementById('next').onclick=()=>{document.getElementById('seed').value++;document.getElementById('reset').click();};window.onkeydown=e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;const keys={ArrowUp:[0,-1,0],ArrowDown:[0,1,0],ArrowLeft:[-1,0,0],ArrowRight:[1,0,0],q:[0,0,-1],e:[0,0,1]};if(keys[e.key]){e.preventDefault();move(keys[e.key]);}};update({seed:0});</script></html>'''.replace('STYLE',STYLE)


def serve(puzzle='single', port=8765):
    env = ContinuousOpticsEnv(puzzle_named(puzzle))
    env.reset(seed=0)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/', '/index.html'):
                self.send_error(404); return
            data=PAGE.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            if self.path != '/state':
                self.send_error(404); return
            try:
                body=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
                reward=0.0
                if 'seed' in body:
                    env.reset(seed=int(body['seed']))
                else:
                    action=np.zeros((env.num_mirrors,3),dtype=np.float32)
                    action[int(body['mirror'])]=body['move']
                    _,reward,_,_,_=env.step(action.ravel())
                frame=env.frame()
                response={'svg':render_svg(env.puzzle,frame),'mirrors':env.num_mirrors,'steps':env.steps,
                          'limit':env.max_steps,'solved':env.trace['solved'],'distance':env.trace['distance'],
                          'reward':reward,'done':env._finished}
                status=200
            except (ValueError,KeyError,IndexError,RuntimeError) as error:
                response={'error':str(error)};status=400
            self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(response).encode())
        def log_message(self, *args):
            pass
    print(f'Open http://127.0.0.1:{port} — Ctrl-C to stop',flush=True)
    server=HTTPServer(('127.0.0.1',port),Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--puzzle',default='single');parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();serve(args.puzzle,args.port)
