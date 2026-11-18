"""EgoFalls whole-clip inference; run only with trusted project artifacts."""
from pathlib import Path
import os, json, hashlib, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import gradio as gr

INPUT_ROOT=Path(os.environ.get('EGOFALLS_INPUT_ROOT','/kaggle/input'))
OUT=Path('/kaggle/working/egofalls_demo') if Path('/kaggle').exists() else Path('./egofalls_demo_logs')
OUT.mkdir(parents=True,exist_ok=True)
# Only load .joblib files from the trusted notebook06 outputs you attached.
paths=sorted(INPUT_ROOT.rglob('test_*_motion.joblib'))
if not paths: raise RuntimeError('Attach the completed fanelp/egofalls-06-video-lopo-comparison outputs: test_*_motion.joblib are required.')
checkpoints={}
for p in paths:
    if p.stem in checkpoints: raise RuntimeError('Duplicate checkpoint names: attach one notebook06 output version only.')
    checkpoints[p.stem]=p
configs=list(INPUT_ROOT.rglob('feature_config.json'))
if not configs: raise RuntimeError('Attach notebook04 video-only-training outputs containing feature_config.json.')
import subprocess, tempfile, shutil, time
import torch, torchvision
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from tqdm.auto import tqdm
assert torch.cuda.is_available(), 'Enable a Kaggle GPU and restart the session'
DEVICE=torch.device('cuda')
print('GPU:',torch.cuda.get_device_name(0))
RESNET_WEIGHTS_PATH=None # optional official resnet18-f37072fd.pth attached as input
RAFT_WEIGHTS_PATH=None # optional official raft_small_C_T_V2-01064c6d.pth
rw=ResNet18_Weights.IMAGENET1K_V1; fw=Raft_Small_Weights.C_T_V2
def load_pretrained(builder,weights,path):
    try:
        net=builder(weights=None if path else weights)
        if path: net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        return net.to(DEVICE).eval()
    except Exception as ex:
        raise RuntimeError('Could not load official pretrained weights. Enable Internet or attach weight files and set their paths; rerun this cell.') from ex
encoder=load_pretrained(resnet18,rw,RESNET_WEIGHTS_PATH)
encoder.fc=torch.nn.Identity()
flow_model=load_pretrained(raft_small,fw,RAFT_WEIGHTS_PATH)
appearance_transform=rw.transforms()
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def weight_file(weights,path):
    return Path(path) if path else Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
SCAN_FPS=4; WIDTH=160; MAX_SECONDS=120; READ_TIMEOUT=60
APPEARANCE_FRAMES=32; MAX_FLOW_PAIRS=16; FLOW_BATCH=4; RAFT_UPDATES=6
DECODER_VERSION='ffmpeg_sequential_v1'; FLOW_BACKEND='raft_small'
extraction_config=dict(version='fullclip_uniform_raft_v2',scan_fps=SCAN_FPS,width=WIDTH,
    max_seconds=MAX_SECONDS,appearance_frames=APPEARANCE_FRAMES,max_flow_pairs=MAX_FLOW_PAIRS,
    raft_updates=RAFT_UPDATES,resnet_weight_sha256=file_hash(weight_file(rw,RESNET_WEIGHTS_PATH)),
    raft_weight_sha256=file_hash(weight_file(fw,RAFT_WEIGHTS_PATH)),
    torch=torch.__version__,torchvision=torchvision.__version__,sampling='uniform_fullclip')
EXPECTED_CONFIG_ID=hashlib.sha256(json.dumps(extraction_config,sort_keys=True).encode()).hexdigest()[:16]

def read_clip(path):
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=avg_frame_rate,width,height,duration:format=duration',
        '-of','json',str(path)],capture_output=True,text=True,timeout=20,check=True)
    info=json.loads(probe.stdout); stream=info['streams'][0]
    from fractions import Fraction
    fps=float(Fraction(stream['avg_frame_rate']))
    duration=float(stream.get('duration') or info.get('format',{}).get('duration',0))
    if not np.isfinite(fps) or fps<=0 or duration<=0: raise ValueError('Invalid media metadata')
    if duration>MAX_SECONDS: raise ValueError('Clip exceeds configured maximum duration; excluded')
    height=max(2,round(int(stream['height'])*WIDTH/int(stream['width'])))
    # One sequential decode; no repeated seeks into H.264 GOPs.
    with tempfile.TemporaryDirectory() as tmp:
        raw=Path(tmp)/'frames.rgb'; err=Path(tmp)/'decode.log'
        command=['ffmpeg','-nostdin','-hide_banner','-loglevel','warning','-threads','2',
            '-i',str(path),'-t',str(MAX_SECONDS),'-an','-vf',
            f'fps={SCAN_FPS},scale={WIDTH}:{height}', '-frames:v',str(int(MAX_SECONDS*SCAN_FPS)+1),
            '-pix_fmt','rgb24','-f','rawvideo','-y',str(raw)]
        with err.open('wb') as log:
            try: run=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=log,timeout=READ_TIMEOUT)
            except subprocess.TimeoutExpired: raise ValueError(f'Decode timeout after {READ_TIMEOUT}s')
        warning=err.read_text(errors='replace')
        log_path=OUT/('decoder_'+__import__('hashlib').sha256(str(path).encode()).hexdigest()[:16]+'.log')
        if warning: log_path.write_text(warning)
        # Reject explicit bitstream errors even when FFmpeg exits successfully.
        bad=any(k in warning.lower() for k in ['error while decoding','sps_id','invalid data','corrupt decoded','missing picture'])
        if run.returncode or bad: raise ValueError(f'Corrupt/failed decode; inspect {log_path.name}')
        data=np.fromfile(raw,dtype=np.uint8); size=WIDTH*height*3
        if len(data)%size: raise ValueError('Incomplete raw frame')
        frames=data.reshape(-1,height,WIDTH,3)
        expected=max(1,round(min(duration,MAX_SECONDS)*SCAN_FPS))
        fraction=min(1.,len(frames)/expected)
        if len(frames)<3 or fraction<.95: raise ValueError(f'Insufficient decoded coverage: {fraction:.1%}')
    return frames,np.arange(len(frames))/SCAN_FPS,dict(fps=fps,duration=duration,
        sampled_frames=len(frames),decode_fraction=fraction,truncated=duration>MAX_SECONDS,
        decoder=DECODER_VERSION,flow_backend=FLOW_BACKEND)

def extract_video_features(frames,t):
    # Independent appearance samples cover the full clip instead of only its largest peak.
    idx=np.linspace(0,len(frames)-1,APPEARANCE_FRAMES).round().astype(int)
    images=torch.from_numpy(frames[idx].copy()).permute(0,3,1,2).to(DEVICE).float()/255
    with torch.inference_mode():
        embedding=encoder(appearance_transform(images)).mean(0).cpu().numpy()
    end=np.unique(np.linspace(1,len(frames)-1,min(MAX_FLOW_PAIRS,len(frames)-1)).round().astype(int))
    diffs=[]; flows=[]
    h,w=frames.shape[1:3]; scale=max(1.,128/min(h,w))
    hh=int(np.ceil(h*scale/8)*8); ww=int(np.ceil(w*scale/8)*8)
    for start in range(0,len(end),FLOW_BATCH):
        js=end[start:start+FLOW_BATCH]
        a=torch.from_numpy(frames[js-1].copy()).permute(0,3,1,2).to(DEVICE).float()/255
        b=torch.from_numpy(frames[js].copy()).permute(0,3,1,2).to(DEVICE).float()/255
        dt=torch.as_tensor(t[js]-t[js-1],device=DEVICE,dtype=torch.float32)
        diffs.extend(((b-a).abs().mean((1,2,3))/dt).cpu().numpy().tolist())
        aa=F.interpolate(a,(hh,ww),mode='bilinear',align_corners=False)*2-1
        bb=F.interpolate(b,(hh,ww),mode='bilinear',align_corners=False)*2-1
        with torch.inference_mode():
            f=flow_model(aa,bb,num_flow_updates=RAFT_UPDATES)[-1]
            # Normalize x/y displacement by their respective resized dimensions.
            f[:,0]/=ww; f[:,1]/=hh
            mag=torch.linalg.vector_norm(f,dim=1).mean((1,2))/dt
        flows.extend(mag.cpu().numpy().tolist())
    result={f'resnet18_{i:03d}':float(v) for i,v in enumerate(embedding)}
    for prefix,values in [('frame_diff',diffs),('optical_flow',flows)]:
        for stat,value in zip(['mean','std','p90','max'],[np.mean(values),np.std(values),np.percentile(values,90),np.max(values)]):
            result[f'{prefix}_{stat}']=float(value)
    return result


# Validate the authoritative training configuration and exact extraction parameters.
bundles={k:joblib.load(p) for k,p in checkpoints.items()}
ids={b['feature_config_id'] for b in bundles.values()}
if len(ids)!=1: raise RuntimeError('Checkpoints have incompatible feature configurations.')
config_id=next(iter(ids))
matching=[]
for p in configs:
    cfg=json.loads(p.read_text())
    if cfg.get('config_id')==config_id: matching.append(cfg)
if not matching: raise RuntimeError('No feature_config.json matches the checkpoint configuration ID.')
training_config=matching[0]
payload={k:v for k,v in training_config.items() if k!='config_id'}
if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]!=config_id:
    raise RuntimeError('Training feature configuration fingerprint is invalid.')
for k,v in extraction_config.items():
    if k not in ('torch','torchvision') and training_config.get(k)!=v:
        raise RuntimeError(f'Extraction mismatch for {k}: training={training_config.get(k)!r}, demo={v!r}')
runtime_note=''
for k in ('torch','torchvision'):
    if training_config.get(k)!=extraction_config[k]:
        runtime_note+=' '+k+' version differs from training; small numerical differences are possible.'
expected={f'{p}_{s}' for p in ('frame_diff','optical_flow') for s in ('mean','std','p90','max')}
for b in bundles.values():
    if set(b['feature_columns'])!=expected or b.get('feature_set')!='motion':
        raise RuntimeError('This demo requires the eight-feature motion checkpoint.')

def infer(video_path, checkpoint_name):
    if not video_path: raise gr.Error('Upload a video first.')
    if checkpoint_name not in bundles: raise gr.Error('Choose an available checkpoint.')
    started=time.perf_counter()
    try:
        frames,t,meta=read_clip(Path(video_path))
        values=extract_video_features(frames,t)
        b=bundles[checkpoint_name]
        row=pd.DataFrame([values])[b['feature_columns']]
        if not np.isfinite(row.to_numpy()).all(): raise ValueError('Non-finite motion features.')
        cls=list(b['model'].classes_).index(1)
        score=float(b['model'].predict_proba(row)[0,cls]); threshold=float(b['threshold'])
        result='Fall detected' if score>=threshold else 'Non-fall'
        picks=np.linspace(0,len(frames)-1,6).round().astype(int)
        fig,axs=plt.subplots(1,6,figsize=(13,2.4),layout='constrained')
        for ax,i in zip(axs,picks): ax.imshow(frames[i]);ax.set_title(f'{t[i]:.2f} s');ax.axis('off')
        details=dict(decision=result,fall_score=round(score,6),threshold=round(threshold,6),
            inference_seconds=round(time.perf_counter()-started,2),video_duration_seconds=meta['duration'],
            sampled_frames=meta['sampled_frames'],decode_fraction=meta['decode_fraction'],
            model_id=b['model_id'],protocol_id=b['protocol_id'],
            held_out_test_participant=b['test_participant'],
            validation_participant=b['validation_participant'],
            training_participants=b['train_participants'],feature_config_id=config_id,
            runtime_note=runtime_note.strip() or 'Training library versions match.')
        heading=f'## {result}\nFall score **{score:.3f}** · saved threshold **{threshold:.3f}** · **{details["inference_seconds"]} s**'
        return heading,details,fig,row.T.reset_index().rename(columns={'index':'feature',0:'value'})
    except (ValueError,RuntimeError,KeyError,IndexError,subprocess.SubprocessError) as ex:
        raise gr.Error(str(ex)) from ex

with gr.Blocks(title='EgoFalls · Video demo') as demo:
    gr.Markdown('# EgoFalls · Test a video\nUpload a short video and run the saved motion model. Whole-clip research prototype; not a live-camera alarm.')
    gr.Markdown('For a held-out demonstration, choose the checkpoint named for the person in the clip. Use staged fall recordings; do not stage a fall for this demo. Scores are model outputs, not calibrated risk estimates.')
    checkpoint=gr.Dropdown(list(bundles),value='test_S_XL_motion' if 'test_S_XL_motion' in bundles else next(iter(bundles)),label='Checkpoint · held-out participant')
    video=gr.Video(label='Upload a video · maximum 120 seconds',sources=['upload'])
    run=gr.Button('Analyze video',variant='primary')
    prediction=gr.Markdown()
    with gr.Row():
        details=gr.JSON(label='Prediction and model provenance')
        features=gr.Dataframe(label='Eight motion features',interactive=False)
    plot=gr.Plot(label='Sampled frames')
    run.click(infer,[video,checkpoint],[prediction,details,plot],concurrency_limit=1)
    gr.Markdown('Known evaluation limits: 86.9% pooled recall; 39.1% worst-person recall; 43.4 false alarms per 100 non-fall clips. Internet is needed for official pretrained weights and the temporary Gradio URL.')

def launch(share=True):
    return demo.queue(max_size=4).launch(share=share,show_error=False,max_file_size='100mb')

if __name__=='__main__': launch()
