const APPS={vlnverse:'VLNVerse',dialnav:'DialNav',opensun3d:'OpenSUN3D',mevis:'MeViSv2 Text'};
const $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let app=new URLSearchParams(location.search).get('app')||'vlnverse';
if(!APPS[app])app='vlnverse';
let state={},version=-1,polling=false,requestPending=false,controlEpoch=0,mode=app==='mevis'?'new_object':'navigate';
let selectedObject=null,selectedProposal=null,refineProposal=null,accepted=new Set();
let view='image',pointer=null,inspecting=false,pendingCleanFrame=null,finishAction='finish',reviewImage=null;
let displayedVideoFrame=null;
let navigationHidden=false,precisionOpen=false;
let historyOpen=false,moveReasonOpen=false,pendingGuide=null;
const navigationDefaults={distance:app==='vlnverse'?1:0.5,angle:app==='vlnverse'?90:45,'pitch-step':15};
document.body.dataset.app=app;
$('apps').innerHTML=Object.entries(APPS).map(([id,name])=>'<a href="/?app='+id+'" class="'+(app===id?'active':'')+'">'+name+'</a>').join('');
document.title=APPS[app];
const button=(text,action,args={},className='')=>'<button class="'+className+'" data-action="'+action+'" data-args=\''+esc(JSON.stringify(args))+'\'>'+text+'</button>';
const modeButton=(label,id)=>'<button data-mode="'+id+'">'+label+'</button>';
function proposalName(p){
 const m=p.id.match(/^g(\d+)_(.+)$/);
 return m?Number(m[1])+' · '+m[2].replace('pinpoint_','').replaceAll('_',' '):p.id;
}
function proposalGroup(p){return p.group||p.id.split('_')[0];}
function groupNumber(p){return Number(proposalGroup(p).replace(/^g/,''));}
function candidateName(p){
 const name=p.id.replace(/^g\d+_/, '').replace(/^pinpoint_/, '').replaceAll('_',' ');
 return name==='volt'?'VOLT':name.charAt(0).toUpperCase()+name.slice(1);
}
function proposalGroups(){
 const groups=new Map();
 for(const p of state.proposals||[]){const id=proposalGroup(p);if(!groups.has(id))groups.set(id,[]);groups.get(id).push(p);}
 return [...groups.values()];
}
function renderProposalList(){
 return proposalGroups().map(group=>'<section class="proposal-group"><h3>Selection '+groupNumber(group[0])+
  '<span>'+group.filter(p=>p.inspected).length+' / '+group.length+' reviewed</span></h3><div class="candidate-grid">'+
  group.map(p=>'<div class="proposal '+(selectedProposal===p.id?'active':'')+'" title="'+esc(p.id)+'">'+
   '<input type="checkbox" aria-label="Include '+esc(proposalName(p))+'" data-accept="'+esc(p.id)+'" '+(accepted.has(p.id)?'checked':'')+'>'+
   '<button class="review" data-action="inspect" data-args=\''+esc(JSON.stringify({proposal_id:p.id}))+'\' aria-label="Review selection '+groupNumber(p)+' '+esc(candidateName(p))+'">'+
   '<span>'+esc(candidateName(p))+(p.inspected?' <span class="check" aria-label="Reviewed">✓</span>':'')+'</span><small>'+p.count.toLocaleString()+' pts</small></button></div>').join('')+
  '</div></section>').join('')||'<p class="empty-list">Choose New selection, then click the target in the 3D scene.</p>';
}
function renderReviewControls(){
 const p=(state.proposals||[]).find(p=>p.id===selectedProposal),bar=$('review-controls');
 bar.hidden=app!=='opensun3d'||!inspecting||!p;
 if(bar.hidden){bar.innerHTML='';return;}
 const group=(state.proposals||[]).filter(q=>proposalGroup(q)===proposalGroup(p)),index=group.findIndex(q=>q.id===p.id);
 bar.innerHTML='<div class="review-position">Selection '+groupNumber(p)+' · '+esc(candidateName(p))+'<small>'+(index+1)+' / '+group.length+' candidates · '+group.filter(q=>q.inspected).length+' reviewed</small></div>'+
  '<button data-review-step="-1" '+(index===0?'disabled':'')+'>← Previous</button><button data-review-step="1" '+(index===group.length-1?'disabled':'')+'>Next →</button>'+
  '<label class="include-current"><input type="checkbox" data-accept="'+esc(p.id)+'" '+(accepted.has(p.id)?'checked':'')+'>Include this mask</label>'+
  '<button data-refine="positive">Refine +</button><button data-refine="negative">Refine −</button>';
}
function syncAccepted(){
 document.querySelectorAll('[data-accept]').forEach(el=>el.checked=accepted.has(el.dataset.accept));
 if(!$('selection-summary'))return;
 const selectedGroups=new Set((state.proposals||[]).filter(p=>accepted.has(p.id)).map(proposalGroup));
 const remaining=(state.proposals||[]).filter(p=>selectedGroups.has(proposalGroup(p))&&!p.inspected).length;
 $('selection-summary').textContent=accepted.size+' included'+(remaining?' · '+remaining+' still need review':accepted.size?' · ready to save':'');
}
function setPointMode(next,target=null){
 mode=next;refineProposal=target;showScene();applyModes();status();
 if($('refine-proposal'))$('refine-proposal').value=target||'';
}
function reviewStep(delta){
 const p=(state.proposals||[]).find(p=>p.id===selectedProposal);if(!p)return;
 const group=(state.proposals||[]).filter(q=>proposalGroup(q)===proposalGroup(p)),index=group.findIndex(q=>q.id===p.id);
 const next=group[index+delta];if(next)control('inspect',{proposal_id:next.id});
}
function navigationControls(){
 const preset=(field,values,unit)=>values.map(v=>'<button data-nav-preset="'+field+'" data-value="'+v+'">'+v+unit+'</button>').join('');
 const look=(label,axis,sign)=>button(label+' <small data-step-label="'+axis+'"></small>','turn',axis==='angle'?{yaw:sign}:{pitch:sign},'nav-look');
 return '<div class="nav-title">Look <span>Arrow keys</span></div><div class="look-grid">'+
 look('← Left','angle',-1)+look('Right →','angle',1)+look('↑ Up','pitch-step',1)+look('Down ↓','pitch-step',-1)+
 button('Turn around <small>180°</small>','rotate',{yaw_degrees:180,pitch_degrees:0},'turn-around')+'</div>'+
 '<div class="nav-presets" aria-label="Turn step">'+preset('angle',[15,45,90],'°')+'</div>'+
 '<div class="nav-title move-title">Move <span>W A S D</span></div><div class="move-grid">'+
 button('Forward ↑','move',{direction:'forward'},'move-forward')+
 button('← Left','move',{direction:'left'})+button('Back ↓','move',{direction:'backward'})+button('Right →','move',{direction:'right'})+'</div>'+
 '<div class="nav-presets" aria-label="Movement step">'+preset('distance',[0.25,0.5,1,2],' m')+'</div>'+
 '<div class="height-controls">'+button('Lower · Q','move',{direction:'down'})+button('Raise · E','move',{direction:'up'})+'</div>'+
 '<details id="navigation-precision" '+(precisionOpen?'open':'')+'><summary>Exact settings</summary><div class="precision-fields">'+
 '<label>Move (m)<input id="distance" aria-label="Move distance in metres" type="number" min="0" max="3" step="any" value="'+navigationDefaults.distance+'"></label>'+
 '<label>Turn (°)<input id="angle" aria-label="Rotation step in degrees" type="number" min="0" max="180" step="any" value="'+navigationDefaults.angle+'"></label>'+
 '<label>Tilt (°)<input id="pitch-step" aria-label="Pitch step in degrees" type="number" min="0" max="89" step="any" value="15"></label></div><button id="precise-rotate">Custom yaw / pitch…</button></details>';
}
function syncNavigation(){
 if(!$('distance'))return;
 document.querySelectorAll('[data-step-label]').forEach(el=>el.textContent=$(''+el.dataset.stepLabel).value+'°');
 document.querySelectorAll('[data-nav-preset]').forEach(el=>el.classList.toggle('selected',Number($(el.dataset.navPreset).value)===Number(el.dataset.value)));
 const unavailable=!state.image||!!state.done||view!=='image'||inspecting||(app==='opensun3d'&&mode!=='navigate');
 $('navigation').hidden=navigationHidden||unavailable;
 $('toggle-navigation').hidden=!['vlnverse','opensun3d'].includes(app);
 $('toggle-navigation').setAttribute('aria-pressed',String(!navigationHidden&&!unavailable));
}
function renderControls(){
 const values={};
 for(const id of ['distance','angle','pitch-step','guide-message','move-reason'])if($(id))values[id]=$(id).value;
 if($('navigation-precision'))precisionOpen=$('navigation-precision').open;
 if($('guide-history'))historyOpen=$('guide-history').open;
 if($('move-options'))moveReasonOpen=$('move-options').open;
 const chatScroll=$('chat')?{top:$('chat').scrollTop,bottom:$('chat').scrollHeight-$('chat').clientHeight-$('chat').scrollTop<10}:null;
 const proposalScroll=document.querySelector('.proposals')?.scrollTop||0;
 $('navigation').innerHTML=['vlnverse','opensun3d'].includes(app)?navigationControls():'';
 $('selection-controls').innerHTML=app==='opensun3d'
  ?modeButton('Navigate','navigate')+'<button id="new-selection">New selection</button>'+modeButton('+ Point','positive')+modeButton('− Point','negative')
  :app==='mevis'?modeButton('New object','new_object')+modeButton('+ Point','refine_positive')+modeButton('− Point','refine_negative'):'';
 let html='';
 if(app==='dialnav'){
  const dialog=state.dialog||[],latest=dialog.filter(m=>m.role==='Guide').at(-1),exits=[...new Set((state.markers||[]).map(m=>m.id))];
  html='<h2 class="panel-title">Guide <span>G to type</span></h2><div class="guide-compose"><textarea id="guide-message" aria-label="Message to Guide" placeholder="Describe what you see and ask the Guide."></textarea><div class="compose-actions"><span>Ctrl + Enter to send</span>'+button('Send','guide',{},'primary')+'</div></div>'+
   '<div class="guide-reading"><section class="latest-guide"><h3>Latest reply</h3><div class="message guide">'+(latest?esc(latest.text):'No reply yet. The Guide receives your text description.')+'</div></section>'+
   '<details id="guide-history" '+(historyOpen?'open':'')+'><summary>Full conversation · '+dialog.filter(m=>m.role==='Guide').length+' replies</summary><div class="chat" id="chat">'+
   dialog.map(m=>'<div class="message '+(m.role==='Guide'?'guide':'')+'"><b>'+esc(m.role)+'</b>'+esc(m.text)+'</div>').join('')+'</div></details></div>'+
   '<div class="dial-moves"><div class="exit-label">Move to exit <span>Number keys</span></div><div class="exit-shortcuts">'+exits.map(id=>button(String(id),'move',{marker:id})).join('')+'</div>'+
   '<details id="move-options" '+(moveReasonOpen?'open':'')+'><summary>Move reason (optional)</summary><input id="move-reason" aria-label="Move reason" type="text" placeholder="Reason for the next exit"></details></div>';
 }
 if(app==='opensun3d'){
  html='<h2 class="panel-title">3D masks <span>'+(state.proposals||[]).length+' candidates</span></h2>'+
   '<div class="panel-body point-context"><div id="point-context"></div><label for="refine-proposal">Refine a mask</label><select id="refine-proposal"><option value="">Choose a mask…</option>'+
   (state.proposals||[]).map(p=>'<option value="'+esc(p.id)+'">Refine '+esc(proposalName(p))+'</option>').join('')+'</select></div>'+
   '<div class="proposals">'+renderProposalList()+'</div><div class="selection-summary" id="selection-summary"></div>'+
   '<div class="panel-bottom"><button id="references">Reference videos</button><button id="pose">Go to recorded pose…</button></div>';
 }
 if(app==='mevis'){
  html='<h2 class="panel-title">Objects <span>'+(state.objects||[]).length+'</span></h2><div class="objects">'+
   ((state.objects||[]).map((id,i)=>'<div class="object '+(selectedObject===id?'active':'')+'">'+
    '<span class="swatch" style="background:#2de8d2"></span>'+
    '<button class="name" data-object="'+id+'">Object '+id+'</button>'+
    '<button class="remove" data-action="remove" data-args=\'{"object_id":'+id+'}\' aria-label="Remove object '+id+'">×</button></div>').join('')||
    '<p class="empty-list">Click the video frame to select an object.</p>')+
   '</div><div class="panel-bottom">'+button('Track full video','inspect',{},'primary')+'</div>';
 }
 $('tools').innerHTML=html;
 $('completion').innerHTML=(app==='mevis'?button('No target','request_empty'):'')+
  button(['vlnverse','dialnav'].includes(app)?'Stop & save':'Save prediction','request_finish',{},'primary');
 for(const [id,value]of Object.entries(values))if($(id))$(id).value=value;
 syncNavigation();
 if($('refine-proposal'))$('refine-proposal').value=refineProposal||'';
 if($('chat'))$('chat').scrollTop=chatScroll&&!chatScroll.bottom?chatScroll.top:$('chat').scrollHeight;
 if(document.querySelector('.proposals'))document.querySelector('.proposals').scrollTop=proposalScroll;
 renderReviewControls();syncAccepted();
 applyModes();
}
function applyModes(){
 document.body.dataset.activeMode=mode;
 document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('selected',b.dataset.mode===mode&&!(app==='opensun3d'&&inspecting)));
 if(app==='opensun3d'){
  const p=(state.proposals||[]).find(p=>p.id===refineProposal),fresh=mode==='positive'&&!p&&!inspecting;
  $('new-selection').classList.toggle('selected',fresh);
  document.querySelector('[data-mode="positive"]').classList.toggle('selected',mode==='positive'&&!!p&&!inspecting);
  $('point-context').textContent=inspecting?'Reviewing a 3D mask':mode==='navigate'?'Move to the target, then choose New selection.':p?'Refining '+proposalName(p)+' · '+(mode==='negative'?'negative':'positive')+' point':'New selection · click the target';
  $('hint').textContent=inspecting?'← → review candidates · Check masks to include · Refine + / − adds a point in the 3D scene':'W A S D move · ← → turn · ↑ ↓ tilt · Q / E lower / raise · N new selection';
 }
 syncNavigation();
}
function selectedObjects(){return state.objects||[];}
function fieldRows(fields){
 return '<dl>'+Object.entries(fields||{}).map(([k,v])=>'<dt>'+esc(k.replaceAll('_',' '))+'</dt><dd>'+esc(typeof v==='object'?JSON.stringify(v,null,2):v)+'</dd>').join('')+'</dl>';
}
function showDetails(){
 $('details-content').innerHTML='<h3>Current observation</h3><p>'+esc(state.observation?.text||state.message||'')+'</p>'+fieldRows(state.observation?.fields)+
  '<h3>Last operation</h3>'+fieldRows(state.feedback)+
  (app==='opensun3d'?'<h3>3D proposals</h3>'+(state.proposals||[]).map(p=>fieldRows(p)).join(''):'');
}
function cameraText(){
 const c=state.camera;if(!c)return '';
 const p=c.position_m||c.position;if(!p)return '';
 return 'Position ('+p.map(v=>Number(v).toFixed(3)).join(', ')+') m · Yaw '+Number(c.yaw_deg??c.yaw_degrees).toFixed(1)+'° · Pitch '+Number(c.pitch_deg??c.pitch_degrees).toFixed(1)+'°';
}
function playbackFrame(){return Math.max(0,Math.min(state.frame_count-1,displayedVideoFrame??Math.floor(($('video').currentTime+1e-5)*(state.fps||5))));}
function syncVideoTimeline(){
 if(app!=='mevis'||view==='image'||!$('frame-number'))return;
 const frame=playbackFrame();
 if(document.activeElement!==$('frame-number'))$('frame-number').value=frame;
 if(document.activeElement!==$('frame-slider'))$('frame-slider').value=frame;
}
function selectDisplayedVideoFrame(){
 if(app!=='mevis'||view==='image')return;
 const frame=playbackFrame();
 view='image';state.display='image';changeView('image');
 if(frame!==state.frame_index)send('seek',{frame_index:frame});
}
function changeView(next){
 if(next==='overlay'&&!state.overlay_video)return;
 view=next;
 const video=view==='original'?state.original_video:view==='overlay'?state.overlay_video:null;
 $('scene').classList.toggle('visible',!video&&!!state.image);
 $('video').classList.toggle('visible',!!video);
 if(video){if($('video').getAttribute('src')!==video){displayedVideoFrame=null;$('video').src=video;}}
 else $('video').pause();
 $('markers').style.display=video?'none':'block';
 $('image-scale').disabled=!!video||!state.image;
 $('stage').classList.toggle('native',!video&&$('image-scale').dataset.native==='true');
 document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===view));
 $('view-meta').textContent=app==='mevis'?(video?(view==='overlay'&&!state.preview_current?'Overlay · tracking outdated':'Full video'):'Frame '+(state.frame_index??0)):
  app==='dialnav'?'360° panorama':app==='opensun3d'?(inspecting?'Proposal review':'Faro scan'):'First-person view';
 if(cameraText())$('view-meta').textContent=cameraText();
 syncVideoTimeline();
 syncNavigation();
 status();
}
function imageBounds(){
 const r=$('stage').getBoundingClientRect(),im=$('scene');
 if(!im.naturalWidth)return null;
 if($('stage').classList.contains('native')){
  const b=im.getBoundingClientRect();
  return{left:b.left,top:b.top,width:b.width,height:b.height,ox:b.left-r.left+$('stage').scrollLeft,oy:b.top-r.top+$('stage').scrollTop};
 }
 const scale=Math.min(r.width/im.naturalWidth,r.height/im.naturalHeight),w=im.naturalWidth*scale,h=im.naturalHeight*scale;
 return{left:r.left+(r.width-w)/2,top:r.top+(r.height-h)/2,width:w,height:h,ox:(r.width-w)/2,oy:(r.height-h)/2};
}
function markers(){
 const b=imageBounds();if(!b)return;
 const exits=(state.markers||[]).flatMap(m=>[-1,0,1].map(offset=>({...m,x:m.x+offset})).filter(n=>n.x*b.width>=-15&&n.x*b.width<=b.width+15));
 $('markers').innerHTML=app==='dialnav'?exits.map(m=>'<button class="exit-marker" style="left:'+(b.ox+m.x*b.width)+'px;top:'+(b.oy+m.y*b.height)+'px" data-action="move" data-args=\'{"marker":'+m.id+'}\' aria-label="Exit '+m.id+'">'+m.id+'</button>').join(''):'';
}
function render(){
 document.body.dataset.revision=String(state.version);
 document.body.classList.toggle('evaluation',!!state.evaluation);
 inspecting=Boolean(state.reviewing);
 if(inspecting){reviewImage=state.image;selectedProposal=state.reviewing;}
 $('task-meta').textContent=APPS[app]+' · '+(state.task_id||'TASK');
 $('objective').textContent=state.objective||'Open a task to begin.';
 $('empty').style.display=state.image?'none':'block';
 if(state.image&&$('scene').getAttribute('src')!==state.image)$('scene').src=state.image;
 $('view-tabs').innerHTML=app==='mevis'
  ?'<button data-view="image">Frame</button><button data-view="original">Video</button><button data-view="overlay" '+(state.overlay_video?'':'disabled')+'>Tracked masks</button>'
  :app==='opensun3d'?'<button id="return-scene" class="'+(inspecting?'':'selected')+'">3D scene</button><button id="return-review" class="'+(inspecting?'selected':'')+'" '+(reviewImage?'':'disabled')+'>Proposal review</button>'
  :'<span class="view-label">'+(app==='dialnav'?'Navigator':'Scene')+'</span>';
 if(app==='mevis'){
  $('timeline').style.display='block';
  $('timeline').innerHTML='<div class="row"><button data-step="-1" aria-label="Previous frame">‹</button>'+
   '<input id="frame-slider" aria-label="Frame timeline" type="range" min="0" max="'+Math.max(0,(state.frame_count||1)-1)+'" value="'+(state.frame_index||0)+'">'+
   '<button data-step="1" aria-label="Next frame">›</button><span>Frame</span><input id="frame-number" aria-label="Frame number" type="number" min="0" max="'+Math.max(0,(state.frame_count||1)-1)+'" value="'+(state.frame_index||0)+'">'+
   '<span id="frame-total">/ '+Math.max(0,(state.frame_count||1)-1)+'</span>'+button('Go','frame')+'</div>';
 }
 renderControls();
 if(state.display==='overlay'&&state.preview_current)view='overlay';
 changeView(view);
 $('hint').textContent=app==='mevis'?'Click to select · ← → change frame':app==='dialnav'?'Click an exit or press its number · G message · Ctrl + Enter send':'W A S D move · ← → turn · ↑ ↓ tilt · Q / E lower / raise';
 if(app==='opensun3d')applyModes();
 markers();
 if($('details-dialog').open)showDetails();
}
function status(){
 $('status').textContent=state.busy?'Processing':state.done?'Saved locally':state.error?'Action failed':state.status==='idle'?'Task closed':'Live';
 $('status').classList.toggle('error',!!state.error);
 $('busy').classList.toggle('visible',!!state.busy);
 $('busy-text').textContent=state.status==='loading'?'Loading…':'Processing…';
 $('feedback').textContent=state.error||state.message||'Open a task to begin.';
 if(!state.busy&&!state.error&&!state.done){
  const f=state.feedback||{};
  if(app==='vlnverse'&&f.action_result){
   $('feedback').textContent=f.action_result.replaceAll('_',' ')+(f.requested_distance_m!==undefined?' · requested '+f.requested_distance_m+' m · travelled '+f.achieved_distance_m+' m':'');
  }else if(app==='opensun3d'&&f.proposal_id){
   $('feedback').textContent=f.proposal_id+' · '+f.source+' · '+f.point_count+' points · threshold '+f.threshold+' · mean probability '+Number(f.mean_foreground_probability).toFixed(4)+(f.affordance_class?' · '+f.affordance_class:'');
  }else if(app==='opensun3d'&&f.resolved_seconds!==undefined){
   $('feedback').textContent='Video '+f.video_id+' · requested '+f.requested_seconds+' s · camera at '+f.resolved_seconds+' s';
  }else if(app==='mevis'&&f.returned_mask_pixels){
   $('feedback').textContent='Frame '+f.frame_index+' · mask pixels: '+f.returned_object_ids.map((id,i)=>'object '+id+' = '+f.returned_mask_pixels[i]).join(', ');
  }
 }
 if(!state.busy&&!state.error&&!state.done&&app==='opensun3d'&&!inspecting&&state.reviewing)$('feedback').textContent='Faro scan · current camera view';
 if(!state.busy&&!state.error&&!state.done&&app==='mevis'&&view==='overlay')$('feedback').textContent=state.preview_current?'Tracked masks · '+state.frame_count+' frames':'Tracking is outdated. Track the video after editing.';
 const blocked=!!state.busy||!!state.done;
 document.querySelectorAll('[data-action],[data-mode],[data-object],[data-accept],[data-step],[data-nav-preset],#frame-slider,#frame-number,#refine-proposal,#distance,#angle,#pitch-step,#precise-rotate').forEach(el=>el.disabled=blocked||!state.image);
 $('start').disabled=!!state.busy;
 $('restart').disabled=!!state.busy;
 document.querySelectorAll('[data-mode="refine_positive"],[data-mode="refine_negative"]').forEach(b=>b.disabled=blocked||selectedObject===null);
 if(app==='dialnav'){
  $('guide-message').disabled=blocked||!state.image;
  $('move-reason').disabled=blocked||!state.image;
 }
 if(app==='opensun3d'){
  $('new-selection').disabled=blocked||!state.image;
  document.querySelectorAll('[data-mode="positive"],[data-mode="negative"]').forEach(b=>b.disabled=blocked||!state.image||!(refineProposal||selectedProposal));
  const p=(state.proposals||[]).find(p=>p.id===selectedProposal);
  const group=p?(state.proposals||[]).filter(q=>proposalGroup(q)===proposalGroup(p)):[],index=group.indexOf(p);
  document.querySelectorAll('[data-review-step]').forEach(b=>b.disabled=blocked||!group[index+Number(b.dataset.reviewStep)]);
  document.querySelectorAll('[data-refine]').forEach(b=>b.disabled=blocked||!p);
 }
}
function acceptState(data){
 state=data;
 if(pendingGuide&&!state.busy&&state.version>pendingGuide.version){
  const sent=(state.dialog||[]).filter(m=>m.role==='Navigator').at(-1)?.text;
  if(!state.error&&sent===pendingGuide.text&&$('guide-message')?.value.trim()===pendingGuide.text)$('guide-message').value='';
  pendingGuide=null;
 }
 if(state.version!==version){version=state.version;render();}
 status();
 if(pendingCleanFrame&&state.version>pendingCleanFrame.version&&!state.busy){
  const pending=pendingCleanFrame;pendingCleanFrame=null;
  if(!state.error){view='image';send('seek',{frame_index:pending.frame});}
 }
}
async function poll(){
 if(polling)return;polling=true;
 const epoch=controlEpoch;
 try{
  const r=await fetch('/api/state/'+app),data=await r.json();
  if(epoch!==controlEpoch||requestPending)return;
  acceptState(data);
 }catch(e){$('feedback').textContent='Connection lost. Check the local GUI server.';}
 finally{polling=false;}
}
async function send(action,args={}){
 if(state.busy||(state.done&&action!=='restart'))return;
 if(action==='remove'&&app==='mevis')pendingCleanFrame={version:state.version,frame:state.frame_index};
 controlEpoch++;requestPending=true;
 state.busy=true;state.error=null;status();
 try{
  const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({app,action,arguments:args})});
  const data=await r.json();if(!r.ok)throw new Error(data.error);
  requestPending=false;acceptState(data);
 }catch(e){requestPending=false;state.busy=false;state.error=e.message;status();}
}
function showScene(){
 if(!state.scene_image)return;
 inspecting=false;state.image=state.scene_image;state.reviewing=null;
 state.message='Faro scan · current camera view';
 $('scene').src=state.scene_image;
 $('return-scene').classList.add('selected');$('return-review').classList.remove('selected');
 changeView('image');
 renderReviewControls();
 if(app==='opensun3d')applyModes();
}
function requestFinish(empty=false){
 if(state.busy||state.done||!state.image)return;
 if(!empty&&app==='opensun3d'&&!accepted.size){state.error='Select a proposal to save.';status();return;}
 if(!empty&&app==='opensun3d'){
  const groups=new Set((state.proposals||[]).filter(p=>accepted.has(p.id)).map(proposalGroup));
  const unseen=(state.proposals||[]).filter(p=>groups.has(proposalGroup(p))&&!p.inspected);
  if(unseen.length){state.error='Review the remaining candidates before saving: '+unseen.map(proposalName).join(', ');status();return;}
 }
 if(!empty&&app==='mevis'&&(!state.preview_current||!selectedObjects().length)){state.error=state.preview_current?'Select an object to save.':'Track the current selections before saving.';status();return;}
 finishAction=empty?'submit_no_target':'finish';
 $('finish-title').textContent=empty?'Save empty prediction':['vlnverse','dialnav'].includes(app)?'Stop & save trajectory':'Save prediction';
 $('finish-summary').textContent=empty?'No matching target.':app==='opensun3d'?accepted.size+' selected proposal(s).':app==='mevis'?selectedObjects().length+' selected object(s).':'Finish at the current camera location.';
 $('finish-dialog').showModal();
}
async function control(action,args){
 if(action==='request_finish'){requestFinish();return;}
 if(action==='request_empty'){requestFinish(true);return;}
 if(action==='move'&&app!=='dialnav'){
  if(!$('distance').reportValidity())return;
  args.distance_m=Number($('distance').value);
 }
 if(action==='move'&&app==='dialnav')args.reason=$('move-reason').value.trim()||'Selected visible exit '+args.marker+'.';
 if(action==='turn'){
  if(!$('angle').reportValidity()||!$('pitch-step').reportValidity())return;
  action='rotate';args={yaw_degrees:(args.yaw||0)*Number($('angle').value),pitch_degrees:(args.pitch||0)*Number($('pitch-step').value)};
 }
 if(action==='guide'){
  const text=$('guide-message').value.trim();if(!text)return;
  action='type';args={text};pendingGuide={text,version:state.version};
 }
 if(action==='frame'){
  if(!$('frame-number').reportValidity())return;
  action='seek';args={frame_index:Number($('frame-number').value)};view='image';
 }
 if(action==='finish'){
  const reason=$('submit-reason').value;
  if(app==='vlnverse'||app==='dialnav'){action='stop';args={reason};}
  if(app==='opensun3d'){action='submit';args={proposal_ids:[...accepted],reason};}
  if(app==='mevis'){action='submit';args={object_ids:selectedObjects().map(String),reason};}
 }
 if(action==='submit_no_target')args.reason=$('submit-reason').value;
 if(action==='inspect'&&app==='opensun3d'){refineProposal=null;view='image';}
 if(action==='remove'&&selectedObject===args.object_id){selectedObject=null;mode='new_object';}
 await send(action,args);
}
function seekFrame(frame){view='image';send('seek',{frame_index:Math.max(0,Math.min(state.frame_count-1,frame))});}
document.addEventListener('click',e=>{
 const b=e.target.closest('button');if(!b||b.disabled)return;
 if(b.dataset.close){$(b.dataset.close).close();return;}
 if(b.dataset.navPreset){$(b.dataset.navPreset).value=b.dataset.value;syncNavigation();return;}
 if(b.id==='toggle-navigation'){
  navigationHidden=!$('navigation').hidden;
  if(!navigationHidden&&app==='opensun3d'&&(inspecting||mode!=='navigate')){mode='navigate';applyModes();showScene();}
  syncNavigation();return;
 }
 if(b.dataset.action)control(b.dataset.action,JSON.parse(b.dataset.args||'{}'));
 if(b.dataset.mode){
  if(app==='opensun3d'){
   if(b.dataset.mode==='navigate'){mode='navigate';showScene();applyModes();}
   else setPointMode(b.dataset.mode,refineProposal||selectedProposal);
   return;
  }
  mode=b.dataset.mode;applyModes();
  if(app==='opensun3d'&&inspecting)showScene();
  if(view!=='image'){selectDisplayedVideoFrame();state.display='image';changeView('image');}
 }
 if(b.id==='new-selection'){setPointMode('positive');$('refine-proposal').value='';}
 if(b.dataset.refine){setPointMode(b.dataset.refine,selectedProposal);$('refine-proposal').value=refineProposal||'';}
 if(b.dataset.reviewStep)reviewStep(Number(b.dataset.reviewStep));
 if(b.dataset.object){selectedObject=Number(b.dataset.object);mode='refine_positive';renderControls();status();}
 if(b.dataset.view){
  if(b.dataset.view==='image')selectDisplayedVideoFrame();
  state.display=b.dataset.view;changeView(b.dataset.view);
 }
 if(b.dataset.step)seekFrame((view==='image'?state.frame_index:playbackFrame())+Number(b.dataset.step));
 if(b.id==='return-scene')showScene();
 if(b.id==='return-review'&&reviewImage){
  inspecting=true;state.image=reviewImage;$('scene').src=reviewImage;
  $('return-review').classList.add('selected');$('return-scene').classList.remove('selected');changeView('image');renderReviewControls();applyModes();status();
 }
 if(b.id==='references')openReferences();
 if(b.id==='pose'){
  const value=$('pose-video').value;
  $('pose-video').innerHTML=(state.references||[]).map((r,i)=>'<option value="'+esc(r.id)+'">Video '+(i+1)+' · '+esc(r.id)+'</option>').join('');
  if(value)$('pose-video').value=value;
  updatePoseLimit();$('pose-dialog').showModal();
 }
 if(b.id==='precise-rotate')$('rotate-dialog').showModal();
 if(b.id==='details'){showDetails();$('details-dialog').showModal();}
 if(b.id==='instructions'){
  $('task-instructions').textContent=state.instructions||'Open a task to read its instructions.';
  $('control-instructions').textContent={
   vlnverse:'Move with W/A/S/D or the Move pad. Default movement is 1 m. Left/right arrows turn 90° by default; up/down arrows tilt 15°. Each pad button performs one movement or turn. Presets change step size in one click. Q/E lower/raise the camera. Exact settings retains arbitrary distance (0–3 m), yaw (−180–180°) and pitch (−89–89°). Controls hides or shows the pad. Stop & save records the final position and reason.',
   dialnav:'Click an exit in the panorama or the numbered exit row, or press its number key. G focuses the Guide message field. Send or Ctrl+Enter sends the text you typed; Enter alone inserts a line break. The latest Guide reply stays above Full conversation, which retains all exchanges. Move reason is optional. Stop & save records the final location and reason. The image edges join; exit numbers apply to the current image only.',
   opensun3d:'Move with W/A/S/D or the Move pad (default 0.5 m). Left/right arrows turn 45°; up/down arrows tilt 15°. Presets change step size; Exact settings retains arbitrary camera parameters. New selection (N) starts a fresh positive click on the Faro scene. Each selection produces a group of candidate masks. Click a candidate to review its four-view montage; Previous/Next or left/right arrows review its siblings. Check masks to include. Refine + or Refine − returns to the 3D scene with that exact mask as the point target. The Refine a mask menu also allows any earlier mask to be refined. New selection always clears the refinement target. Escape returns to navigation. Review all candidates from each included selection before saving. Reference videos and Go to recorded pose remain available independently of 3D navigation.',
   mevis:'Use the frame number or timeline to inspect any frame. New object adds a selection at your click. Select an object in the list, then use + Point or − Point to refine it. Remove deletes an object. Track full video generates the complete overlay. After every edit, track and review again. Save prediction includes all active objects; No target saves an empty answer.'
  }[app]+'\n\n100% shows the image at its native resolution with scrollbars. Fit restores the full image. Observation details contains the current API-visible feedback.';
  $('instructions-dialog').showModal();
 }
 if(b.id==='image-scale'){
  const native=b.dataset.native!=='true';b.dataset.native=String(native);b.textContent=native?'Fit':'100%';
  b.setAttribute('aria-label',native?'Fit image to viewer':'Show image at native resolution');
  $('stage').classList.toggle('native',native);$('stage').scrollLeft=0;$('stage').scrollTop=0;markers();
 }
});
document.addEventListener('change',e=>{
 if(['distance','angle','pitch-step'].includes(e.target.id))syncNavigation();
 if(e.target.id==='frame-slider')seekFrame(Number(e.target.value));
 if(e.target.id==='refine-proposal')setPointMode('positive',e.target.value||null);
 if(e.target.id==='pose-video')updatePoseLimit();
 if(e.target.dataset.accept){if(e.target.checked)accepted.add(e.target.dataset.accept);else accepted.delete(e.target.dataset.accept);syncAccepted();}
});
function updatePoseLimit(){
 const r=(state.references||[]).find(r=>r.id===$('pose-video').value);
 if(r)$('pose-seconds').max=r.seconds;
}
$('finish-form').onsubmit=e=>{e.preventDefault();$('finish-dialog').close();control(finishAction,{});};
$('pose-form').onsubmit=e=>{e.preventDefault();$('pose-dialog').close();view='image';send('seek',{video_id:$('pose-video').value,seconds:Number($('pose-seconds').value)});};
$('rotate-form').onsubmit=e=>{e.preventDefault();$('rotate-dialog').close();view='image';send('rotate',{yaw_degrees:Number($('yaw-degrees').value),pitch_degrees:Number($('pitch-degrees').value)});};
$('start').onclick=()=>send('start');
$('restart').onclick=()=>{
 if(state.busy)return;
 accepted.clear();selectedObject=null;selectedProposal=null;refineProposal=null;reviewImage=null;pendingGuide=null;historyOpen=false;view='image';mode=app==='mevis'?'new_object':'navigate';$('submit-reason').value='';send('restart');
};
$('scene').onload=markers;
new ResizeObserver(markers).observe($('stage'));
$('scene').addEventListener('contextmenu',e=>e.preventDefault());
$('scene').addEventListener('pointerdown',e=>{
 if(state.busy||state.done||inspecting)return;
 pointer={x:e.clientX,y:e.clientY,button:e.button};e.preventDefault();$('scene').setPointerCapture(e.pointerId);
});
$('scene').addEventListener('pointerup',e=>{
 if(!pointer||state.busy||state.done)return;
 const p=pointer;pointer=null;const dx=e.clientX-p.x,dy=e.clientY-p.y;
 if((app==='vlnverse'||app==='opensun3d')&&Math.hypot(dx,dy)>7){
  view='image';send('rotate',{yaw_degrees:Math.max(-180,Math.min(180,-dx*.15)),pitch_degrees:Math.max(-89,Math.min(89,dy*.15))});return;
 }
 if(!['opensun3d','mevis'].includes(app)||mode==='navigate'||inspecting||p.button!==0)return;
 const b=imageBounds();if(!b)return;const x=(e.clientX-b.left)/b.width,y=(e.clientY-b.top)/b.height;
 if(x<0||x>1||y<0||y>1)return;
 const args={x:x*1000,y:y*1000};
 if(app==='mevis'){
  args.frame_index=state.frame_index;args.mode=mode;
  if(mode!=='new_object'){
   if(selectedObject===null){state.error='Select an object to refine first.';status();return;}
   args.object_id=selectedObject;
  }
 }else{args.polarity=mode;if(refineProposal)args.proposal_id=refineProposal;}
 const r=$('stage').getBoundingClientRect();$('crosshair').style.cssText='display:block;left:'+(e.clientX-r.left)+'px;top:'+(e.clientY-r.top)+'px';
 send('click',args);setTimeout(()=>$('crosshair').style.display='none',1500);
});
document.addEventListener('keydown',e=>{
 if(app==='dialnav'&&e.target.id==='guide-message'&&e.key==='Enter'&&(e.ctrlKey||e.metaKey)&&!e.repeat&&!state.busy&&!state.done){e.preventDefault();control('guide',{});return;}
 if(e.ctrlKey||e.metaKey||e.altKey||e.target.matches('input:not([type=checkbox]),textarea,select,video')||document.querySelector('dialog[open]')||state.busy||state.done||!state.image||e.repeat||view!=='image')return;
 if(app==='dialnav'){
  if(e.key.toLowerCase()==='g'){e.preventDefault();$('guide-message').focus();}
  else if(/^[1-9]$/.test(e.key)&&(state.markers||[]).some(m=>m.id===Number(e.key))){e.preventDefault();control('move',{marker:Number(e.key)});}
  return;
 }
 if(app==='opensun3d'){
  if(e.key.toLowerCase()==='n'){e.preventDefault();setPointMode('positive');$('refine-proposal').value='';return;}
  if(e.key==='Escape'){e.preventDefault();mode='navigate';showScene();applyModes();return;}
  if(inspecting){if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();reviewStep(e.key==='ArrowLeft'?-1:1);}return;}
 }
 if(app==='vlnverse'||app==='opensun3d'){
  const direction={w:'forward',s:'backward',a:'left',d:'right',q:'down',e:'up'}[e.key.toLowerCase()];
  const turn={ArrowLeft:{yaw:-1},ArrowRight:{yaw:1},ArrowUp:{pitch:1},ArrowDown:{pitch:-1}}[e.key];
  if(direction){e.preventDefault();control('move',{direction});}
  else if(turn){e.preventDefault();control('turn',turn);}
 }else if(app==='mevis'&&['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();seekFrame(state.frame_index+(e.key==='ArrowRight'?1:-1));}
});
function openReferences(){
 $('reference-tabs').innerHTML=(state.references||[]).map((r,i)=>'<button data-reference="'+i+'">Video '+(i+1)+'</button>').join('');
 $('reference-dialog').showModal();showReference(0);
}
function showReference(index){
 const r=state.references[index];if(!r||!r.url)return;
 $('reference-video').src=r.url;document.querySelectorAll('[data-reference]').forEach(b=>b.classList.toggle('selected',Number(b.dataset.reference)===index));
}
$('reference-tabs').onclick=e=>{if(e.target.dataset.reference)showReference(Number(e.target.dataset.reference));};
$('close-reference').onclick=()=>{$('reference-video').pause();$('reference-dialog').close();};
$('reference-dialog').addEventListener('close',()=>$('reference-video').pause());
$('video').addEventListener('timeupdate',syncVideoTimeline);
$('video').addEventListener('loadedmetadata',syncVideoTimeline);
if($('video').requestVideoFrameCallback){
 const presented=(now,metadata)=>{
  displayedVideoFrame=Math.round(metadata.mediaTime*(state.fps||5));
  syncVideoTimeline();$('video').requestVideoFrameCallback(presented);
 };
 $('video').requestVideoFrameCallback(presented);
}
poll();setInterval(poll,650);
