// Validate an entire mouse/keyboard batch before sending any OS/browser input.
// No DOM, task API, or environment operation is an input primitive.
const LIMIT=16;
const WIDTH=1440,HEIGHT=1000;
const COORDINATE_MAX=1000;
const modifiers=new Set(['Control','Shift','Alt','Meta']);
const keyPattern=/^(Control|Shift|Alt|Meta|Enter|Tab|Escape|Backspace|Delete|Space|ArrowLeft|ArrowRight|ArrowUp|ArrowDown|Home|End|PageUp|PageDown|[a-zA-Z0-9])$/;
function primitive(name,a){
  if(!a||typeof a!=='object'||Array.isArray(a))throw Error('Each input must be an object');
  const allowed=name==='mouse'?['operation','x','y','button','to_x','to_y','delta_x','delta_y','repeat']:['text','keys','repeat'];
  if(Object.keys(a).some(k=>!allowed.includes(k)))throw Error('Unknown input field');
  const repeat=a.repeat??1;
  if(!Number.isInteger(repeat)||repeat<1||repeat>LIMIT)throw Error('repeat must be an integer from 1 to 16');
  if(name==='mouse'){
    const point=(x,y)=>{if(!Number.isFinite(x)||!Number.isFinite(y)||x<0||x>COORDINATE_MAX||y<0||y>COORDINATE_MAX)throw Error('Mouse coordinates must be between 0 and 1000 on each axis');};
    point(a.x,a.y);
    if(!['move','click','double_click','drag','scroll'].includes(a.operation))throw Error('Unknown mouse operation');
    if(!['left','right','middle'].includes(a.button??'left'))throw Error('Invalid mouse button');
    if(a.operation==='drag')point(a.to_x,a.to_y);
    for(const k of ['delta_x','delta_y'])if(k in a&&!Number.isFinite(a[k]))throw Error('Scroll delta must be finite');
  }else{
    if(Object.hasOwn(a,'text')===Object.hasOwn(a,'keys'))throw Error('Provide exactly one of text or keys');
    if('text' in a){if(typeof a.text!=='string'||a.text.length>10000)throw Error('Invalid text');}
    else{
      if(!Array.isArray(a.keys)||!a.keys.length||a.keys.length>4||a.keys.some(k=>typeof k!=='string'||!keyPattern.test(k)))throw Error('Unsupported key');
      if(new Set(a.keys).size!==a.keys.length||a.keys.filter(k=>!modifiers.has(k)).length>1)throw Error('keys is one simultaneous chord; use sequence or repeat for successive presses');
      if(a.keys.some(k=>['Control','Meta','Alt'].includes(k))&&a.keys.some(k=>/^[a-zA-Z0-9]$/.test(k))&&!['a','c','v','x','z'].includes(a.keys.at(-1).toLowerCase()))throw Error('Browser or developer shortcut disabled');
    }
  }
  const value={...a};delete value.repeat;
  return Array.from({length:repeat},()=>({...value}));
}
function expandInputs(name,args){
  if(!['keyboard','mouse'].includes(name))throw Error('Only mouse and keyboard are available');
  if(!args||typeof args!=='object'||Array.isArray(args))throw Error('Input arguments must be an object');
  let inputs;
  if('sequence' in args){
    if(Object.keys(args).length!==1||!Array.isArray(args.sequence)||!args.sequence.length||args.sequence.length>LIMIT)throw Error('Use sequence alone, with 1 to 16 inputs');
    inputs=args.sequence.flatMap(a=>primitive(name,a));
  }else inputs=primitive(name,args);
  if(inputs.length>LIMIT)throw Error('A batch may contain at most 16 primitive inputs including repeats');
  if(inputs.reduce((n,a)=>n+(a.text?.length||0),0)>10000)throw Error('A batch may type at most 10000 characters');
  return inputs;
}
async function performInput(page,name,a){
  if(name==='mouse'){
    // Convert once at the browser boundary. Scroll deltas remain pixel distances.
    const pixel=(value,size)=>Math.min(size-1,Math.round(value*size/COORDINATE_MAX));
    const x=pixel(a.x,WIDTH),y=pixel(a.y,HEIGHT);
    await page.mouse.move(x,y);const button=a.button||'left';
    if(a.operation==='click'||a.operation==='double_click')await page.mouse.click(x,y,{button,clickCount:a.operation==='double_click'?2:1});
    else if(a.operation==='drag'){
      await page.mouse.down({button});
      try{await page.mouse.move(pixel(a.to_x,WIDTH),pixel(a.to_y,HEIGHT),{steps:12});}finally{await page.mouse.up({button});}
    }else if(a.operation==='scroll')await page.mouse.wheel({deltaX:Math.max(-1200,Math.min(1200,a.delta_x||0)),deltaY:Math.max(-1200,Math.min(1200,a.delta_y||0))});
  // Insert into the existing focus without emitting shortcut keydown events.
  // A missed field click cannot invoke navigation or focus shortcuts mid-text.
  }else if('text' in a)await page.keyboard.sendCharacter(a.text);
  else{try{for(const key of a.keys)await page.keyboard.down(key);}finally{for(const key of [...a.keys].reverse())await page.keyboard.up(key);}}
}
async function executeInputs(page,name,args,settle){
  const inputs=expandInputs(name,args);let executed=0;
  for(const input of inputs){
    await performInput(page,name,input);executed++;
    const status=await settle();
    if(status==='Action failed')return{ok:false,error:'The interface reported an action failure; the batch stopped.',executed_inputs:executed,requested_inputs:inputs.length};
    if(status==='Saved locally')break;
  }
  return{ok:true,executed_inputs:executed,requested_inputs:inputs.length};
}
module.exports={expandInputs,executeInputs,WIDTH,HEIGHT,LIMIT,COORDINATE_MAX};
