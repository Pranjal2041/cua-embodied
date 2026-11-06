const {test}=require('node:test');
const assert=require('node:assert/strict');
const {expandInputs,executeInputs}=require('../cua_embodied/computer/driver/browser_inputs.cjs');

test('repeat and sequence expand into successive inputs',()=>{
  assert.equal(expandInputs('keyboard',{keys:['w'],repeat:3}).length,3);
  assert.deepEqual(expandInputs('keyboard',{sequence:[{keys:['w'],repeat:2},{keys:['ArrowRight']}]}).map(a=>a.keys),[['w'],['w'],['ArrowRight']]);
  assert.equal(expandInputs('mouse',{sequence:[{operation:'click',x:1,y:2,repeat:2},{operation:'move',x:3,y:4}]}).length,3);
  assert.equal(expandInputs('keyboard',{sequence:[{keys:['Control','a']},{text:'1.5'}]})[1].text,'1.5');
});
test('invalid batches reject before any input',async()=>{
  let calls=0;const page={mouse:{move:()=>calls++}};
  await assert.rejects(()=>executeInputs(page,'mouse',{sequence:[{operation:'click',x:5,y:5},{operation:'click',x:1440,y:5}]},async()=> 'Live'));
  assert.equal(calls,0);
  for(const args of [{keys:['w'],repeat:17},{keys:['w','w']},{keys:['w','ArrowRight']},{sequence:[{keys:['w'],repeat:16},{keys:['w']}]},{sequence:[],keys:['w']},{keys:['Control','l']},{keys:['F12']},{text:'a',keys:['w']}])assert.throws(()=>expandInputs('keyboard',args));
  assert.throws(()=>expandInputs('mouse',{operation:'click',x:5,y:-1}));
  assert.throws(()=>expandInputs('navigate',{keys:['w']}));
});
test('waits between primitive inputs and releases keys',async()=>{
  const events=[];const page={keyboard:{down:async key=>events.push('down '+key),up:async key=>events.push('up '+key)}};
  const result=await executeInputs(page,'keyboard',{keys:['w'],repeat:3},async()=>{events.push('settled');return 'Live';});
  assert.deepEqual(events,['down w','up w','settled','down w','up w','settled','down w','up w','settled']);
  assert.deepEqual(result,{ok:true,executed_inputs:3,requested_inputs:3});
});
test('stops batch at visible completion or failure',async()=>{
  const page={keyboard:{down:async()=>{},up:async()=>{}}};
  for(const status of ['Saved locally','Action failed']){
    const result=await executeInputs(page,'keyboard',{keys:['w'],repeat:4},async()=>status);
    assert.equal(result.executed_inputs,1);assert.equal(result.requested_inputs,4);
    assert.equal(result.ok,status==='Saved locally');
  }
});

test('normalizes every mouse position exactly once and preserves scroll distances',async()=>{
  const events=[];
  const page={mouse:Object.fromEntries(['move','click','down','up','wheel'].map(k=>[k,async(...args)=>events.push([k,...args])]))};
  const settle=async()=> 'Live';
  for(const operation of ['move','click','double_click','drag','scroll']){
    events.length=0;
    await executeInputs(page,'mouse',{operation,x:500,y:250,to_x:1000,to_y:1000,delta_x:123,delta_y:-456},settle);
    assert.deepEqual(events[0],['move',720,250]);
    if(operation.includes('click'))assert.deepEqual(events[1],['click',720,250,{button:'left',clickCount:operation==='double_click'?2:1}]);
    if(operation==='drag')assert.deepEqual(events.slice(1),[['down',{button:'left'}],['move',1439,999,{steps:12}],['up',{button:'left'}]]);
    if(operation==='scroll')assert.deepEqual(events[1],['wheel',{deltaX:123,deltaY:-456}]);
  }
  await executeInputs(page,'mouse',{operation:'move',x:0,y:0},settle);
  assert.deepEqual(events.at(-1),['move',0,0]);
});

test('normalized bounds apply to every axis and drag endpoint',()=>{
  for(const field of ['x','y','to_x','to_y'])for(const value of [-1,1000.1,1440,NaN,Infinity]){
    assert.throws(()=>expandInputs('mouse',{operation:'drag',x:1,y:2,to_x:3,to_y:4,[field]:value}),/0 and 1000/);
  }
  assert.equal(expandInputs('mouse',{operation:'click',x:1000,y:1000}).length,1);
});

test('text uses current-focus insertion without emitting shortcut key events',async()=>{
  const inserted=[];const page={keyboard:{sendCharacter:async text=>inserted.push(text)}};
  await executeInputs(page,'keyboard',{text:'right. Where should I go?'},async()=> 'Live');
  assert.deepEqual(inserted,['right. Where should I go?']);
});
