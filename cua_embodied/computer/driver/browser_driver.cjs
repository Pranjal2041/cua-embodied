// The model-facing boundary: two input operations, screenshot observations.
// No DOM, JavaScript execution, network, shell, or benchmark tools are exposed.
const puppeteer=require('puppeteer');
const readline=require('node:readline');
const {executeInputs,WIDTH,HEIGHT}=require('./browser_inputs.cjs');
const {PROTOCOL_TIMEOUT_MS,waitForReady,describeError}=require('./browser_readiness.cjs');
(async()=>{
  const url=process.argv[2];
  const parsed=new URL(url);
  if(!['127.0.0.1','localhost'].includes(parsed.hostname))throw new Error('Local GUI URL required');
  const browser=await puppeteer.launch({headless:true,protocolTimeout:PROTOCOL_TIMEOUT_MS,args:['--disable-dev-shm-usage','--disable-gpu'],defaultViewport:{width:WIDTH,height:HEIGHT}});
  const page=await browser.newPage();
  await page.goto(url,{waitUntil:'networkidle2'});
  // Observe only visible readiness signals; no task API or hidden state is used.
  const settle=()=>waitForReady(page);
  const reply=async(extra={})=>process.stdout.write(JSON.stringify({...extra,image:await page.screenshot({encoding:'base64'})})+'\n');
  await settle();
  await reply({ready:true});
  const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
  for await(const line of lines){
    try{
      const {name,arguments:a={}}=JSON.parse(line);
      if(name==='close')break;
      await reply(await executeInputs(page,name,a,settle));
    }catch(e){await reply({ok:false,error:describeError(e)});}
  }
  await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1;});
