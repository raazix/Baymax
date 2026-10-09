type Cv = any;
let openCvLoading: Promise<Cv> | null = null;

export function loadOpenCv(): Promise<Cv> {
  const existing = (window as any).cv;
  if (existing?.Mat) return Promise.resolve(existing);
  if (openCvLoading) return openCvLoading;

  openCvLoading = new Promise<Cv>((resolve, reject) => {
    let settled = false;
    let awaitingModule = false;
    let poll = 0;
    const scriptId = 'lineguard-opencv-runtime';
    const finish = (error?: Error, runtime?: Cv) => {
      if (settled) return;
      settled = true;
      window.clearInterval(poll);
      window.clearTimeout(timeout);
      document.removeEventListener('error', onError, true);
      if (error) document.getElementById(scriptId)?.remove();
      if (error) reject(error);
      else resolve(runtime ?? (window as any).cv);
    };
    const check = () => {
      const candidate = (window as any).cv;
      if (candidate?.Mat) finish(undefined, candidate);
      else if (candidate && typeof candidate.then === 'function' && !awaitingModule) {
        awaitingModule = true;
        Promise.resolve(candidate).then(runtime => {
          if (!runtime?.Mat) { finish(new Error('OpenCV resolved without its Mat API.')); return; }
          (window as any).cv = runtime;
          finish(undefined, runtime);
        }, error => finish(new Error(`OpenCV WebAssembly initialization failed: ${String(error)}`)));
      }
    };
    const timeout = window.setTimeout(() => {
      const candidate = (window as any).cv;
      finish(new Error(candidate
        ? `OpenCV.js loaded but its WebAssembly runtime did not become ready (calledRun=${Boolean(candidate.calledRun)}).`
        : 'OpenCV.js loaded without exposing its browser runtime.'));
    }, 60000);

    const onError = (event: Event) => {
      const target = event.target;
      if (target instanceof HTMLScriptElement && target.id === scriptId) finish(new Error('Could not load the OpenCV.js runtime script.'));
    };
    document.addEventListener('error', onError, true);
    const script = document.getElementById(scriptId) as HTMLScriptElement | null ?? document.createElement('script');
    script.id = scriptId;
    script.src = '/api/opencvjs';
    script.async = true;
    script.onload = check;
    script.onerror = () => finish(new Error('Could not load the OpenCV.js runtime script.'));
    poll = window.setInterval(check, 100);
    if (!script.isConnected) document.head.appendChild(script);
  }).catch(error => {
    openCvLoading = null;
    throw error;
  });
  return openCvLoading;
}
