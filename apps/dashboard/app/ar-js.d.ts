declare module '@ar-js-org/ar.js/three.js/build/ar-threex.mjs' {
  export class ArToolkitSource {
    constructor(options: { sourceType: 'webcam'; sourceWidth: number; sourceHeight: number });
    domElement: HTMLVideoElement | null;
    ready: boolean;
    init(onReady: () => void, onError?: (error: { name?: string; message?: string }) => void): void;
    onResizeElement(): void;
    copyElementSizeTo(element: HTMLElement): void;
    dispose(): void;
  }
  export class ArToolkitContext {
    static baseURL: string;
    constructor(options: { cameraParametersUrl: string; detectionMode: 'mono'; canvasWidth: number; canvasHeight: number });
    arController: { canvas: HTMLCanvasElement } | null;
    init(onReady: () => void): void;
    getProjectionMatrix(): import('three').Matrix4;
    update(source: HTMLVideoElement): void;
  }
  export class ArMarkerControls {
    constructor(context: ArToolkitContext, target: import('three').Object3D, options: {
      type: 'pattern'; patternUrl: string; size: number; smooth: boolean; smoothCount: number;
    });
  }
}
