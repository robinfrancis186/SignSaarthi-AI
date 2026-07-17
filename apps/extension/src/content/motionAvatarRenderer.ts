export type MotionPoint = {
  x: number;
  y: number;
  visibility?: number;
};

export type MotionFrame = {
  pose: MotionPoint[];
  leftHand: MotionPoint[];
  rightHand: MotionPoint[];
  face?: MotionPoint[];
};

export type MotionClip = {
  id: string;
  label: string;
  fps: number;
  frames: MotionFrame[];
};

export type MotionAvatarIdentity = "Ananya" | "Arjun" | "Meera" | "Kabir";

export type MotionClipLoadOptions = {
  transitionFrames?: number;
};

export type MotionAvatarRendererOptions = {
  avatarName?: MotionAvatarIdentity;
  onFrame?: (index: number) => void;
  onComplete?: () => void;
};

export interface MotionAvatarController {
  loadClip(clip: MotionClip, options?: MotionClipLoadOptions): void;
  play(): void;
  pause(): void;
  replay(): void;
  setSpeed(speed: number): void;
  destroy(): void;
  isPlaying(): boolean;
  getFrameIndex(): number;
}

type RenderPoint = Required<MotionPoint>;

type CoordinateMapper = (point: MotionPoint) => RenderPoint;

type ScheduledFrame =
  | { kind: "animation-frame"; id: number }
  | { kind: "timeout"; id: number };

type ArmElements = {
  underlay: SVGPathElement;
  upperArm: SVGPathElement;
  forearm: SVGPathElement;
  elbow: SVGCircleElement;
  cuff: SVGCircleElement;
};

type HandBoneElements = {
  underlay: SVGLineElement;
  line: SVGLineElement;
};

type HandElements = {
  palm: SVGPathElement;
  bones: HandBoneElements[];
  joints: SVGCircleElement[];
};

type AvatarElements = {
  svg: SVGSVGElement;
  title: SVGTitleElement;
  description: SVGDescElement;
  leftArm: ArmElements;
  rightArm: ArmElements;
  neck: SVGPathElement;
  torsoShadow: SVGPathElement;
  torso: SVGPathElement;
  collar: SVGPathElement;
  torsoAccent: SVGPathElement;
  leftEar: SVGCircleElement;
  rightEar: SVGCircleElement;
  face: SVGEllipseElement;
  hair: SVGPathElement;
  leftBrow: SVGPathElement;
  rightBrow: SVGPathElement;
  leftEye: SVGCircleElement;
  rightEye: SVGCircleElement;
  nose: SVGPathElement;
  mouth: SVGPathElement;
  faceHighlight: SVGPathElement;
  leftHand: HandElements;
  rightHand: HandElements;
  poseMarkers: SVGCircleElement[];
};

type AvatarPalette = {
  skin: readonly [string, string, string];
  shirt: readonly [string, string, string];
  hair: readonly [string, string];
};

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const VIEWBOX_WIDTH = 320;
const VIEWBOX_HEIGHT = 360;
const DEFAULT_FPS = 30;
const MIN_VISIBILITY = 0.04;
const MIN_SPEED = 0.1;
const MAX_SPEED = 8;
const MAX_POSE_POINTS = 33;
const MAX_HAND_POINTS = 21;
const MAX_FACE_POINTS = 18;

const FACE = {
  leftBrowStart: 0,
  leftBrowEnd: 4,
  rightBrowStart: 5,
  rightBrowEnd: 9,
  leftEyeTop: 10,
  leftEyeBottom: 11,
  rightEyeTop: 12,
  rightEyeBottom: 13,
  mouthLeft: 14,
  mouthRight: 15,
  mouthTop: 16,
  mouthBottom: 17
} as const;

const AVATAR_PALETTES: Record<MotionAvatarIdentity, AvatarPalette> = {
  Ananya: {
    skin: ["#fff1df", "#efb995", "#c77f68"],
    shirt: ["#20c7bd", "#167f91", "#163e58"],
    hair: ["#26394c", "#0a1522"]
  },
  Arjun: {
    skin: ["#e8b28d", "#bb795c", "#78443d"],
    shirt: ["#5f86ff", "#294d9b", "#16264c"],
    hair: ["#202832", "#080d13"]
  },
  Meera: {
    skin: ["#f4c7a6", "#d99072", "#9e594c"],
    shirt: ["#df6688", "#8f2f58", "#431c38"],
    hair: ["#34252b", "#100b0e"]
  },
  Kabir: {
    skin: ["#dba37c", "#a9654d", "#673a33"],
    shirt: ["#8ac66f", "#34714f", "#163d38"],
    hair: ["#292a2a", "#0b0c0c"]
  }
};

const POSE = {
  nose: 0,
  leftEye: 2,
  rightEye: 5,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24
} as const;

// MediaPipe Hands topology. Every connection is retained in the SVG and shown
// whenever both endpoint landmarks are available.
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20]
];

const FINGERTIP_INDICES = new Set([4, 8, 12, 16, 20]);
const PALM_INDICES = [0, 5, 9, 13, 17] as const;
const IDLE_FRAME = createIdleFrame();
const NORMALIZED_MAPPER: CoordinateMapper = (point) => ({
  x: point.x * VIEWBOX_WIDTH,
  y: point.y * VIEWBOX_HEIGHT,
  visibility: visibilityOf(point)
});

let rendererInstance = 0;

export function createMotionAvatarRenderer(
  container: HTMLElement,
  options: MotionAvatarRendererOptions = {}
): MotionAvatarController {
  const instanceId = `signsaarthi-motion-avatar-${++rendererInstance}`;
  const elements = createAvatarElements(instanceId, options.avatarName ?? "Ananya");
  const scheduler = createAnimationScheduler();

  let clip: MotionClip | null = null;
  let mapper = NORMALIZED_MAPPER;
  let elapsedMs = 0;
  let frameIndex = 0;
  let lastNotifiedFrame = -1;
  let lastTimestamp: number | null = null;
  let scheduledFrame: ScheduledFrame | null = null;
  let speed = 1;
  let playing = false;
  let destroyed = false;
  let lastRenderedFrame: MotionFrame | null = null;

  container.append(elements.svg);
  renderAvatar(elements, IDLE_FRAME, NORMALIZED_MAPPER, true);

  function notifyFrame(index: number, force = false): void {
    if (!force && lastNotifiedFrame === index) {
      return;
    }

    lastNotifiedFrame = index;
    options.onFrame?.(index);
  }

  function stopScheduling(): void {
    if (scheduledFrame) {
      scheduler.cancel(scheduledFrame);
      scheduledFrame = null;
    }
    lastTimestamp = null;
  }

  function scheduleNextFrame(): void {
    scheduledFrame = scheduler.request(tick);
  }

  function tick(timestamp: number): void {
    scheduledFrame = null;
    if (!playing || destroyed || !clip) {
      return;
    }

    if (lastTimestamp === null) {
      lastTimestamp = timestamp;
    } else {
      const deltaMs = Math.max(0, timestamp - lastTimestamp);
      lastTimestamp = timestamp;
      elapsedMs += deltaMs * speed;
    }

    const finished = renderTimeline();
    if (finished) {
      playing = false;
      lastTimestamp = null;
      elements.svg.dataset.state = "complete";
      options.onComplete?.();
      return;
    }

    scheduleNextFrame();
  }

  function renderTimeline(forceNotification = false): boolean {
    if (!clip || clip.frames.length === 0) {
      frameIndex = 0;
      return true;
    }

    const finalIndex = clip.frames.length - 1;
    const framePosition = Math.min((elapsedMs * clip.fps) / 1000, finalIndex);
    const lowerIndex = Math.min(Math.floor(framePosition), finalIndex);
    const upperIndex = Math.min(lowerIndex + 1, finalIndex);
    const interpolation = upperIndex === lowerIndex ? 0 : framePosition - lowerIndex;
    const lowerFrame = clip.frames[lowerIndex] ?? emptyFrame();
    const upperFrame = clip.frames[upperIndex] ?? lowerFrame;
    const displayFrame =
      interpolation > 0 ? interpolateFrame(lowerFrame, upperFrame, interpolation) : lowerFrame;

    frameIndex = lowerIndex;
    lastRenderedFrame = displayFrame;
    elements.svg.dataset.frameIndex = String(frameIndex);
    elements.svg.dataset.frameProgress = interpolation.toFixed(3);
    renderAvatar(elements, displayFrame, mapper, false);
    notifyFrame(frameIndex, forceNotification);

    return framePosition >= finalIndex;
  }

  const controller: MotionAvatarController = {
    loadClip(nextClip, loadOptions = {}) {
      if (destroyed) {
        return;
      }

      playing = false;
      stopScheduling();
      const sanitizedClip = sanitizeClip(nextClip);
      const transitionFrames = Math.max(0, Math.min(12, Math.round(loadOptions.transitionFrames ?? 0)));
      clip = withTransitionFrames(sanitizedClip, lastRenderedFrame, transitionFrames);
      mapper = createCoordinateMapper(clip.frames);
      elapsedMs = 0;
      frameIndex = 0;
      lastNotifiedFrame = -1;

      if (clip.frames.length === 0) {
        elements.title.textContent = "SignSaarthi motion avatar at rest";
        elements.description.textContent =
          "A neutral upper-body avatar ready to render pose and hand motion.";
        elements.svg.dataset.state = "idle";
        elements.svg.dataset.clipId = clip.id;
        delete elements.svg.dataset.transitionFrames;
        lastRenderedFrame = IDLE_FRAME;
        renderAvatar(elements, IDLE_FRAME, NORMALIZED_MAPPER, true);
        return;
      }

      elements.title.textContent = `SignSaarthi avatar signing ${clip.label}`;
      elements.description.textContent =
        "An animated upper-body avatar driven by interpolated pose and hand landmarks.";
      elements.svg.dataset.state = "paused";
      elements.svg.dataset.clipId = clip.id;
      elements.svg.dataset.transitionFrames = String(transitionFrames);
      renderTimeline(true);
    },

    play() {
      if (destroyed || playing || !clip || clip.frames.length < 2) {
        return;
      }

      const durationMs = ((clip.frames.length - 1) / clip.fps) * 1000;
      if (elapsedMs >= durationMs) {
        return;
      }

      playing = true;
      lastTimestamp = null;
      elements.svg.dataset.state = "playing";
      scheduleNextFrame();
    },

    pause() {
      if (destroyed) {
        return;
      }

      playing = false;
      stopScheduling();
      if (clip?.frames.length) {
        elements.svg.dataset.state = "paused";
      }
    },

    replay() {
      if (destroyed || !clip || clip.frames.length === 0) {
        return;
      }

      playing = false;
      stopScheduling();
      elapsedMs = 0;
      frameIndex = 0;
      renderTimeline(true);

      if (clip.frames.length > 1) {
        playing = true;
        elements.svg.dataset.state = "playing";
        scheduleNextFrame();
      }
    },

    setSpeed(nextSpeed) {
      if (!Number.isFinite(nextSpeed) || nextSpeed <= 0) {
        return;
      }
      speed = clamp(nextSpeed, MIN_SPEED, MAX_SPEED);
    },

    destroy() {
      if (destroyed) {
        return;
      }

      destroyed = true;
      playing = false;
      stopScheduling();
      elements.svg.remove();
      clip = null;
      lastRenderedFrame = null;
    },

    isPlaying() {
      return playing;
    },

    getFrameIndex() {
      return frameIndex;
    }
  };

  return controller;
}

function createAvatarElements(instanceId: string, avatarName: MotionAvatarIdentity): AvatarElements {
  const palette = AVATAR_PALETTES[avatarName];
  const skinGradientId = `${instanceId}-skin`;
  const shirtGradientId = `${instanceId}-shirt`;
  const hairGradientId = `${instanceId}-hair`;

  const svg = createSvgElement("svg", {
    viewBox: `0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`,
    width: String(VIEWBOX_WIDTH),
    height: String(VIEWBOX_HEIGHT),
    role: "img",
    focusable: "false",
    preserveAspectRatio: "xMidYMid meet",
    "data-motion-avatar": "",
    "data-avatar-name": avatarName,
    "data-state": "idle",
    "aria-labelledby": `${instanceId}-title ${instanceId}-description`,
    style:
      "display:block;width:100%;height:100%;min-width:0;min-height:0;overflow:visible;isolation:isolate"
  });

  const title = createSvgElement("title", { id: `${instanceId}-title` });
  title.textContent = "SignSaarthi motion avatar at rest";
  const description = createSvgElement("desc", { id: `${instanceId}-description` });
  description.textContent = "A neutral upper-body avatar ready to render pose and hand motion.";
  svg.append(title, description);

  const defs = createSvgElement("defs");
  appendLinearGradient(defs, skinGradientId, "15%", "0%", "85%", "100%", [
    ["0%", palette.skin[0]],
    ["52%", palette.skin[1]],
    ["100%", palette.skin[2]]
  ]);
  appendLinearGradient(defs, shirtGradientId, "12%", "0%", "88%", "100%", [
    ["0%", palette.shirt[0]],
    ["50%", palette.shirt[1]],
    ["100%", palette.shirt[2]]
  ]);
  appendLinearGradient(defs, hairGradientId, "0%", "0%", "100%", "100%", [
    ["0%", palette.hair[0]],
    ["100%", palette.hair[1]]
  ]);
  svg.append(defs);

  const floorShadow = createSvgElement("ellipse", {
    cx: "160",
    cy: "338",
    rx: "76",
    ry: "8",
    fill: "#07131f",
    opacity: "0.38",
    "aria-hidden": "true"
  });
  svg.append(floorShadow);

  const armsLayer = createSvgElement("g", { "aria-hidden": "true" });
  const leftArm = createArmElements("left", `url(#${shirtGradientId})`, skinGradientId);
  const rightArm = createArmElements("right", `url(#${shirtGradientId})`, skinGradientId);
  armsLayer.append(
    leftArm.underlay,
    rightArm.underlay,
    leftArm.upperArm,
    rightArm.upperArm,
    leftArm.forearm,
    rightArm.forearm,
    leftArm.elbow,
    rightArm.elbow,
    leftArm.cuff,
    rightArm.cuff
  );
  svg.append(armsLayer);

  const neck = createSvgElement("path", {
    fill: `url(#${skinGradientId})`,
    stroke: "#9f6258",
    "stroke-width": "1.5",
    "stroke-linejoin": "round",
    "data-avatar-part": "neck",
    "aria-hidden": "true"
  });
  svg.append(neck);

  const torsoLayer = createSvgElement("g", { "aria-hidden": "true" });
  const torsoShadow = createSvgElement("path", {
    fill: "#06131f",
    opacity: "0.52",
    transform: "translate(0 6)",
    "data-avatar-part": "torso-shadow"
  });
  const torso = createSvgElement("path", {
    fill: `url(#${shirtGradientId})`,
    stroke: "#55e6d7",
    "stroke-opacity": "0.38",
    "stroke-width": "1.5",
    "stroke-linejoin": "round",
    "data-avatar-part": "torso"
  });
  const collar = createSvgElement("path", {
    fill: "none",
    stroke: "#b8fff3",
    "stroke-opacity": "0.76",
    "stroke-width": "3",
    "stroke-linecap": "round",
    "data-avatar-part": "collar"
  });
  const torsoAccent = createSvgElement("path", {
    fill: "none",
    stroke: "#f6c85f",
    "stroke-opacity": "0.7",
    "stroke-width": "2",
    "stroke-linecap": "round",
    "data-avatar-part": "torso-accent"
  });
  torsoLayer.append(torsoShadow, torso, collar, torsoAccent);
  svg.append(torsoLayer);

  const headLayer = createSvgElement("g", { "aria-hidden": "true" });
  const leftEar = createSvgElement("circle", {
    fill: `url(#${skinGradientId})`,
    stroke: "#9f6258",
    "stroke-width": "1.2"
  });
  const rightEar = leftEar.cloneNode(false) as SVGCircleElement;
  const face = createSvgElement("ellipse", {
    fill: `url(#${skinGradientId})`,
    stroke: "#9f6258",
    "stroke-width": "1.6",
    "data-avatar-part": "head"
  });
  const faceHighlight = createSvgElement("path", {
    fill: "none",
    stroke: "#fff9ef",
    "stroke-opacity": "0.48",
    "stroke-width": "2.2",
    "stroke-linecap": "round"
  });
  const hair = createSvgElement("path", {
    fill: `url(#${hairGradientId})`,
    stroke: "#3b5268",
    "stroke-width": "1.2",
    "stroke-linejoin": "round",
    "data-avatar-part": "hair"
  });
  const leftBrow = createSvgElement("path", {
    fill: "none",
    stroke: "#51372f",
    "stroke-width": "2.3",
    "stroke-linecap": "round",
    "data-avatar-part": "left-brow"
  });
  const rightBrow = createSvgElement("path", {
    fill: "none",
    stroke: "#51372f",
    "stroke-width": "2.3",
    "stroke-linecap": "round",
    "data-avatar-part": "right-brow"
  });
  const leftEye = createSvgElement("circle", { r: "2.5", fill: "#102231" });
  const rightEye = leftEye.cloneNode(false) as SVGCircleElement;
  const nose = createSvgElement("path", {
    fill: "none",
    stroke: "#a6665d",
    "stroke-width": "1.6",
    "stroke-linecap": "round",
    "stroke-linejoin": "round"
  });
  const mouth = createSvgElement("path", {
    fill: "none",
    stroke: "#8f4f55",
    "stroke-width": "2",
    "stroke-linecap": "round",
    "data-avatar-part": "mouth"
  });
  headLayer.append(
    leftEar,
    rightEar,
    face,
    faceHighlight,
    hair,
    leftBrow,
    rightBrow,
    leftEye,
    rightEye,
    nose,
    mouth
  );
  svg.append(headLayer);

  const handsLayer = createSvgElement("g", { "aria-hidden": "true" });
  const leftHand = createHandElements("left");
  const rightHand = createHandElements("right");
  appendHandElements(handsLayer, leftHand);
  appendHandElements(handsLayer, rightHand);
  svg.append(handsLayer);

  const poseMarkerLayer = createSvgElement("g", {
    opacity: "0",
    "pointer-events": "none",
    "aria-hidden": "true"
  });
  const poseMarkers = Array.from({ length: MAX_POSE_POINTS }, (_, index) => {
    const marker = createSvgElement("circle", {
      r: "0.01",
      visibility: "hidden",
      "data-motion-landmark": `pose-${index}`
    });
    poseMarkerLayer.append(marker);
    return marker;
  });
  svg.append(poseMarkerLayer);

  return {
    svg,
    title,
    description,
    leftArm,
    rightArm,
    neck,
    torsoShadow,
    torso,
    collar,
    torsoAccent,
    leftEar,
    rightEar,
    face,
    hair,
    leftBrow,
    rightBrow,
    leftEye,
    rightEye,
    nose,
    mouth,
    faceHighlight,
    leftHand,
    rightHand,
    poseMarkers
  };
}

function createArmElements(
  side: "left" | "right",
  shirtFill: string,
  skinGradientId: string
): ArmElements {
  const sharedPathAttributes = {
    fill: "none",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "aria-hidden": "true"
  };
  const underlay = createSvgElement("path", {
    ...sharedPathAttributes,
    stroke: "#06131f",
    "stroke-width": "22",
    opacity: "0.72",
    "data-avatar-part": `${side}-arm-underlay`
  });
  const upperArm = createSvgElement("path", {
    ...sharedPathAttributes,
    stroke: shirtFill,
    "stroke-width": "17",
    "data-avatar-part": `${side}-upper-arm`
  });
  const forearm = createSvgElement("path", {
    ...sharedPathAttributes,
    stroke: `url(#${skinGradientId})`,
    "stroke-width": "12",
    "data-avatar-part": `${side}-forearm`
  });
  const elbow = createSvgElement("circle", {
    r: "6.2",
    fill: `url(#${skinGradientId})`,
    stroke: "#a7665c",
    "stroke-width": "1",
    "data-avatar-part": `${side}-elbow`
  });
  const cuff = createSvgElement("circle", {
    r: "6.8",
    fill: side === "left" ? "#6ce7d8" : "#f6c85f",
    stroke: "#071a27",
    "stroke-width": "2.5",
    "data-avatar-part": `${side}-wrist`
  });

  return { underlay, upperArm, forearm, elbow, cuff };
}

function createHandElements(side: "left" | "right"): HandElements {
  const palm = createSvgElement("path", {
    fill: side === "left" ? "#55e6d7" : "#f6c85f",
    "fill-opacity": "0.14",
    stroke: "#ffd9bd",
    "stroke-opacity": "0.62",
    "stroke-width": "1.2",
    "stroke-linejoin": "round",
    visibility: "hidden",
    "data-avatar-part": `${side}-palm`
  });

  const bones = HAND_CONNECTIONS.map(([from, to]) => {
    const underlay = createSvgElement("line", {
      stroke: "#07131e",
      "stroke-width": "6.2",
      "stroke-linecap": "round",
      opacity: "0.7",
      visibility: "hidden"
    });
    const line = createSvgElement("line", {
      stroke: side === "left" ? "#a8fff2" : "#ffe1ad",
      "stroke-width": "3.3",
      "stroke-linecap": "round",
      visibility: "hidden",
      "data-hand": side,
      "data-bone": `${from}-${to}`
    });
    return { underlay, line };
  });

  const joints = Array.from({ length: MAX_HAND_POINTS }, (_, index) =>
    createSvgElement("circle", {
      r: index === 0 ? "4.2" : FINGERTIP_INDICES.has(index) ? "3.2" : "2.35",
      fill: index === 0 ? "#f4ba94" : "#fff5e8",
      stroke: side === "left" ? "#238f96" : "#bd8450",
      "stroke-width": "1.1",
      visibility: "hidden",
      "data-hand": side,
      "data-landmark": String(index),
      "data-motion-landmark": `${side}-hand-${index}`
    })
  );

  return { palm, bones, joints };
}

function appendHandElements(parent: SVGGElement, hand: HandElements): void {
  parent.append(hand.palm);
  for (const bone of hand.bones) {
    parent.append(bone.underlay, bone.line);
  }
  parent.append(...hand.joints);
}

function renderAvatar(
  elements: AvatarElements,
  frame: MotionFrame,
  mapper: CoordinateMapper,
  isIdle: boolean
): void {
  const leftHandPoints = mapSeries(frame.leftHand, mapper, MAX_HAND_POINTS);
  const rightHandPoints = mapSeries(frame.rightHand, mapper, MAX_HAND_POINTS);

  const leftShoulder = resolvePosePoint(frame.pose, POSE.leftShoulder, mapper);
  const rightShoulder = resolvePosePoint(frame.pose, POSE.rightShoulder, mapper);
  const leftHandWrist = visibleRenderPoint(leftHandPoints[0]);
  const rightHandWrist = visibleRenderPoint(rightHandPoints[0]);
  const leftPoseWrist = resolvePosePoint(frame.pose, POSE.leftWrist, mapper);
  const rightPoseWrist = resolvePosePoint(frame.pose, POSE.rightWrist, mapper);
  const leftWrist = leftHandWrist ?? leftPoseWrist.point;
  const rightWrist = rightHandWrist ?? rightPoseWrist.point;
  const leftElbow = resolveElbow(
    frame.pose,
    POSE.leftElbow,
    mapper,
    leftShoulder.point,
    leftWrist,
    -1
  );
  const rightElbow = resolveElbow(
    frame.pose,
    POSE.rightElbow,
    mapper,
    rightShoulder.point,
    rightWrist,
    1
  );

  renderArm(elements.leftArm, leftShoulder.point, leftElbow, leftWrist);
  renderArm(elements.rightArm, rightShoulder.point, rightElbow, rightWrist);

  const leftHip = resolveHip(
    frame.pose,
    POSE.leftHip,
    mapper,
    leftShoulder.point,
    1
  );
  const rightHip = resolveHip(
    frame.pose,
    POSE.rightHip,
    mapper,
    rightShoulder.point,
    -1
  );
  renderTorso(elements, leftShoulder.point, rightShoulder.point, leftHip, rightHip);
  renderHead(
    elements,
    frame.pose,
    frame.face ?? [],
    mapper,
    leftShoulder.point,
    rightShoulder.point
  );
  renderHand(elements.leftHand, leftHandPoints, isIdle);
  renderHand(elements.rightHand, rightHandPoints, isIdle);
  renderPoseMarkers(elements.poseMarkers, frame.pose, mapper);
}

function renderArm(
  elements: ArmElements,
  shoulder: RenderPoint,
  elbow: RenderPoint,
  wrist: RenderPoint
): void {
  const fullPath = `M ${format(shoulder.x)} ${format(shoulder.y)} L ${format(elbow.x)} ${format(
    elbow.y
  )} L ${format(wrist.x)} ${format(wrist.y)}`;
  setAttributes(elements.underlay, { d: fullPath });
  setAttributes(elements.upperArm, {
    d: `M ${format(shoulder.x)} ${format(shoulder.y)} L ${format(elbow.x)} ${format(elbow.y)}`
  });
  setAttributes(elements.forearm, {
    d: `M ${format(elbow.x)} ${format(elbow.y)} L ${format(wrist.x)} ${format(wrist.y)}`
  });
  setCirclePosition(elements.elbow, elbow);
  setCirclePosition(elements.cuff, wrist);
}

function renderTorso(
  elements: AvatarElements,
  leftShoulder: RenderPoint,
  rightShoulder: RenderPoint,
  leftHip: RenderPoint,
  rightHip: RenderPoint
): void {
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  const hipMid = midpoint(leftHip, rightHip);
  const torsoPath = [
    `M ${format(leftShoulder.x)} ${format(leftShoulder.y)}`,
    `Q ${format(shoulderMid.x)} ${format(shoulderMid.y - 10)} ${format(rightShoulder.x)} ${format(
      rightShoulder.y
    )}`,
    `L ${format(rightHip.x)} ${format(rightHip.y)}`,
    `Q ${format(hipMid.x)} ${format(hipMid.y + 9)} ${format(leftHip.x)} ${format(leftHip.y)}`,
    "Z"
  ].join(" ");

  setAttributes(elements.torso, { d: torsoPath });
  setAttributes(elements.torsoShadow, { d: torsoPath });

  const collarLeft = lerpPoint(leftShoulder, shoulderMid, 0.68);
  const collarRight = lerpPoint(rightShoulder, shoulderMid, 0.68);
  const collarDrop = Math.max(12, distance(leftShoulder, rightShoulder) * 0.11);
  setAttributes(elements.collar, {
    d: `M ${format(collarLeft.x)} ${format(collarLeft.y + 1)} Q ${format(shoulderMid.x)} ${format(
      shoulderMid.y + collarDrop
    )} ${format(collarRight.x)} ${format(collarRight.y + 1)}`
  });
  setAttributes(elements.torsoAccent, {
    d: `M ${format(shoulderMid.x)} ${format(shoulderMid.y + collarDrop + 10)} L ${format(
      shoulderMid.x
    )} ${format(Math.min(hipMid.y - 18, shoulderMid.y + 58))}`
  });
}

function renderHead(
  elements: AvatarElements,
  pose: MotionPoint[],
  facePoints: MotionPoint[],
  mapper: CoordinateMapper,
  leftShoulder: RenderPoint,
  rightShoulder: RenderPoint
): void {
  const nose = resolvePosePoint(pose, POSE.nose, mapper);
  const leftEar = resolvePosePoint(pose, POSE.leftEar, mapper);
  const rightEar = resolvePosePoint(pose, POSE.rightEar, mapper);
  const shoulderWidth = distance(leftShoulder, rightShoulder);
  const measuredEarWidth =
    leftEar.isObserved && rightEar.isObserved ? distance(leftEar.point, rightEar.point) : 0;
  const headRadiusX = clamp(
    measuredEarWidth > 0 ? measuredEarWidth * 0.67 : shoulderWidth * 0.2,
    25,
    39
  );
  const headRadiusY = headRadiusX * 1.2;

  let center: RenderPoint;
  if (nose.isObserved && leftEar.isObserved && rightEar.isObserved) {
    const earCenter = midpoint(leftEar.point, rightEar.point);
    center = {
      x: lerp(earCenter.x, nose.point.x, 0.42),
      y: lerp(earCenter.y, nose.point.y, 0.3) + headRadiusY * 0.05,
      visibility: Math.min(nose.point.visibility, leftEar.point.visibility, rightEar.point.visibility)
    };
  } else if (nose.isObserved) {
    center = { ...nose.point, y: nose.point.y + headRadiusY * 0.06 };
  } else if (leftEar.isObserved && rightEar.isObserved) {
    center = midpoint(leftEar.point, rightEar.point);
  } else {
    center = resolvePosePoint(IDLE_FRAME.pose, POSE.nose, NORMALIZED_MAPPER).point;
  }

  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  const neckWidth = headRadiusX * 0.56;
  const neckTop = center.y + headRadiusY * 0.72;
  const neckBottom = Math.max(neckTop + 8, shoulderMid.y + 5);
  setAttributes(elements.neck, {
    d: [
      `M ${format(center.x - neckWidth / 2)} ${format(neckTop)}`,
      `L ${format(center.x - neckWidth * 0.62)} ${format(neckBottom)}`,
      `L ${format(center.x + neckWidth * 0.62)} ${format(neckBottom)}`,
      `L ${format(center.x + neckWidth / 2)} ${format(neckTop)}`,
      "Z"
    ].join(" ")
  });

  setAttributes(elements.face, {
    cx: format(center.x),
    cy: format(center.y),
    rx: format(headRadiusX),
    ry: format(headRadiusY)
  });
  setAttributes(elements.leftEar, {
    cx: format(center.x - headRadiusX * 0.96),
    cy: format(center.y + headRadiusY * 0.03),
    r: format(headRadiusX * 0.17)
  });
  setAttributes(elements.rightEar, {
    cx: format(center.x + headRadiusX * 0.96),
    cy: format(center.y + headRadiusY * 0.03),
    r: format(headRadiusX * 0.17)
  });

  setAttributes(elements.hair, {
    d: [
      `M ${format(center.x - headRadiusX * 0.94)} ${format(center.y - headRadiusY * 0.12)}`,
      `C ${format(center.x - headRadiusX * 0.82)} ${format(center.y - headRadiusY * 1.04)}`,
      `${format(center.x + headRadiusX * 0.78)} ${format(center.y - headRadiusY * 1.05)}`,
      `${format(center.x + headRadiusX * 0.95)} ${format(center.y - headRadiusY * 0.08)}`,
      `C ${format(center.x + headRadiusX * 0.46)} ${format(center.y - headRadiusY * 0.42)}`,
      `${format(center.x - headRadiusX * 0.32)} ${format(center.y - headRadiusY * 0.56)}`,
      `${format(center.x - headRadiusX * 0.94)} ${format(center.y - headRadiusY * 0.12)}`,
      "Z"
    ].join(" ")
  });

  const observedLeftEye = faceFeatureMidpoint(
    facePoints,
    FACE.leftEyeTop,
    FACE.leftEyeBottom,
    mapper
  ) ?? resolveObservedPoint(pose, POSE.leftEye, mapper);
  const observedRightEye = faceFeatureMidpoint(
    facePoints,
    FACE.rightEyeTop,
    FACE.rightEyeBottom,
    mapper
  ) ?? resolveObservedPoint(pose, POSE.rightEye, mapper);
  const leftEye = constrainFacePoint(
    observedLeftEye ?? {
      x: center.x - headRadiusX * 0.34,
      y: center.y - headRadiusY * 0.08,
      visibility: 1
    },
    center,
    headRadiusX,
    headRadiusY
  );
  const rightEye = constrainFacePoint(
    observedRightEye ?? {
      x: center.x + headRadiusX * 0.34,
      y: center.y - headRadiusY * 0.08,
      visibility: 1
    },
    center,
    headRadiusX,
    headRadiusY
  );
  setCirclePosition(elements.leftEye, leftEye);
  setCirclePosition(elements.rightEye, rightEye);
  setAttributes(elements.leftBrow, {
    d: renderBrowPath(
      facePoints,
      FACE.leftBrowStart,
      FACE.leftBrowEnd,
      mapper,
      center,
      headRadiusX,
      headRadiusY,
      leftEye,
      -1
    )
  });
  setAttributes(elements.rightBrow, {
    d: renderBrowPath(
      facePoints,
      FACE.rightBrowStart,
      FACE.rightBrowEnd,
      mapper,
      center,
      headRadiusX,
      headRadiusY,
      rightEye,
      1
    )
  });

  const noseX = constrain(nose.isObserved ? nose.point.x : center.x, center.x, headRadiusX * 0.26);
  const noseY = constrain(
    nose.isObserved ? nose.point.y : center.y + headRadiusY * 0.09,
    center.y + headRadiusY * 0.04,
    headRadiusY * 0.22
  );
  setAttributes(elements.nose, {
    d: `M ${format(noseX)} ${format(noseY - headRadiusY * 0.08)} L ${format(
      noseX - headRadiusX * 0.05
    )} ${format(noseY + headRadiusY * 0.11)} Q ${format(noseX)} ${format(
      noseY + headRadiusY * 0.15
    )} ${format(noseX + headRadiusX * 0.08)} ${format(noseY + headRadiusY * 0.1)}`
  });
  setAttributes(elements.mouth, {
    d: renderMouthPath(facePoints, mapper, center, headRadiusX, headRadiusY)
  });
  setAttributes(elements.faceHighlight, {
    d: `M ${format(center.x - headRadiusX * 0.62)} ${format(
      center.y - headRadiusY * 0.34
    )} Q ${format(center.x - headRadiusX * 0.78)} ${format(center.y)} ${format(
      center.x - headRadiusX * 0.56
    )} ${format(center.y + headRadiusY * 0.3)}`
  });
}

function renderHand(elements: HandElements, points: Array<RenderPoint | undefined>, isIdle: boolean): void {
  const palmPoints = PALM_INDICES.map((index) => visibleRenderPoint(points[index])).filter(
    (point): point is RenderPoint => point !== undefined
  );
  if (palmPoints.length >= 3) {
    const palmPath = palmPoints
      .map((point, index) => `${index === 0 ? "M" : "L"} ${format(point.x)} ${format(point.y)}`)
      .join(" ");
    setAttributes(elements.palm, {
      d: `${palmPath} Z`,
      visibility: "visible",
      opacity: isIdle ? "0.48" : "0.72"
    });
  } else {
    setAttributes(elements.palm, { visibility: "hidden" });
  }

  HAND_CONNECTIONS.forEach(([from, to], index) => {
    const start = visibleRenderPoint(points[from]);
    const end = visibleRenderPoint(points[to]);
    const bone = elements.bones[index];
    if (!bone || !start || !end) {
      if (bone) {
        setAttributes(bone.underlay, { visibility: "hidden" });
        setAttributes(bone.line, { visibility: "hidden" });
      }
      return;
    }

    const opacity = format(clamp(Math.min(start.visibility, end.visibility), 0.12, 1));
    const attributes = {
      x1: format(start.x),
      y1: format(start.y),
      x2: format(end.x),
      y2: format(end.y),
      visibility: "visible",
      opacity
    };
    setAttributes(bone.underlay, attributes);
    setAttributes(bone.line, attributes);
  });

  elements.joints.forEach((joint, index) => {
    const point = visibleRenderPoint(points[index]);
    if (!point) {
      setAttributes(joint, { visibility: "hidden" });
      return;
    }

    setCirclePosition(joint, point);
    setAttributes(joint, {
      visibility: "visible",
      opacity: format(clamp(point.visibility, 0.18, 1))
    });
  });
}

function renderPoseMarkers(
  markers: SVGCircleElement[],
  pose: MotionPoint[],
  mapper: CoordinateMapper
): void {
  markers.forEach((marker, index) => {
    const point = mapVisiblePoint(pose[index], mapper);
    if (!point) {
      setAttributes(marker, { visibility: "hidden", "data-visible": "false" });
      return;
    }

    setCirclePosition(marker, point);
    setAttributes(marker, { visibility: "visible", "data-visible": "true" });
  });
}

function resolvePosePoint(
  pose: MotionPoint[],
  index: number,
  mapper: CoordinateMapper
): { point: RenderPoint; isObserved: boolean } {
  const observed = mapVisiblePoint(pose[index], mapper);
  if (observed) {
    return { point: observed, isObserved: true };
  }

  const idle = IDLE_FRAME.pose[index];
  return {
    point: idle ? NORMALIZED_MAPPER(idle) : { x: VIEWBOX_WIDTH / 2, y: VIEWBOX_HEIGHT / 2, visibility: 1 },
    isObserved: false
  };
}

function resolveObservedPoint(
  points: MotionPoint[],
  index: number,
  mapper: CoordinateMapper
): RenderPoint | undefined {
  return mapVisiblePoint(points[index], mapper);
}

function resolveElbow(
  pose: MotionPoint[],
  index: number,
  mapper: CoordinateMapper,
  shoulder: RenderPoint,
  wrist: RenderPoint,
  direction: -1 | 1
): RenderPoint {
  const observed = mapVisiblePoint(pose[index], mapper);
  if (observed) {
    return observed;
  }

  const center = midpoint(shoulder, wrist);
  return {
    x: center.x + direction * Math.min(12, distance(shoulder, wrist) * 0.1),
    y: center.y + 4,
    visibility: Math.min(shoulder.visibility, wrist.visibility)
  };
}

function resolveHip(
  pose: MotionPoint[],
  index: number,
  mapper: CoordinateMapper,
  shoulder: RenderPoint,
  inwardDirection: -1 | 1
): RenderPoint {
  const observed = mapVisiblePoint(pose[index], mapper);
  if (observed) {
    return observed;
  }

  return {
    x: shoulder.x + inwardDirection * 18,
    y: Math.min(VIEWBOX_HEIGHT - 18, shoulder.y + 154),
    visibility: shoulder.visibility
  };
}

function constrainFacePoint(
  point: RenderPoint,
  center: RenderPoint,
  radiusX: number,
  radiusY: number
): RenderPoint {
  return {
    x: constrain(point.x, center.x, radiusX * 0.58),
    y: constrain(point.y, center.y - radiusY * 0.08, radiusY * 0.32),
    visibility: point.visibility
  };
}

function faceFeatureMidpoint(
  points: MotionPoint[],
  firstIndex: number,
  secondIndex: number,
  mapper: CoordinateMapper
): RenderPoint | undefined {
  const first = mapVisiblePoint(points[firstIndex], mapper);
  const second = mapVisiblePoint(points[secondIndex], mapper);
  return first && second ? midpoint(first, second) : first ?? second;
}

function renderBrowPath(
  points: MotionPoint[],
  startIndex: number,
  endIndex: number,
  mapper: CoordinateMapper,
  center: RenderPoint,
  radiusX: number,
  radiusY: number,
  eye: RenderPoint,
  direction: -1 | 1
): string {
  const observed = [] as RenderPoint[];
  for (let index = startIndex; index <= endIndex; index += 1) {
    const point = mapVisiblePoint(points[index], mapper);
    if (point) {
      observed.push(constrainFacePoint(point, center, radiusX, radiusY));
    }
  }

  if (observed.length >= 3) {
    const first = observed[0]!;
    const middle = observed[Math.floor(observed.length / 2)]!;
    const last = observed.at(-1)!;
    return `M ${format(first.x)} ${format(first.y)} Q ${format(middle.x)} ${format(
      middle.y
    )} ${format(last.x)} ${format(last.y)}`;
  }

  const inwardLift = direction < 0 ? 0.22 : 0.21;
  return `M ${format(eye.x - radiusX * 0.15)} ${format(
    eye.y - radiusY * 0.15
  )} Q ${format(eye.x)} ${format(eye.y - radiusY * inwardLift)} ${format(
    eye.x + radiusX * 0.15
  )} ${format(eye.y - radiusY * 0.14)}`;
}

function renderMouthPath(
  points: MotionPoint[],
  mapper: CoordinateMapper,
  center: RenderPoint,
  radiusX: number,
  radiusY: number
): string {
  const mouthPoint = (index: number): RenderPoint | undefined => {
    const point = mapVisiblePoint(points[index], mapper);
    if (!point) {
      return undefined;
    }
    return {
      x: constrain(point.x, center.x, radiusX * 0.48),
      y: constrain(point.y, center.y + radiusY * 0.38, radiusY * 0.28),
      visibility: point.visibility
    };
  };
  const left = mouthPoint(FACE.mouthLeft);
  const right = mouthPoint(FACE.mouthRight);
  const top = mouthPoint(FACE.mouthTop);
  const bottom = mouthPoint(FACE.mouthBottom);

  if (left && right && top && bottom) {
    return [
      `M ${format(left.x)} ${format(left.y)}`,
      `Q ${format(top.x)} ${format(top.y)} ${format(right.x)} ${format(right.y)}`,
      `Q ${format(bottom.x)} ${format(bottom.y)} ${format(left.x)} ${format(left.y)}`
    ].join(" ");
  }

  return `M ${format(center.x - radiusX * 0.22)} ${format(
    center.y + radiusY * 0.42
  )} Q ${format(center.x)} ${format(center.y + radiusY * 0.53)} ${format(
    center.x + radiusX * 0.22
  )} ${format(center.y + radiusY * 0.41)}`;
}

function interpolateFrame(from: MotionFrame, to: MotionFrame, amount: number): MotionFrame {
  return {
    pose: interpolateSeries(from.pose, to.pose, amount, MAX_POSE_POINTS),
    leftHand: interpolateSeries(from.leftHand, to.leftHand, amount, MAX_HAND_POINTS),
    rightHand: interpolateSeries(from.rightHand, to.rightHand, amount, MAX_HAND_POINTS),
    face: interpolateSeries(from.face ?? [], to.face ?? [], amount, MAX_FACE_POINTS)
  };
}

function withTransitionFrames(
  clip: MotionClip,
  previousFrame: MotionFrame | null,
  transitionFrameCount: number
): MotionClip {
  const firstFrame = clip.frames[0];
  if (!previousFrame || !firstFrame || transitionFrameCount === 0) {
    return clip;
  }

  const transitionFrames = Array.from({ length: transitionFrameCount }, (_, index) =>
    interpolateFrame(previousFrame, firstFrame, index / transitionFrameCount)
  );
  return {
    ...clip,
    frames: [...transitionFrames, ...clip.frames]
  };
}

function interpolateSeries(
  from: MotionPoint[],
  to: MotionPoint[],
  amount: number,
  limit: number
): MotionPoint[] {
  const length = Math.min(Math.max(from.length, to.length), limit);
  const result = new Array<MotionPoint>(length);

  for (let index = 0; index < length; index += 1) {
    const start = validMotionPoint(from[index]);
    const end = validMotionPoint(to[index]);
    if (start && end) {
      result[index] = {
        x: lerp(start.x, end.x, amount),
        y: lerp(start.y, end.y, amount),
        visibility: lerp(visibilityOf(start), visibilityOf(end), amount)
      };
    } else if (start) {
      result[index] = {
        ...start,
        visibility: visibilityOf(start) * (1 - amount)
      };
    } else if (end) {
      result[index] = {
        ...end,
        visibility: visibilityOf(end) * amount
      };
    }
  }

  return result;
}

function sanitizeClip(clip: MotionClip): MotionClip {
  const framesValue: unknown = clip?.frames;
  const frames = Array.isArray(framesValue) ? framesValue.map(sanitizeFrame) : [];
  const fps = Number.isFinite(clip?.fps) && clip.fps > 0 ? clamp(clip.fps, 1, 120) : DEFAULT_FPS;
  const id = typeof clip?.id === "string" ? clip.id : "motion-clip";
  const label = typeof clip?.label === "string" && clip.label.trim() ? clip.label.trim() : id;

  return { id, label, fps, frames };
}

function sanitizeFrame(frame: unknown): MotionFrame {
  if (!frame || typeof frame !== "object") {
    return emptyFrame();
  }

  const candidate = frame as Partial<MotionFrame>;
  return {
    pose: sanitizeSeries(candidate.pose, MAX_POSE_POINTS),
    leftHand: sanitizeSeries(candidate.leftHand, MAX_HAND_POINTS),
    rightHand: sanitizeSeries(candidate.rightHand, MAX_HAND_POINTS),
    face: sanitizeSeries(candidate.face, MAX_FACE_POINTS)
  };
}

function sanitizeSeries(value: unknown, limit: number): MotionPoint[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const result = new Array<MotionPoint>(Math.min(value.length, limit));
  for (let index = 0; index < result.length; index += 1) {
    const point = validMotionPoint(value[index]);
    if (point) {
      result[index] = { ...point, visibility: visibilityOf(point) };
    }
  }
  return result;
}

function createCoordinateMapper(frames: MotionFrame[]): CoordinateMapper {
  let pointCount = 0;
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const frame of frames) {
    for (const series of [frame.pose, frame.leftHand, frame.rightHand]) {
      for (const candidate of series) {
        const point = validMotionPoint(candidate);
        if (point && visibilityOf(point) >= MIN_VISIBILITY) {
          pointCount += 1;
          minX = Math.min(minX, point.x);
          maxX = Math.max(maxX, point.x);
          minY = Math.min(minY, point.y);
          maxY = Math.max(maxY, point.y);
        }
      }
    }
  }

  if (pointCount === 0) {
    return NORMALIZED_MAPPER;
  }

  const looksNormalized = minX >= -1 && maxX <= 2 && minY >= -1 && maxY <= 2;
  if (looksNormalized) {
    return NORMALIZED_MAPPER;
  }

  const fitsViewBox =
    minX >= -VIEWBOX_WIDTH * 0.25 &&
    maxX <= VIEWBOX_WIDTH * 1.25 &&
    minY >= -VIEWBOX_HEIGHT * 0.25 &&
    maxY <= VIEWBOX_HEIGHT * 1.25;
  if (fitsViewBox) {
    return (point) => ({ x: point.x, y: point.y, visibility: visibilityOf(point) });
  }

  const padding = 24;
  const sourceWidth = Math.max(1, maxX - minX);
  const sourceHeight = Math.max(1, maxY - minY);
  const scale = Math.min(
    (VIEWBOX_WIDTH - padding * 2) / sourceWidth,
    (VIEWBOX_HEIGHT - padding * 2) / sourceHeight
  );
  const offsetX = (VIEWBOX_WIDTH - sourceWidth * scale) / 2 - minX * scale;
  const offsetY = (VIEWBOX_HEIGHT - sourceHeight * scale) / 2 - minY * scale;

  return (point) => ({
    x: point.x * scale + offsetX,
    y: point.y * scale + offsetY,
    visibility: visibilityOf(point)
  });
}

function createAnimationScheduler(): {
  request(callback: FrameRequestCallback): ScheduledFrame;
  cancel(frame: ScheduledFrame): void;
} {
  const hasAnimationFrame = typeof window.requestAnimationFrame === "function";
  return {
    request(callback) {
      if (hasAnimationFrame) {
        return { kind: "animation-frame", id: window.requestAnimationFrame(callback) };
      }

      const id = window.setTimeout(() => callback(window.performance.now()), 16);
      return { kind: "timeout", id };
    },
    cancel(frame) {
      if (frame.kind === "animation-frame") {
        window.cancelAnimationFrame(frame.id);
      } else {
        window.clearTimeout(frame.id);
      }
    }
  };
}

function createIdleFrame(): MotionFrame {
  const pose = new Array<MotionPoint>(MAX_POSE_POINTS);
  pose[POSE.nose] = { x: 0.5, y: 0.2, visibility: 1 };
  pose[POSE.leftEye] = { x: 0.475, y: 0.19, visibility: 1 };
  pose[POSE.rightEye] = { x: 0.525, y: 0.19, visibility: 1 };
  pose[POSE.leftEar] = { x: 0.44, y: 0.21, visibility: 1 };
  pose[POSE.rightEar] = { x: 0.56, y: 0.21, visibility: 1 };
  pose[POSE.leftShoulder] = { x: 0.35, y: 0.35, visibility: 1 };
  pose[POSE.rightShoulder] = { x: 0.65, y: 0.35, visibility: 1 };
  pose[POSE.leftElbow] = { x: 0.29, y: 0.5, visibility: 1 };
  pose[POSE.rightElbow] = { x: 0.71, y: 0.5, visibility: 1 };
  pose[POSE.leftWrist] = { x: 0.32, y: 0.64, visibility: 1 };
  pose[POSE.rightWrist] = { x: 0.68, y: 0.64, visibility: 1 };
  pose[POSE.leftHip] = { x: 0.4, y: 0.79, visibility: 1 };
  pose[POSE.rightHip] = { x: 0.6, y: 0.79, visibility: 1 };

  return {
    pose,
    leftHand: createIdleHand(0.32, 0.64, -1),
    rightHand: createIdleHand(0.68, 0.64, 1),
    face: []
  };
}

function createIdleHand(wristX: number, wristY: number, direction: -1 | 1): MotionPoint[] {
  const point = (x: number, y: number): MotionPoint => ({
    x: wristX + x * direction,
    y: wristY + y,
    visibility: 0.82
  });

  return [
    point(0, 0),
    point(0.018, 0.025),
    point(0.029, 0.052),
    point(0.038, 0.078),
    point(0.047, 0.099),
    point(-0.002, 0.045),
    point(-0.004, 0.082),
    point(-0.006, 0.116),
    point(-0.008, 0.145),
    point(-0.016, 0.044),
    point(-0.02, 0.086),
    point(-0.022, 0.124),
    point(-0.023, 0.158),
    point(-0.029, 0.042),
    point(-0.036, 0.081),
    point(-0.039, 0.115),
    point(-0.042, 0.143),
    point(-0.041, 0.035),
    point(-0.051, 0.067),
    point(-0.056, 0.094),
    point(-0.061, 0.118)
  ];
}

function mapSeries(
  series: MotionPoint[],
  mapper: CoordinateMapper,
  limit: number
): Array<RenderPoint | undefined> {
  const result = new Array<RenderPoint | undefined>(Math.min(series.length, limit));
  for (let index = 0; index < result.length; index += 1) {
    const point = validMotionPoint(series[index]);
    if (point) {
      result[index] = mapper(point);
    }
  }
  return result;
}

function mapVisiblePoint(
  point: MotionPoint | undefined,
  mapper: CoordinateMapper
): RenderPoint | undefined {
  const valid = validMotionPoint(point);
  if (!valid || visibilityOf(valid) < MIN_VISIBILITY) {
    return undefined;
  }
  return mapper(valid);
}

function visibleRenderPoint(point: RenderPoint | undefined): RenderPoint | undefined {
  return point && point.visibility >= MIN_VISIBILITY ? point : undefined;
}

function validMotionPoint(value: unknown): MotionPoint | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }

  const point = value as Partial<MotionPoint>;
  if (typeof point.x !== "number" || !Number.isFinite(point.x)) {
    return undefined;
  }
  if (typeof point.y !== "number" || !Number.isFinite(point.y)) {
    return undefined;
  }
  if (
    point.visibility !== undefined &&
    (typeof point.visibility !== "number" || !Number.isFinite(point.visibility))
  ) {
    return undefined;
  }

  return point as MotionPoint;
}

function visibilityOf(point: MotionPoint): number {
  return clamp(point.visibility ?? 1, 0, 1);
}

function emptyFrame(): MotionFrame {
  return { pose: [], leftHand: [], rightHand: [], face: [] };
}

function midpoint(first: RenderPoint, second: RenderPoint): RenderPoint {
  return {
    x: (first.x + second.x) / 2,
    y: (first.y + second.y) / 2,
    visibility: Math.min(first.visibility, second.visibility)
  };
}

function lerpPoint(first: RenderPoint, second: RenderPoint, amount: number): RenderPoint {
  return {
    x: lerp(first.x, second.x, amount),
    y: lerp(first.y, second.y, amount),
    visibility: lerp(first.visibility, second.visibility, amount)
  };
}

function distance(first: RenderPoint, second: RenderPoint): number {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

function lerp(from: number, to: number, amount: number): number {
  return from + (to - from) * amount;
}

function constrain(value: number, center: number, radius: number): number {
  return clamp(value, center - radius, center + radius);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function format(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function setCirclePosition(circle: SVGCircleElement, point: RenderPoint): void {
  setAttributes(circle, { cx: format(point.x), cy: format(point.y) });
}

function createSvgElement<K extends keyof SVGElementTagNameMap>(
  tagName: K,
  attributes: Record<string, string> = {}
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NAMESPACE, tagName);
  setAttributes(element, attributes);
  return element;
}

function setAttributes(element: Element, attributes: Record<string, string>): void {
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, value);
  }
}

function appendLinearGradient(
  defs: SVGDefsElement,
  id: string,
  x1: string,
  y1: string,
  x2: string,
  y2: string,
  stops: ReadonlyArray<readonly [string, string]>
): void {
  const gradient = createSvgElement("linearGradient", { id, x1, y1, x2, y2 });
  for (const [offset, color] of stops) {
    gradient.append(createSvgElement("stop", { offset, "stop-color": color }));
  }
  defs.append(gradient);
}
