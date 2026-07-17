import { getFingerspellingLetters } from "@signsaarthi/avatar-engine";
import alphabetLibraryData from "../assets/motion/islrtc-alphabet-library.json";
import type { MotionClip, MotionFrame, MotionPoint } from "./motionAvatarRenderer";

type AlphabetLibrary = {
  clips: MotionClip[];
};

const TRANSITION_FRAME_COUNT = 3;
const alphabetLibrary = alphabetLibraryData as AlphabetLibrary;

export const alphabetMotionClips: ReadonlyMap<string, MotionClip> = new Map(
  alphabetLibrary.clips.map((clip) => [clip.label.toUpperCase(), clip])
);

export function createFingerspellingMotion(value: string): MotionClip | undefined {
  const letters = getFingerspellingLetters(value);
  if (letters.length === 0) {
    return undefined;
  }

  const sourceClips = letters.map((letter) => alphabetMotionClips.get(letter));
  if (sourceClips.some((clip) => clip === undefined)) {
    return undefined;
  }

  const clips = sourceClips as MotionClip[];
  const frames: MotionFrame[] = [];
  for (const [index, clip] of clips.entries()) {
    const previous = clips[index - 1];
    if (previous) {
      const from = previous.frames.at(-1);
      const to = clip.frames[0];
      if (from && to) {
        for (let transition = 1; transition <= TRANSITION_FRAME_COUNT; transition += 1) {
          frames.push(interpolateFrame(from, to, transition / (TRANSITION_FRAME_COUNT + 1)));
        }
      }
    }
    frames.push(...clip.frames.map(cloneFrame));
  }

  const compactLetters = letters.join("").toLowerCase();
  return {
    id: `islrtc-fingerspell-${compactLetters.slice(0, 40)}-${stableHash(compactLetters)}`,
    label: value,
    fps: clips[0]?.fps ?? 20,
    frames
  };
}

function interpolateFrame(from: MotionFrame, to: MotionFrame, amount: number): MotionFrame {
  return {
    pose: interpolatePoints(from.pose, to.pose, amount),
    leftHand: interpolatePoints(from.leftHand, to.leftHand, amount),
    rightHand: interpolatePoints(from.rightHand, to.rightHand, amount)
  };
}

function interpolatePoints(from: MotionPoint[], to: MotionPoint[], amount: number): MotionPoint[] {
  const length = Math.max(from.length, to.length);
  return Array.from({ length }, (_, index) => {
    const start = from[index] ?? to[index] ?? { x: 0.5, y: 0.5 };
    const end = to[index] ?? from[index] ?? start;
    const point: MotionPoint = {
      x: start.x + (end.x - start.x) * amount,
      y: start.y + (end.y - start.y) * amount
    };
    const startVisibility = start.visibility;
    const endVisibility = end.visibility;
    if (startVisibility !== undefined || endVisibility !== undefined) {
      point.visibility =
        (startVisibility ?? endVisibility ?? 1) +
        ((endVisibility ?? startVisibility ?? 1) - (startVisibility ?? endVisibility ?? 1)) * amount;
    }
    return point;
  });
}

function cloneFrame(frame: MotionFrame): MotionFrame {
  return {
    pose: frame.pose.map((point) => ({ ...point })),
    leftHand: frame.leftHand.map((point) => ({ ...point })),
    rightHand: frame.rightHand.map((point) => ({ ...point }))
  };
}

function stableHash(value: string): string {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash.toString(36);
}
