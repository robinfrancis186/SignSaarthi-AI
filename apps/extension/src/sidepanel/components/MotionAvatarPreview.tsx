import type { AvatarQueueStep, MotionClip as RuntimeMotionClip } from "@signsaarthi/shared";
import type { ReactElement } from "react";
import { useEffect, useRef, useState } from "react";
import type { MotionAvatarIdentity, MotionClip } from "../../content/motionAvatarRenderer";
import { createMotionQueuePlayer, type MotionQueuePlayer } from "../../content/motionQueuePlayer";
import { loadMotionClipMap } from "../../lib/motionLibraryLoader";

type MotionAvatarPreviewProps = {
  avatarName: MotionAvatarIdentity;
  steps?: readonly AvatarQueueStep[] | undefined;
  runtimeMotionClips?: readonly RuntimeMotionClip[] | undefined;
  paused: boolean;
  replayKey: number;
  speed: number;
  onStepChange?: ((index: number) => void) | undefined;
};

export function MotionAvatarPreview({
  avatarName,
  steps,
  runtimeMotionClips,
  paused,
  replayKey,
  speed,
  onStepChange
}: MotionAvatarPreviewProps): ReactElement {
  const hostRef = useRef<HTMLDivElement>(null);
  const [motionClips, setMotionClips] = useState<ReadonlyMap<string, MotionClip>>();
  const [motionLibraryState, setMotionLibraryState] = useState<"loading" | "ready" | "error">(
    "loading"
  );
  const playerRef = useRef<MotionQueuePlayer | undefined>(undefined);
  const loadedStepsRef = useRef<readonly AvatarQueueStep[] | undefined>(undefined);
  const onStepChangeRef = useRef(onStepChange);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    let cancelled = false;
    void loadMotionClipMap().then(
      (clips) => {
        if (!cancelled) {
          const merged = new Map(clips);
          for (const clip of runtimeMotionClips ?? []) {
            merged.set(clip.id, clip as MotionClip);
          }
          setMotionClips(merged);
          setMotionLibraryState("ready");
        }
      },
      () => {
        if (!cancelled) {
          setMotionLibraryState("error");
        }
      }
    );
    return () => {
      cancelled = true;
    };
  }, [runtimeMotionClips]);

  useEffect(() => {
    onStepChangeRef.current = onStepChange;
  }, [onStepChange]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !motionClips) {
      return;
    }
    const player = createMotionQueuePlayer(host, motionClips, {
      avatarName,
      onStepChange(index) {
        onStepChangeRef.current?.(index);
      }
    });
    playerRef.current = player;
    loadedStepsRef.current = undefined;
    return () => {
      player.destroy();
      if (playerRef.current === player) {
        playerRef.current = undefined;
        loadedStepsRef.current = undefined;
      }
    };
  }, [avatarName, motionClips]);

  useEffect(() => {
    playerRef.current?.setSpeed(speed);
  }, [speed]);

  useEffect(() => {
    const player = playerRef.current;
    if (!player) {
      return;
    }

    const nextSteps = steps ?? [];
    const loadedSteps = loadedStepsRef.current;
    if (loadedSteps && hasPlaybackCompatiblePrefix(loadedSteps, nextSteps)) {
      const appendedSteps = nextSteps.slice(loadedSteps.length);
      if (appendedSteps.length > 0) {
        player.enqueue(appendedSteps);
      }
    } else {
      player.load(nextSteps, speed);
    }
    loadedStepsRef.current = [...nextSteps];

    if (!pausedRef.current) {
      player.play();
    }
  }, [avatarName, motionClips, speed, steps]);

  useEffect(() => {
    const player = playerRef.current;
    if (!player) {
      return;
    }
    if (paused) {
      player.pause();
    } else {
      player.play();
    }
  }, [paused]);

  useEffect(() => {
    if (replayKey > 0) {
      playerRef.current?.replay(!pausedRef.current);
    }
  }, [replayKey]);

  return (
    <div className="sidepanel-motion-avatar-shell" data-motion-library-state={motionLibraryState}>
      <div
        className="sidepanel-motion-avatar"
        data-avatar-name={avatarName}
        data-motion-step-count={steps?.length ?? 0}
        ref={hostRef}
      />
      {motionLibraryState === "error" ? (
        <p className="motion-library-error" role="status">
          Motion unavailable. Word actions remain visible.
        </p>
      ) : null}
    </div>
  );
}

function hasPlaybackCompatiblePrefix(
  loadedSteps: readonly AvatarQueueStep[],
  nextSteps: readonly AvatarQueueStep[]
): boolean {
  if (loadedSteps.length > nextSteps.length) {
    return false;
  }

  return loadedSteps.every((loadedStep, index) => {
    const nextStep = nextSteps[index];
    return Boolean(
      nextStep &&
        loadedStep.id === nextStep.id &&
        loadedStep.glossItemId === nextStep.glossItemId &&
        loadedStep.kind === nextStep.kind &&
        loadedStep.label === nextStep.label &&
        loadedStep.signAssetId === nextStep.signAssetId &&
        loadedStep.motionClipId === nextStep.motionClipId &&
        loadedStep.motionSource === nextStep.motionSource &&
        loadedStep.fingerspellingSource === nextStep.fingerspellingSource &&
        loadedStep.expertReviewed === nextStep.expertReviewed &&
        loadedStep.confidence === nextStep.confidence
    );
  });
}
