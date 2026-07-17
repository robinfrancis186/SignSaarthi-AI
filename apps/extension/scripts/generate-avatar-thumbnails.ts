import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { JSDOM } from "jsdom";
import alphabetLibrary from "../src/assets/motion/islrtc-alphabet-library.json";
import {
  createMotionAvatarRenderer,
  type MotionAvatarIdentity,
  type MotionClip
} from "../src/content/motionAvatarRenderer";

const identities: MotionAvatarIdentity[] = ["Ananya", "Arjun", "Meera", "Kabir"];
const sourceClips = alphabetLibrary.clips as MotionClip[];
const outputDirectory = resolve(process.argv[2] ?? "/tmp/signsaarthi-avatar-assets");

await mkdir(outputDirectory, { recursive: true });

const dom = new JSDOM("<!doctype html><html><body><div id=\"host\"></div></body></html>", {
  pretendToBeVisual: true
});
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document
});

for (const [index, identity] of identities.entries()) {
  const host = dom.window.document.querySelector<HTMLDivElement>("#host");
  if (!host) {
    throw new Error("Avatar thumbnail host was not created.");
  }
  host.replaceChildren();

  const controller = createMotionAvatarRenderer(host, { avatarName: identity });
  const clip = sourceClips[(index * 6 + 2) % sourceClips.length];
  if (!clip) {
    throw new Error("The official alphabet library has no motion clips.");
  }
  controller.loadClip(clip);

  const svg = host.querySelector<SVGSVGElement>("svg");
  if (!svg) {
    throw new Error(`The ${identity} renderer did not produce an SVG.`);
  }

  svg.setAttribute("width", "512");
  svg.setAttribute("height", "512");
  svg.setAttribute("viewBox", "0 10 320 320");
  svg.setAttribute("style", "display:block;background:#101827");
  svg.removeAttribute("aria-labelledby");
  svg.setAttribute("aria-label", `${identity} SignSaarthi motion avatar`);

  const background = dom.window.document.createElementNS("http://www.w3.org/2000/svg", "rect");
  background.setAttribute("x", "0");
  background.setAttribute("y", "10");
  background.setAttribute("width", "320");
  background.setAttribute("height", "320");
  background.setAttribute("fill", "#101827");
  const defs = svg.querySelector("defs");
  defs?.after(background);

  await writeFile(
    resolve(outputDirectory, `avatar-${identity.toLowerCase()}.svg`),
    `<?xml version="1.0" encoding="UTF-8"?>\n${svg.outerHTML}\n`,
    "utf8"
  );
  controller.destroy();
}
