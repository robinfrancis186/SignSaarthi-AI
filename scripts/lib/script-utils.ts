import { statfsSync } from "node:fs";

export function readBooleanEnv(name: string, fallback = false): boolean {
  const raw = process.env[name];
  if (raw === undefined) {
    return fallback;
  }
  return ["1", "true", "yes", "on"].includes(raw.toLowerCase());
}

export function readNumberEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number, received ${raw}`);
  }
  return value;
}

export function readListEnv(name: string, fallback: string[]): string[] {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function bytesToGiB(bytes: number): number {
  return Math.round((bytes / 1024 ** 3) * 100) / 100;
}

export function getFreeBytes(path: string): number {
  const stats = statfsSync(path);
  return Number(stats.bavail) * Number(stats.bsize);
}

export async function fetchJson<T>(url: URL | string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`Request failed for ${url.toString()}: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}
