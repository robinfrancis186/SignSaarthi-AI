import { fetchJson } from "./lib/script-utils";

type DatasetInfo = {
  id?: string;
  gated?: boolean | string;
  private?: boolean;
  cardData?: {
    license?: string;
  };
};

type WhoAmI = {
  name?: string;
  auth?: {
    type?: string;
    accessToken?: {
      displayName?: string;
      role?: string;
      fineGrained?: {
        canReadGatedRepos?: boolean;
      };
    };
  };
};

const token = process.env["HF_TOKEN"] ?? process.env["HUGGINGFACE_TOKEN"];
const datasetUrl = "https://huggingface.co/datasets/Exploration-Lab/iSign";
const probeFileUrl = "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/iSign_v1.1.csv";
const init: RequestInit = token ? { headers: { Authorization: `Bearer ${token}` } } : {};

try {
  const info = await fetchJson<DatasetInfo>("https://huggingface.co/api/datasets/Exploration-Lab/iSign", init);
  const identity = token
    ? await fetchJson<WhoAmI>("https://huggingface.co/api/whoami-v2", init).catch(() => undefined)
    : undefined;
  const fileProbe = await fetch(probeFileUrl, {
    ...init,
    headers: {
      ...(init.headers instanceof Headers ? Object.fromEntries(init.headers.entries()) : init.headers),
      Range: "bytes=0-255"
    }
  });
  const hasAcceptedFileAccess = fileProbe.ok || fileProbe.status === 206;

  console.log(
    JSON.stringify(
      {
        dataset: "Exploration-Lab/iSign",
        url: datasetUrl,
        access: hasAcceptedFileAccess ? "usable" : "metadata-only",
        gated: info.gated ?? "unknown",
        private: info.private ?? false,
        license: info.cardData?.license ?? "unknown",
        authenticated: Boolean(token),
        identity: identity
          ? {
              user: identity.name ?? "unknown",
              tokenName: identity.auth?.accessToken?.displayName ?? "unknown",
              tokenRole: identity.auth?.accessToken?.role ?? "unknown",
              canReadGatedRepos: identity.auth?.accessToken?.fineGrained?.canReadGatedRepos ?? "unknown"
            }
          : undefined,
        fileProbe: {
          path: "iSign_v1.1.csv",
          status: fileProbe.status,
          ok: hasAcceptedFileAccess
        },
        nextStep:
          hasAcceptedFileAccess
            ? "iSign file access is available. Use this HF_TOKEN for iSign download/training jobs."
            : identity?.auth?.accessToken?.fineGrained?.canReadGatedRepos === false
              ? "Create or update the Hugging Face token so it can read gated repositories, then rerun with HF_TOKEN."
            : !token
              ? "Accept the dataset terms in a Hugging Face browser session, create a read token, then rerun with HF_TOKEN."
              : "This token is present but cannot read iSign files. Confirm the same Hugging Face user accepted the dataset terms."
      },
      null,
      2
    )
  );
} catch (error) {
  console.log(
    JSON.stringify(
      {
        dataset: "Exploration-Lab/iSign",
        url: datasetUrl,
        access: "blocked",
        authenticated: Boolean(token),
        reason: error instanceof Error ? error.message : String(error),
        nextStep:
          "iSign is gated. Open the dataset page, request/accept access as the signed-in Hugging Face user, then set HF_TOKEN locally."
      },
      null,
      2
    )
  );
}
