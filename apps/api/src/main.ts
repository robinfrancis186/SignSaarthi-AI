import { buildServer } from "./server.js";

const app = buildServer({ requireClientHeader: true });
const port = Number(process.env.API_PORT ?? 8787);

app
  .listen({ port, host: "127.0.0.1" })
  .then(() => {
    app.log.info(`SignSaarthi API listening on http://127.0.0.1:${port}`);
  })
  .catch((error: unknown) => {
    app.log.error(error);
    process.exit(1);
  });
