import "@testing-library/jest-dom/vitest";

import { configure } from "@testing-library/react";

// (round 13, I17) findBy* / waitFor wait up to 10 s (default 1 s): on a loaded host a re-render after a mocked
// response can take longer than a second. They still return as soon as the expected state is there.
configure({ asyncUtilTimeout: 10_000 });
