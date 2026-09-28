import { useEffect, useState } from "react";

/**
 * Whole seconds since `startedAt` (a `performance.now()` timestamp), ticking once a second while
 * `active`. Whole seconds keep the counter calm: it changes once a second and never flickers.
 */
export function useElapsed(startedAt: number, active = true): number {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!active) return;
    // Align ticks with the second boundaries of this run so the counter advances evenly.
    const tick = () => setNow(performance.now());
    let interval: number | undefined;
    const offset = 1000 - ((performance.now() - startedAt) % 1000);
    const timeout = window.setTimeout(() => {
      tick();
      interval = window.setInterval(tick, 1000);
    }, offset);
    return () => {
      window.clearTimeout(timeout);
      if (interval !== undefined) window.clearInterval(interval);
    };
  }, [startedAt, active]);
  return Math.max(0, Math.floor((now - startedAt) / 1000));
}
