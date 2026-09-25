import { AreaSeries, ColorType, createChart, LineStyle, type IChartApi, type ISeriesApi } from "lightweight-charts";
import { useEffect, useRef } from "react";

import type { PricePoint } from "@/lib/marketData";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function withAlpha(hex: string, alpha: number): string {
  const match = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!match) return hex;
  const n = parseInt(match[1]!, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

interface Props {
  points: PricePoint[];
  /** Changes when the theme flips so colours are re-read from CSS variables. */
  themeKey: string;
  label: string;
}

/** Area chart of daily closes (TradingView Lightweight Charts, loaded on demand). */
export default function PriceChart({ points, themeKey, label }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    if (!container.current) return;
    const chart = createChart(container.current, {
      autoSize: true,
      handleScroll: false,
      handleScale: false,
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.15, bottom: 0.08 } },
      timeScale: { borderVisible: false, fixLeftEdge: true, fixRightEdge: true },
      crosshair: { vertLine: { style: LineStyle.Dashed }, horzLine: { style: LineStyle.Dashed } },
    });
    chartRef.current = chart;
    seriesRef.current = chart.addSeries(AreaSeries, { lineWidth: 2, priceLineVisible: false });
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;
    const rising = (points[points.length - 1]?.value ?? 0) >= (points[0]?.value ?? 0);
    // A-share convention: red for a rise, green for a fall.
    const color = cssVar(rising ? "--up" : "--down") || (rising ? "#cf3528" : "#10805a");
    chart.applyOptions({
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: cssVar("--muted"),
        fontFamily: getComputedStyle(document.body).fontFamily,
        fontSize: 11,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: withAlpha(cssVar("--line") || "#d9e0eb", 0.7) } },
    });
    series.applyOptions({ lineColor: color, topColor: withAlpha(color, 0.28), bottomColor: withAlpha(color, 0.02) });
    series.setData(points);
    chart.timeScale().fitContent();
  }, [points, themeKey]);

  return <div ref={container} role="img" aria-label={label} className="h-44 w-full sm:h-52" />;
}
